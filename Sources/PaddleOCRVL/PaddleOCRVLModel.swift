import Foundation
import MLX
import MLXNN

public class PaddleOCRVLModel: Module {
    @ModuleInfo(key: "vision_model") public var visionModel: NaViTVisionEncoder
    @ModuleInfo(key: "multi_modal_projector") var projector: MultiModalProjector
    @ModuleInfo(key: "model") public var languageModel: ERNIEModelInner
    @ModuleInfo(key: "lm_head") public var lmHead: Linear

    let config: PaddleOCRVLConfig

    public init(config: PaddleOCRVLConfig) {
        self.config = config

        self._visionModel.wrappedValue = NaViTVisionEncoder(config: config.visionConfig)
        self._projector.wrappedValue = MultiModalProjector(config: config)
        self._languageModel.wrappedValue = ERNIEModelInner(config: config.textConfig)
        self._lmHead.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            config.textConfig.vocabSize,
            bias: false
        )

        super.init()
    }

    public func getImageFeatures(_ pixelValues: MLXArray) -> MLXArray {
        let visionFeatures = visionModel.getImageFeatures(pixelValues)
        return projector(visionFeatures)
    }

    public func mergeInputIdsWithImageFeatures(
        inputIds: MLXArray,
        imageFeatures: MLXArray
    ) -> MLXArray {
        let inputsEmbeds = languageModel.getEmbedding(inputIds)
        let imageTokenMask = inputIds .== config.visionTokenId

        return mergeEmbeddings(
            inputsEmbeds: inputsEmbeds,
            imageFeatures: imageFeatures,
            mask: imageTokenMask
        )
    }

    private func mergeEmbeddings(
        inputsEmbeds: MLXArray,
        imageFeatures: MLXArray,
        mask: MLXArray
    ) -> MLXArray {
        let seqLen = inputsEmbeds.dim(1)
        let numImageTokens = imageFeatures.dim(1)

        let maskInt = mask.asType(.int32)
        let firstTrueIdxArray = argMax(maskInt, axis: 1)
        let firstTrueIdx = firstTrueIdxArray[0].item(Int.self)

        let prePad = firstTrueIdx
        let postPad = seqLen - firstTrueIdx - numImageTokens

        var alignedImageFeatures = imageFeatures
        if prePad > 0 || postPad > 0 {
            let paddingWidths: [IntOrPair] = [[0, 0], [prePad, max(0, postPad)], [0, 0]]
            alignedImageFeatures = padded(imageFeatures, widths: paddingWidths)
        }

        if alignedImageFeatures.dim(1) > seqLen {
            alignedImageFeatures = alignedImageFeatures[0..., 0..<seqLen, 0...]
        } else if alignedImageFeatures.dim(1) < seqLen {
            let extraPad = seqLen - alignedImageFeatures.dim(1)
            let extraPadWidths: [IntOrPair] = [[0, 0], [0, extraPad], [0, 0]]
            alignedImageFeatures = padded(alignedImageFeatures, widths: extraPadWidths)
        }

        let expandedMask = mask.expandedDimensions(axis: -1)

        eval(expandedMask)
        eval(alignedImageFeatures)
        eval(inputsEmbeds)

        return MLX.which(expandedMask, alignedImageFeatures, inputsEmbeds)
    }

    public func forward(
        inputIds: MLXArray,
        pixelValues: MLXArray?,
        cache: [KVCache]?
    ) -> MLXArray {
        var inputsEmbeds: MLXArray

        if let pixelValues = pixelValues {
            let imageFeatures = getImageFeatures(pixelValues)
            eval(imageFeatures)
            inputsEmbeds = mergeInputIdsWithImageFeatures(
                inputIds: inputIds,
                imageFeatures: imageFeatures
            )
        } else {
            inputsEmbeds = languageModel.getEmbedding(inputIds)
        }

        let hiddenStates = languageModel.forward(inputsEmbeds, cache: cache)
        return lmHead(hiddenStates)
    }

    public func forwardGeneration(
        inputIds: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray {
        let embeds = languageModel.getEmbedding(inputIds)
        let hiddenStates = languageModel.forward(embeds, cache: cache)
        return lmHead(hiddenStates)
    }

    public func newCache() -> [KVCache] {
        (0..<config.textConfig.numHiddenLayers).map { _ in KVCache() }
    }
}

extension PaddleOCRVLModel {
    public static func load(from directory: URL) throws -> PaddleOCRVLModel {
        let config = try PaddleOCRVLConfig.load(from: directory)
        let model = PaddleOCRVLModel(config: config)
        try loadWeights(for: model, from: directory)
        return model
    }

    private static func loadWeights(for model: PaddleOCRVLModel, from directory: URL) throws {
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }

        if safetensorFiles.isEmpty {
            throw PaddleOCRVLError.modelLoadFailed("No safetensors files found in \(directory.path)")
        }

        var allWeights: [String: MLXArray] = [:]

        for file in safetensorFiles {
            let weights = try MLX.loadArrays(url: file)
            for (key, value) in weights {
                allWeights[key] = value
            }
        }

        let sanitizedWeights = sanitizeWeights(allWeights)
        let parameters = ModuleParameters.unflattened(sanitizedWeights)
        try model.update(parameters: parameters, verify: .noUnusedKeys)

        loadSpecialWeights(for: model, from: allWeights)
    }

    private static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key
            var adjustedValue = value

            if key.hasPrefix("visual.") {
                newKey = key.replacingOccurrences(of: "visual.", with: "vision_model.")
            }

            if key.contains("rotary_emb.inv_freq") {
                continue
            }

            if key.contains("conv") || key.contains("projection") || key.contains("patch_embed"),
               key.contains("weight"),
               value.ndim == 4 {
                let shape = value.shape
                if shape[1] != shape[2] && shape[2] == shape[3] {
                    adjustedValue = value.transposed(0, 2, 3, 1)
                }
            }

            result[newKey] = adjustedValue
        }

        return result
    }

    private static func loadSpecialWeights(
        for model: PaddleOCRVLModel,
        from weights: [String: MLXArray]
    ) {
        if let posEmbed = weights["visual.pos_embed"] ?? weights["vision_model.embeddings.position_embedding.weight"] {
            model.visionModel.positionEmbedding = posEmbed
        }

        if let clsEmbed = weights["visual.cls_token"] ?? weights["vision_model.embeddings.class_embedding"] {
            model.visionModel.classEmbedding = clsEmbed.expandedDimensions(axis: 0)
        }
    }
}
