import Foundation
import MLX
import MLXNN
import MLXFast

public class VisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    public init(hiddenSize: Int, intermediateSize: Int) {
        self._fc1.wrappedValue = Linear(hiddenSize, intermediateSize)
        self._fc2.wrappedValue = Linear(intermediateSize, hiddenSize)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

public class VisionAttention: Module {
    let numHeads: Int
    let scale: Float
    let headDim: Int

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    public init(config: PaddleOCRVLVisionConfig) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)

        self._qkv.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 3, bias: true)
        self._proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)

        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let (batchSize, seqLen, _) = (
            hiddenStates.dim(0),
            hiddenStates.dim(1),
            hiddenStates.dim(2)
        )

        var qkvOut = qkv(hiddenStates)
        qkvOut = qkvOut.reshaped(batchSize, seqLen, 3, numHeads, headDim)
        qkvOut = qkvOut.transposed(2, 0, 3, 1, 4)

        let q = qkvOut[0]
        let k = qkvOut[1]
        let v = qkvOut[2]

        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: scale, mask: .none
        )

        let output = attnOutput
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, seqLen, -1)

        return proj(output)
    }
}

public class VisionEncoderLayer: Module {
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "self_attn") var attn: VisionAttention
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: VisionMLP

    public init(config: PaddleOCRVLVisionConfig) {
        self._layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._attn.wrappedValue = VisionAttention(config: config)
        self._layerNorm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._mlp.wrappedValue = VisionMLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let residual = hiddenStates
        var h = layerNorm1(hiddenStates)
        h = attn(h)
        h = residual + h
        h = h + mlp(layerNorm2(h))
        return h
    }
}

public class PatchEmbedding: Module {
    @ModuleInfo(key: "proj") public var projection: Conv2d

    let patchSize: Int
    let hiddenSize: Int

    public init(config: PaddleOCRVLVisionConfig) {
        self.patchSize = config.patchSize
        self.hiddenSize = config.hiddenSize

        self._projection.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize)
        )
    }

    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var patches = projection(pixelValues)
        let (b, h, w, c) = (patches.dim(0), patches.dim(1), patches.dim(2), patches.dim(3))
        patches = patches.reshaped(b, h * w, c)
        return patches
    }
}

public class NaViTVisionEncoder: Module {
    @ModuleInfo(key: "patch_embed") public var patchEmbed: PatchEmbedding
    @ModuleInfo(key: "layers") var layers: [VisionEncoderLayer]
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

    var positionEmbedding: MLXArray?
    var classEmbedding: MLXArray?

    let config: PaddleOCRVLVisionConfig
    let patchSize: Int
    let hiddenSize: Int

    public init(config: PaddleOCRVLVisionConfig) {
        self.config = config
        self.patchSize = config.patchSize
        self.hiddenSize = config.hiddenSize

        self._patchEmbed.wrappedValue = PatchEmbedding(config: config)

        var visionLayers: [VisionEncoderLayer] = []
        for _ in 0..<config.numHiddenLayers {
            visionLayers.append(VisionEncoderLayer(config: config))
        }
        self._layers.wrappedValue = visionLayers

        self._postLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)

        let numPatches = (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
        self.positionEmbedding = MLXArray.zeros([1, numPatches + 1, config.hiddenSize])
        self.classEmbedding = MLXArray.zeros([1, 1, config.hiddenSize])

        super.init()
    }

    private func get2DPositionalEmbedding(height: Int, width: Int) -> MLXArray {
        let numPatches = height * width
        if let posEmbed = positionEmbedding {
            let origNumPatches = posEmbed.dim(1) - 1
            if numPatches == origNumPatches {
                return posEmbed[0..., 1..., 0...]
            }

            let origSize = Int(sqrt(Double(origNumPatches)))
            let patchPosEmbed = posEmbed[0..., 1..., 0...]
                .reshaped(1, origSize, origSize, hiddenSize)

            let scaleH = Float(height) / Float(origSize)
            let scaleW = Float(width) / Float(origSize)
            let upsample = Upsample(scaleFactor: [scaleH, scaleW], mode: .cubic(alignCorners: false))
            let resized = upsample(patchPosEmbed)
            return resized.reshaped(1, numPatches, hiddenSize)
        }
        return MLXArray.zeros([1, numPatches, hiddenSize])
    }

    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let batchSize = pixelValues.dim(0)
        let height = pixelValues.dim(1) / patchSize
        let width = pixelValues.dim(2) / patchSize

        var hiddenStates = patchEmbed(pixelValues)

        let posEmbed = get2DPositionalEmbedding(height: height, width: width)
        hiddenStates = hiddenStates + posEmbed

        if let clsEmbed = classEmbedding {
            let clsTokens = broadcast(clsEmbed, to: [batchSize, 1, hiddenSize])
            hiddenStates = concatenated([clsTokens, hiddenStates], axis: 1)

            if let fullPosEmbed = positionEmbedding {
                let clsPosEmbed = fullPosEmbed[0..., ..<1, 0...]
                let clsPosEmbedBroadcast = broadcast(clsPosEmbed, to: [batchSize, 1, hiddenSize])
                hiddenStates[0..., ..<1, 0...] = hiddenStates[0..., ..<1, 0...] + clsPosEmbedBroadcast
            }
        }

        for layer in layers {
            hiddenStates = layer(hiddenStates)
        }

        hiddenStates = postLayerNorm(hiddenStates)

        return hiddenStates
    }

    public func getImageFeatures(_ pixelValues: MLXArray) -> MLXArray {
        let features = callAsFunction(pixelValues)
        return features[0..., 1..., 0...]
    }
}
