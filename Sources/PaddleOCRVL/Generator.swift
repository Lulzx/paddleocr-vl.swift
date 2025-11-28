import Foundation
import MLX
import Tokenizers

public class PaddleOCRVLGenerator {
    let model: PaddleOCRVLModel
    let tokenizer: any Tokenizer
    let config: PaddleOCRVLConfig

    private let eosTokenId: Int
    private let visionTokenId: Int
    private let visionStartTokenId: Int
    private let visionEndTokenId: Int
    private let stopTokenIds: Set<Int>

    public init(model: PaddleOCRVLModel, tokenizer: any Tokenizer, config: PaddleOCRVLConfig) {
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.visionTokenId = config.visionTokenId
        self.visionStartTokenId = config.visionStartTokenId
        self.visionEndTokenId = config.visionEndTokenId
        self.eosTokenId = tokenizer.eosTokenId ?? 1
        self.stopTokenIds = [eosTokenId]
    }

    public func buildPrompt(task: PaddleOCRTask = .ocr) -> String {
        task.prompt
    }

    public func generate(
        processedImages: ProcessedImages,
        task: PaddleOCRTask = .ocr,
        maxNewTokens: Int = 1024,
        temperature: Float = 0.0,
        topP: Float = 1.0
    ) -> GenerationResult {
        let textPrompt = buildPrompt(task: task)
        let textIds = tokenizer.encode(text: textPrompt, addSpecialTokens: false)

        let numImageTokens = processedImages.numImageTokens

        var inputIds: [Int] = []

        if let bosId = tokenizer.bosTokenId {
            inputIds.append(bosId)
        }

        inputIds.append(visionStartTokenId)
        inputIds.append(contentsOf: Array(repeating: visionTokenId, count: numImageTokens))
        inputIds.append(visionEndTokenId)
        inputIds.append(contentsOf: textIds)

        var inputIdArray = MLXArray(inputIds.map { Int32($0) }).reshaped(1, -1)

        let cache = model.newCache()

        var logits = model.forward(
            inputIds: inputIdArray,
            pixelValues: processedImages.pixelValues,
            cache: cache
        )

        var generatedTokens: [Int] = []
        var generatedText = ""

        for _ in 0..<maxNewTokens {
            let lastLogits = logits[0, -1]

            let nextTokenId: Int
            if temperature <= 0 {
                nextTokenId = argMax(lastLogits).item(Int.self)
            } else {
                nextTokenId = sampleWithTemperature(
                    logits: lastLogits,
                    temperature: temperature,
                    topP: topP
                )
            }

            if stopTokenIds.contains(nextTokenId) {
                break
            }

            generatedTokens.append(nextTokenId)

            let decoded = tokenizer.decode(tokens: [nextTokenId])
            generatedText += decoded

            inputIdArray = MLXArray([Int32(nextTokenId)]).reshaped(1, 1)
            logits = model.forwardGeneration(inputIds: inputIdArray, cache: cache)
        }

        return GenerationResult(
            tokens: generatedTokens,
            text: generatedText,
            tokenCount: generatedTokens.count
        )
    }

    public func generate(
        pixelValues: MLXArray,
        task: PaddleOCRTask = .ocr,
        maxNewTokens: Int = 1024,
        temperature: Float = 0.0,
        topP: Float = 1.0
    ) -> GenerationResult {
        let textPrompt = buildPrompt(task: task)
        let textIds = tokenizer.encode(text: textPrompt, addSpecialTokens: false)

        let imageHeight = pixelValues.dim(1)
        let imageWidth = pixelValues.dim(2)
        let patchSize = config.visionConfig.patchSize
        let numImageTokens = (imageHeight / patchSize) * (imageWidth / patchSize)

        var inputIds: [Int] = []

        if let bosId = tokenizer.bosTokenId {
            inputIds.append(bosId)
        }

        inputIds.append(visionStartTokenId)
        inputIds.append(contentsOf: Array(repeating: visionTokenId, count: numImageTokens))
        inputIds.append(visionEndTokenId)
        inputIds.append(contentsOf: textIds)

        var inputIdArray = MLXArray(inputIds.map { Int32($0) }).reshaped(1, -1)

        let cache = model.newCache()

        var logits = model.forward(
            inputIds: inputIdArray,
            pixelValues: pixelValues,
            cache: cache
        )

        var generatedTokens: [Int] = []
        var generatedText = ""

        for _ in 0..<maxNewTokens {
            let lastLogits = logits[0, -1]

            let nextTokenId: Int
            if temperature <= 0 {
                nextTokenId = argMax(lastLogits).item(Int.self)
            } else {
                nextTokenId = sampleWithTemperature(
                    logits: lastLogits,
                    temperature: temperature,
                    topP: topP
                )
            }

            if stopTokenIds.contains(nextTokenId) {
                break
            }

            generatedTokens.append(nextTokenId)

            let decoded = tokenizer.decode(tokens: [nextTokenId])
            generatedText += decoded

            inputIdArray = MLXArray([Int32(nextTokenId)]).reshaped(1, 1)
            logits = model.forwardGeneration(inputIds: inputIdArray, cache: cache)
        }

        return GenerationResult(
            tokens: generatedTokens,
            text: generatedText,
            tokenCount: generatedTokens.count
        )
    }

    private func sampleWithTemperature(logits: MLXArray, temperature: Float, topP: Float) -> Int {
        let scaledLogits = logits / temperature

        if topP < 1.0 {
            let probs = softmax(scaledLogits, axis: -1)
            let sortedIndices = argSort(probs, axis: -1)
            let sortedProbs = take(probs, sortedIndices, axis: -1).squeezed(axis: 0)
            let cumulativeProbs = cumsum(sortedProbs, axis: -1)

            let topProbs = MLX.where(
                cumulativeProbs .> (1 - topP), sortedProbs, zeros(like: sortedProbs))

            let sortedToken = categorical(log(topProbs + 1e-10))
            let token = sortedIndices.squeezed(axis: 0)[sortedToken]
            return token.item(Int.self)
        } else {
            let sample = categorical(scaledLogits)
            return sample.item(Int.self)
        }
    }

    public func generateBatch(
        batchImages: BatchProcessedImages,
        task: PaddleOCRTask = .ocr,
        maxNewTokens: Int = 1024,
        temperature: Float = 0.0,
        topP: Float = 1.0
    ) -> BatchGenerationResult {
        let results = batchImages.items.map { processedImages in
            generate(
                processedImages: processedImages,
                task: task,
                maxNewTokens: maxNewTokens,
                temperature: temperature,
                topP: topP
            )
        }
        return BatchGenerationResult(results: results)
    }
}

public struct GenerationResult: Sendable {
    public let tokens: [Int]
    public let text: String
    public let tokenCount: Int
}

public struct BatchGenerationResult: Sendable {
    public let results: [GenerationResult]

    public var texts: [String] {
        results.map { $0.text }
    }

    public var batchSize: Int {
        results.count
    }
}
