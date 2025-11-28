import Foundation
import MLX
import MLXNN

public class MultiModalProjector: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    let inputDim: Int
    let outputDim: Int
    let hiddenAct: String

    public init(config: PaddleOCRVLConfig) {
        self.inputDim = config.visionConfig.hiddenSize
        self.outputDim = config.textConfig.hiddenSize
        self.hiddenAct = config.visionConfig.projectorHiddenAct

        self._linear1.wrappedValue = Linear(inputDim, outputDim)
        self._linear2.wrappedValue = Linear(outputDim, outputDim)

        super.init()
    }

    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        var x = linear1(features)
        x = gelu(x)
        x = linear2(x)
        return x
    }
}
