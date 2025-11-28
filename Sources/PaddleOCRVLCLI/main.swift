import ArgumentParser
import PaddleOCRVL
import Foundation
import Dispatch
import MLX
import Tokenizers
import Hub
import CoreImage

struct PaddleOCRVLCommand: ParsableCommand {
    static var configuration = CommandConfiguration(
        commandName: "paddleocr-vl",
        abstract: "A command-line tool for PaddleOCR-VL that extracts text from images.",
        version: "1.0.0",
        subcommands: [OCRCommand.self, InfoCommand.self],
        defaultSubcommand: OCRCommand.self
    )
}

PaddleOCRVLCommand.main()

private func getHuggingFaceCacheDirectory() -> URL {
    let env = ProcessInfo.processInfo.environment

    if let hubCache = env["HF_HUB_CACHE"], !hubCache.isEmpty {
        return URL(fileURLWithPath: hubCache)
    }

    if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
        return URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
    }

    let homeDir = FileManager.default.homeDirectoryForCurrentUser
    return homeDir.appendingPathComponent(".cache/huggingface/hub")
}

private func createHubApi() -> HubApi {
    let hfCacheDir = getHuggingFaceCacheDirectory()
    try? FileManager.default.createDirectory(at: hfCacheDir, withIntermediateDirectories: true)
    return HubApi(downloadBase: hfCacheDir)
}

func isHuggingFaceModelId(_ modelSpec: String) -> Bool {
    if modelSpec.hasPrefix("/") || modelSpec.hasPrefix("./") || modelSpec.hasPrefix("../") {
        return false
    }

    let url = URL(fileURLWithPath: modelSpec).standardizedFileURL
    if FileManager.default.fileExists(atPath: url.path) {
        return false
    }

    let baseSpec = modelSpec.split(separator: ":")[0]
    let parts = baseSpec.split(separator: "/")

    guard parts.count == 2 else {
        return false
    }

    let org = String(parts[0])
    let repo = String(parts[1])

    guard !org.isEmpty && !repo.isEmpty else {
        return false
    }

    let pathIndicators = ["models", "model", "weights", "data", "datasets", "checkpoints", "output", "tmp", "temp", "cache"]
    if pathIndicators.contains(org.lowercased()) {
        return false
    }

    if org.filter({ $0 == "." }).count > 1 {
        return false
    }

    return true
}

private func downloadModel(
    hub: HubApi,
    id: String,
    revision: String,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> URL {
    do {
        let repo = Hub.Repo(id: id)
        let modelFiles = ["*.safetensors", "*.json"]
        return try await hub.snapshot(
            from: repo,
            revision: revision,
            matching: modelFiles,
            progressHandler: progressHandler
        )
    } catch Hub.HubClientError.authorizationRequired {
        throw ValidationError("Model '\(id)' not found or requires authentication")
    } catch {
        let nserror = error as NSError
        if nserror.domain == NSURLErrorDomain && nserror.code == NSURLErrorNotConnectedToInternet {
            throw ValidationError("No internet connection. Please check your network or use a local model path.")
        } else {
            throw error
        }
    }
}

private final class MutableBox<T>: @unchecked Sendable {
    var value: T
    init(_ value: T) {
        self.value = value
    }
}

private func blockingAwait<T>(_ operation: @escaping () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let resultBox = MutableBox<Result<T, Error>?>(nil)
    Task {
        do {
            let value = try await operation()
            resultBox.value = .success(value)
        } catch {
            resultBox.value = .failure(error)
        }
        semaphore.signal()
    }
    semaphore.wait()
    guard let result = resultBox.value else {
        fatalError("Snapshot task completed without a result.")
    }
    return try result.get()
}

private func findCachedModel(modelId: String, verbose: Bool, requireWeights: Bool = true) -> URL? {
    let fm = FileManager.default
    let homeDir = fm.homeDirectoryForCurrentUser

    let hfCachePath = homeDir
        .appendingPathComponent(".cache/huggingface/hub/models--\(modelId.replacingOccurrences(of: "/", with: "--"))")
        .appendingPathComponent("snapshots")

    if verbose {
        print("Checking HF cache: \(hfCachePath.path)")
    }

    if fm.fileExists(atPath: hfCachePath.path) {
        if let snapshots = try? fm.contentsOfDirectory(at: hfCachePath, includingPropertiesForKeys: nil) {
            for snapshot in snapshots {
                let configFile = snapshot.appendingPathComponent("config.json")
                if fm.fileExists(atPath: configFile.path) {
                    if requireWeights {
                        let contents = (try? fm.contentsOfDirectory(at: snapshot, includingPropertiesForKeys: nil)) ?? []
                        let hasSafetensors = contents.contains { $0.pathExtension == "safetensors" }
                        if !hasSafetensors {
                            if verbose {
                                print("Found cached model at \(snapshot.path) but missing safetensors weights")
                            }
                            continue
                        }
                    }
                    if verbose {
                        print("Found cached model at: \(snapshot.path)")
                    }
                    return snapshot
                }
            }
        }
    }

    return nil
}

func resolveModelDirectory(
    modelSpec: String,
    verbose: Bool
) throws -> URL {
    let localURL = URL(fileURLWithPath: modelSpec).standardizedFileURL
    if FileManager.default.fileExists(atPath: localURL.path) {
        return localURL
    }

    if !isHuggingFaceModelId(modelSpec) {
        throw ValidationError("Model directory not found: \(modelSpec)")
    }

    let parts = modelSpec.split(separator: ":", maxSplits: 1)
    let modelId = String(parts[0])
    let revision = parts.count > 1 ? String(parts[1]) : "main"

    if verbose {
        print("Resolving HuggingFace model: \(modelId) (revision: \(revision))")
    }

    if let cachedURL = findCachedModel(modelId: modelId, verbose: verbose) {
        return cachedURL
    }

    if verbose {
        print("Model not found in cache. Downloading from HuggingFace Hub...")
    }

    do {
        let snapshotURL = try blockingAwait {
            try await downloadModel(
                hub: createHubApi(),
                id: modelId,
                revision: revision,
                progressHandler: { progress in
                    if verbose {
                        let completed = progress.completedUnitCount
                        let total = progress.totalUnitCount
                        let percent = progress.fractionCompleted * 100
                        let message = String(format: "\rDownloading: %d/%d files (%.0f%%)", completed, total, percent)
                        FileHandle.standardError.write(Data(message.utf8))
                    }
                }
            )
        }

        if verbose {
            print("\nModel ready at: \(snapshotURL.path)")
        }

        return snapshotURL
    } catch {
        throw ValidationError("""
            Failed to download model '\(modelId)' from HuggingFace Hub.
            Error: \(error.localizedDescription)

            You can also download manually using:
              huggingface-cli download \(modelId)
            """)
    }
}

enum CLIProcessingMode: String, ExpressibleByArgument, CaseIterable {
    case base
    case dynamic

    func toProcessingMode() -> ProcessingMode {
        switch self {
        case .base: return .base
        case .dynamic: return .dynamic
        }
    }
}

enum CLITask: String, ExpressibleByArgument, CaseIterable {
    case ocr
    case table
    case formula
    case chart

    func toTask() -> PaddleOCRTask {
        switch self {
        case .ocr: return .ocr
        case .table: return .table
        case .formula: return .formula
        case .chart: return .chart
        }
    }
}

struct OCRCommand: ParsableCommand {
    static var configuration = CommandConfiguration(
        commandName: "ocr",
        abstract: "Perform OCR on one or more image files."
    )

    @Argument(help: "Path(s) to the input image file(s) for OCR.")
    var imagePaths: [String]

    @Option(name: .shortAndLong, help: "Model path or HuggingFace model ID (e.g., PaddlePaddle/PaddleOCR-VL).")
    var model: String = "PaddlePaddle/PaddleOCR-VL"

    @Option(name: .shortAndLong, help: "Optional path to save the OCR output text.")
    var output: String?

    @Option(name: .shortAndLong, help: "Task type: ocr, table, formula, chart.")
    var task: CLITask = .ocr

    @Option(name: .long, help: "Maximum number of tokens to generate.")
    var maxTokens: Int = 1024

    @Option(name: .long, help: "Processing mode: base (448x448), dynamic (NaViT-style).")
    var mode: CLIProcessingMode = .base

    @Flag(name: .long, help: "Verbose output with timing information.")
    var verbose: Bool = false

    @Option(name: .long, help: "GPU memory cache limit in MB (default: unlimited).")
    var cacheLimit: Int?

    func run() throws {
        guard !imagePaths.isEmpty else {
            throw ValidationError("At least one image path is required")
        }

        let effectiveMode = mode.toProcessingMode()
        let effectiveTask = task.toTask()

        try runBlocking(
            imagePaths: imagePaths,
            modelPath: model,
            outputPath: output,
            task: effectiveTask,
            maxTokens: maxTokens,
            mode: effectiveMode,
            verbose: verbose,
            cacheLimit: cacheLimit
        )
    }

    private func runBlocking(
        imagePaths: [String],
        modelPath: String,
        outputPath: String?,
        task: PaddleOCRTask,
        maxTokens: Int,
        mode: ProcessingMode,
        verbose: Bool,
        cacheLimit: Int?
    ) throws {
        let group = DispatchGroup()
        group.enter()

        var capturedError: Error?
        var capturedResults: [String]?

        Task {
            do {
                capturedResults = try await runAsync(
                    imagePaths: imagePaths,
                    modelPath: modelPath,
                    task: task,
                    maxTokens: maxTokens,
                    mode: mode,
                    verbose: verbose,
                    cacheLimit: cacheLimit
                )
            } catch {
                capturedError = error
            }
            group.leave()
        }

        group.wait()

        if let error = capturedError {
            throw error
        }

        guard let results = capturedResults else {
            return
        }

        for (index, result) in results.enumerated() {
            let imagePath = imagePaths[index]

            if let outputPath {
                let actualOutputPath: String
                if results.count > 1 {
                    let ext = URL(fileURLWithPath: outputPath).pathExtension
                    if ext.isEmpty {
                        actualOutputPath = "\(outputPath)_\(index).txt"
                    } else {
                        let base = outputPath.dropLast(ext.count + 1)
                        actualOutputPath = "\(base)_\(index).\(ext)"
                    }
                } else {
                    actualOutputPath = outputPath
                }

                try result.write(toFile: actualOutputPath, atomically: true, encoding: .utf8)
                if verbose {
                    print("[\(index + 1)/\(results.count)] Output saved to: \(actualOutputPath)")
                }
            } else {
                if results.count > 1 {
                    print("=== [\(index + 1)/\(results.count)] \(imagePath) ===")
                }
                print(result)
                if results.count > 1 && index < results.count - 1 {
                    print("")
                }
            }
        }
    }

    private func runAsync(
        imagePaths: [String],
        modelPath: String,
        task: PaddleOCRTask,
        maxTokens: Int,
        mode: ProcessingMode,
        verbose: Bool,
        cacheLimit: Int?
    ) async throws -> [String] {
        for imagePath in imagePaths {
            guard FileManager.default.fileExists(atPath: imagePath) else {
                throw ValidationError("Image file not found: \(imagePath)")
            }

            guard PaddleOCRVLPipeline.isSupportedImage(imagePath) else {
                throw ValidationError("Unsupported image format for \(imagePath). Supported: \(PaddleOCRVLPipeline.supportedFormats.joined(separator: ", "))")
            }
        }

        if let limit = cacheLimit {
            MLX.GPU.set(cacheLimit: limit * 1024 * 1024)
            if verbose {
                print("GPU cache limit set to \(limit)MB")
            }
        }

        let modelURL = try resolveModelDirectory(modelSpec: modelPath, verbose: verbose)

        if verbose {
            print("Loading model from: \(modelPath)")
            print("Task: \(task.rawValue)")
            print("Mode: \(mode.rawValue)")
        }

        let startTime = Date()

        let pipeline = try await PaddleOCRVLPipeline(modelURL: modelURL, mode: mode)

        if verbose {
            let loadTime = Date().timeIntervalSince(startTime)
            print("Model loaded in \(String(format: "%.2f", loadTime))s")
            print("Processing \(imagePaths.count) image(s)...")
        }

        let ocrStartTime = Date()

        let results: [String]
        if imagePaths.count == 1 {
            let result = try pipeline.recognize(imagePath: imagePaths[0], task: task, maxTokens: maxTokens)
            results = [result]
        } else {
            results = try pipeline.recognizeBatch(imagePaths: imagePaths, task: task, maxTokens: maxTokens)
        }

        if verbose {
            let ocrTime = Date().timeIntervalSince(ocrStartTime)
            print("OCR completed in \(String(format: "%.2f", ocrTime))s")
            if imagePaths.count > 1 {
                print("Average time per image: \(String(format: "%.2f", ocrTime / Double(imagePaths.count)))s")
            }
        }

        return results
    }
}

struct InfoCommand: ParsableCommand {
    static var configuration = CommandConfiguration(
        commandName: "info",
        abstract: "Display information about the model and supported formats."
    )

    func run() throws {
        print("PaddleOCR-VL CLI v1.0.0")
        print("")
        print("Model: PaddlePaddle/PaddleOCR-VL (0.9B)")
        print("  - Ultra-compact vision-language model for document parsing")
        print("  - NaViT-style dynamic resolution visual encoder")
        print("  - ERNIE-4.5-0.3B language model")
        print("  - Supports 109 languages")
        print("")
        print("Supported tasks:")
        for task in PaddleOCRTask.allCases {
            print("  - \(task.rawValue): \(task.prompt)")
        }
        print("")
        print("Supported image formats:")
        for format in PaddleOCRVLPipeline.supportedFormats {
            print("  - \(format)")
        }
        print("")
        print("Processing modes:")
        print("  - base:    448x448 fixed resolution (default)")
        print("  - dynamic: NaViT-style dynamic resolution")
        print("")
        print("Usage examples:")
        print("  paddleocr-vl ocr image.png")
        print("  paddleocr-vl ocr image.png --task table")
        print("  paddleocr-vl ocr image.png --task formula --output result.txt")
        print("  paddleocr-vl ocr image.png --model /path/to/model --verbose")
        print("  paddleocr-vl ocr image.png --model PaddlePaddle/PaddleOCR-VL")
        print("  paddleocr-vl ocr *.png --task chart --mode dynamic")
        print("")
        print("Model can be specified as:")
        print("  - Local path: /path/to/model or ./models/paddleocr-vl")
        print("  - HuggingFace ID: PaddlePaddle/PaddleOCR-VL")
        print("  - HuggingFace ID with revision: PaddlePaddle/PaddleOCR-VL:main")
    }
}
