// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "PaddleOCRVL",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "PaddleOCRVL", targets: ["PaddleOCRVL"]),
        .executable(name: "PaddleOCRVLCLI", targets: ["PaddleOCRVLCLI"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.29.1")),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "0.1.21")),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.0")
    ],
    targets: [
        .target(
            name: "PaddleOCRVL",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers")
            ]
        ),
        .executableTarget(
            name: "PaddleOCRVLCLI",
            dependencies: [
                "PaddleOCRVL",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        )
    ]
)
