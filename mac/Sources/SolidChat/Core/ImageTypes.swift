import Foundation

// MARK: - Image models on disk

/// What a diffusion or upscaler checkpoint is for. sd.cpp loads a single-file
/// checkpoint for generation and a separate ESRGAN network for upscaling.
enum ImageModelKind: String, Codable, Sendable {
    case diffusion      // .gguf / .safetensors / .ckpt — txt2img, img2img, inpaint
    case upscaler       // ESRGAN .pth — upscale only
}

/// Rough architecture, inferred from filename and size. Drives the default
/// resolution and step count, which differ a lot between families.
enum ImageArch: String, Codable, Sendable {
    case sd15, sdxl, sd3, flux, qwen, unknown

    var nativeSize: Int {
        switch self {
        case .sd15: return 512
        case .sdxl, .sd3: return 1024
        case .flux, .qwen: return 1024
        case .unknown: return 512
        }
    }

    var defaultSteps: Int {
        switch self {
        case .sd15: return 20
        case .sdxl: return 30
        case .sd3: return 28
        case .flux, .qwen: return 20
        case .unknown: return 20
        }
    }

    /// Flux and friends are guidance-distilled; a high CFG wrecks them.
    var defaultCFG: Double {
        switch self {
        case .flux, .qwen: return 1.0
        case .sd15: return 7.0
        case .sdxl: return 6.0
        case .sd3: return 4.5
        case .unknown: return 7.0
        }
    }

    var label: String {
        switch self {
        case .sd15: return "SD 1.5"
        case .sdxl: return "SDXL"
        case .sd3: return "SD3"
        case .flux: return "FLUX"
        case .qwen: return "Qwen-Image"
        case .unknown: return "—"
        }
    }
}

struct ImageModel: Identifiable, Hashable, Codable, Sendable {
    var id: String              // absolute path — unique and stable
    var name: String
    var kind: ImageModelKind
    var arch: ImageArch
    var path: String
    var size: Int64
    var quant: String

    /// Diffusion models tolerate quantisation far worse than language models do.
    ///
    /// Measured on SD 1.5, identical prompt and seed 42, background corners averaged:
    ///
    ///     Q4_0   RGB 160,  0, 14   — a prompted white backdrop rendered saturated red
    ///     F16    RGB 169,162,177   — near-neutral grey
    ///     F16 + vae-ft-mse         — 170,164,179, i.e. a better VAE does not fix it
    ///
    /// The green channel collapses to zero at Q4_0; this is not a subtle tint. The
    /// damage is in the weights, not the backend — CPU and Metal agree — so the only
    /// fix is a wider quantisation. Q8_0 and F16 are the safe choices.
    var quantWarning: String? {
        guard kind == .diffusion else { return nil }
        let q = quant.uppercased()
        guard q.hasPrefix("Q2") || q.hasPrefix("Q3") || q.hasPrefix("Q4") || q.hasPrefix("IQ")
        else { return nil }
        return "\(quant) wrecks colour on diffusion models — a prompted white background came out fully red at Q4_0, where F16 gave neutral grey. A better VAE does not fix it. Prefer Q8_0 or F16."
    }
}

// MARK: - Generation parameters

enum ImageMode: String, Codable, CaseIterable, Sendable {
    case generate, edit, upscale

    var label: String {
        switch self {
        case .generate: return "Generate"
        case .edit: return "Edit"
        case .upscale: return "Upscale"
        }
    }

    var symbol: String {
        switch self {
        case .generate: return "wand.and.sparkles"
        case .edit: return "photo.badge.plus"
        case .upscale: return "arrow.up.left.and.arrow.down.right"
        }
    }
}

struct ImageParams: Codable, Hashable, Sendable {
    var prompt: String = ""
    var negativePrompt: String = ""
    var width: Int = 512
    var height: Int = 512
    var steps: Int = 20
    var cfgScale: Double = 7.0
    /// -1 means "pick a fresh random seed for every run".
    var seed: Int64 = -1
    var batchCount: Int = 1
    var sampler: String = "euler_a"
    var scheduler: String = "discrete"
    /// img2img only: 0 keeps the input, 1 ignores it.
    var strength: Double = 0.6
    var clipSkip: Int = -1

    static let samplers = ["euler", "euler_a", "heun", "dpm2", "dpm++2s_a",
                           "dpm++2m", "dpm++2mv2", "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd"]
    static let schedulers = ["discrete", "karras", "exponential", "ays", "gits", "smoothstep", "sgm_uniform"]
}

// MARK: - Results

/// One finished image. The PNG lives on disk; only the metadata is persisted.
struct GeneratedImage: Identifiable, Hashable, Codable, Sendable {
    var id: UUID = UUID()
    var fileName: String
    var mode: ImageMode
    var prompt: String
    var negativePrompt: String
    var modelName: String
    var width: Int
    var height: Int
    var steps: Int
    var cfgScale: Double
    var seed: Int64
    var sampler: String
    var elapsed: TimeInterval
    var created: Date = .now

    /// Resolved against the gallery directory at read time so moving the
    /// folder — or the app — never orphans an entry.
    func url(in directory: URL) -> URL { directory.appending(path: fileName) }
}

/// Live progress of a running job.
struct ImageProgress: Equatable, Sendable {
    var step: Int = 0
    var totalSteps: Int = 0
    var stage: String = ""
    var elapsed: TimeInterval = 0

    var fraction: Double {
        totalSteps > 0 ? min(Double(step) / Double(totalSteps), 1) : 0
    }
}
