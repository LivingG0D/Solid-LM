import Foundation

// MARK: - Engines

enum EngineKind: String, Codable, CaseIterable, Sendable {
    case llama, mlx, vllm

    var label: String {
        switch self {
        case .llama: return "llama.cpp"
        case .mlx: return "MLX"
        case .vllm: return "vLLM"
        }
    }
}

// MARK: - Models

enum FitClass: String, Codable, Sendable {
    case fits, tight, tooBig

    var note: String {
        switch self {
        case .fits: return "fits in memory"
        case .tight: return "tight — close other apps"
        case .tooBig: return "larger than usable memory"
        }
    }
}

struct LocalModel: Identifiable, Hashable, Codable, Sendable {
    var id: String              // "publisher/name"
    var name: String
    var publisher: String
    var engines: [EngineKind]
    var path: String            // .gguf file, or model directory for MLX
    var size: Int64             // bytes on disk (all shards)
    var quant: String
    var arch: String
    var estTokS: Double
    var fit: FitClass
    /// Sibling `mtp-*.gguf` draft head, when the publisher shipped one.
    ///
    /// These are Multi-Token Prediction heads for speculative decoding: the draft
    /// proposes several tokens and the full model verifies them in one pass, so the
    /// weights are read once for multiple tokens. That is the only way past the
    /// memory-bandwidth ceiling — measured 14.1 -> 25.5 tok/s on Gemma4-12B.
    var draftPath: String?

    /// Optional so a model list persisted before drafts existed still decodes.
    var hasDraft: Bool { draftPath != nil }
}

/// Per-model load settings, persisted by model id.
struct LoadConfig: Codable, Hashable, Sendable {
    var ctx: Int = 8192
    var gpuLayers: Int = 999    // full offload: auto ratio measured 3.99 vs 37.9 tok/s
    var flashAttention: Bool = true
    var quantizeKVCache: Bool = false
    /// Use the model's `mtp-*.gguf` draft head when one exists. Measured 1.8x on
    /// Gemma4-12B; costs ~0.25 GB of memory for the draft.
    var speculativeDecoding: Bool = true
    /// Tokens drafted per verify pass. 3 measured fastest; 5+ is slower than none.
    var draftTokens: Int = 3

    /// Older configs predate these two fields, and a missing key must not wipe the
    /// user's saved context length and offload settings.
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        ctx = try c.decodeIfPresent(Int.self, forKey: .ctx) ?? 8192
        gpuLayers = try c.decodeIfPresent(Int.self, forKey: .gpuLayers) ?? 999
        flashAttention = try c.decodeIfPresent(Bool.self, forKey: .flashAttention) ?? true
        quantizeKVCache = try c.decodeIfPresent(Bool.self, forKey: .quantizeKVCache) ?? false
        speculativeDecoding = try c.decodeIfPresent(Bool.self, forKey: .speculativeDecoding) ?? true
        draftTokens = try c.decodeIfPresent(Int.self, forKey: .draftTokens) ?? 3
    }

    init() {}
}

// MARK: - Engine state

enum EngineState: Equatable, Sendable {
    case idle
    case loading
    case ready
    case failed(String)

    var isReady: Bool { self == .ready }
    var isLoading: Bool { self == .loading }
    var errorText: String? { if case .failed(let m) = self { return m }; return nil }
}

// MARK: - Chat

struct MsgStats: Codable, Hashable, Sendable {
    var tokPerSec: Double
    var ttft: Double
    var tokens: Int
}

struct Msg: Identifiable, Codable, Hashable, Sendable {
    var id: UUID = UUID()
    var role: Role
    var content: String
    var thinking: String = ""
    var stats: MsgStats?
    /// Engine/transport failure shown with this message. Kept out of `content` on
    /// purpose: content is replayed to the model as history, and an error string
    /// fed back as the assistant's own words derails the next turn.
    var error: String?

    enum Role: String, Codable, Sendable { case user, assistant }
}

struct Conversation: Identifiable, Codable, Hashable, Sendable {
    var id: UUID = UUID()
    var title: String = ""
    var messages: [Msg] = []
    var modelID: String = ""
    var updated: Date = .now
}

// MARK: - Sampling

struct Samplers: Codable, Hashable, Sendable {
    var system: String = ""
    var temperature: Double = 0.6
    var topK: Int = 64
    var topP: Double = 0.9
    var minP: Double = 0.05
    var repeatPenalty: Double = 1.1
    var maxTokens: Int = 4096
}

// MARK: - HuggingFace

struct HFFile: Codable, Hashable, Sendable {
    var path: String
    var size: Int64
}

struct HFRepo: Codable, Hashable, Sendable {
    var repo: String
    var kind: Kind
    var gated: Bool
    var files: [HFFile]
    var quants: [String]

    enum Kind: String, Codable, Sendable { case gguf, mlx }
}

struct DownloadJob: Identifiable, Hashable, Sendable {
    var id: UUID = UUID()
    var repo: String
    var quant: String
    var destination: URL
    var total: Int64
    var done: Int64 = 0
    var state: State = .queued
    var error: String = ""
    var bytesPerSec: Double = 0
    var currentFile: String = ""

    enum State: String, Sendable { case queued, downloading, installing, done, failed, cancelled }

    var fraction: Double { total > 0 ? min(Double(done) / Double(total), 1) : 0 }
    var eta: TimeInterval {
        guard state == .downloading, bytesPerSec > 1 else { return 0 }
        return Double(total - done) / bytesPerSec
    }
}

// MARK: - Formatting helpers

enum Fmt {
    static func bytes(_ n: Int64) -> String {
        let gb = Double(n) / 1_073_741_824
        if gb >= 1 { return String(format: "%.1f GB", gb) }
        return String(format: "%.0f MB", Double(n) / 1_048_576)
    }

    static func speed(_ bps: Double) -> String {
        bps >= 1_048_576 ? String(format: "%.1f MB/s", bps / 1_048_576)
                         : String(format: "%.0f KB/s", bps / 1024)
    }

    static func duration(_ s: TimeInterval) -> String {
        let t = Int(s.rounded())
        if t <= 0 { return "" }
        if t < 60 { return "\(t)s" }
        return "\(t / 60)m \(t % 60)s"
    }
}
