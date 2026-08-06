import Foundation

// MARK: - Types

/// Which runtime a listing is for. Maps onto HuggingFace's library filter, which is
/// the only reliable way to tell a GGUF repo from an MLX one without opening it.
enum MarketFormat: String, CaseIterable, Codable, Sendable {
    case gguf, mlx

    var label: String {
        switch self {
        case .gguf: return "GGUF"
        case .mlx: return "MLX"
        }
    }

    var engineNote: String {
        switch self {
        case .gguf: return "llama.cpp"
        case .mlx: return "MLX"
        }
    }

    var filterValue: String { rawValue }
}

enum MarketSort: String, CaseIterable, Codable, Sendable {
    case trending, downloads, likes, recent

    var label: String {
        switch self {
        case .trending: return "Trending"
        case .downloads: return "Most downloaded"
        case .likes: return "Most liked"
        case .recent: return "Recently updated"
        }
    }

    /// HuggingFace's sort keys, which do not match the display names.
    var apiValue: String {
        switch self {
        case .trending: return "trendingScore"
        case .downloads: return "downloads"
        case .likes: return "likes"
        case .recent: return "lastModified"
        }
    }
}

struct MarketModel: Identifiable, Hashable, Sendable {
    var id: String              // "author/name"
    var author: String
    var name: String
    var downloads: Int
    var likes: Int
    var gated: Bool
    var tags: [String]
    var updated: Date?
    var pipelineTag: String?

    /// Tags worth showing. The raw list is mostly noise — licences, library names,
    /// language codes and the format we already filtered on.
    var displayTags: [String] {
        let noise: Set<String> = ["gguf", "mlx", "transformers", "safetensors", "text-generation",
                                  "conversational", "endpoints_compatible", "region:us", "autotrain_compatible"]
        return tags
            .filter { !noise.contains($0) && !$0.contains(":") && $0.count < 22 }
            .prefix(4)
            .map { $0 }
    }

    var downloadsLabel: String {
        switch downloads {
        case 1_000_000...: return String(format: "%.1fM", Double(downloads) / 1_000_000)
        case 1_000...: return String(format: "%.0fk", Double(downloads) / 1_000)
        default: return "\(downloads)"
        }
    }
}

// MARK: - Client

/// Read-only browse over the HuggingFace model index.
///
/// Deliberately thin: this only *finds* repos. Resolving files and downloading them
/// stays in `Downloader`, which already handles quant selection, parallel ranged
/// fetches, resume and atomic install.
enum MarketClient {

    enum Failure: LocalizedError {
        case badStatus(Int, String)
        case decoding(String)

        var errorDescription: String? {
            switch self {
            case .badStatus(429, _):
                return "HuggingFace is rate-limiting this machine. Wait a moment, or add a token in Settings to raise the limit."
            case .badStatus(let code, let body):
                return "HuggingFace returned \(code)\(body.isEmpty ? "" : ": \(body)")"
            case .decoding(let what):
                return "Could not read the HuggingFace response: \(what)"
            }
        }
    }

    static func search(query: String,
                       format: MarketFormat,
                       sort: MarketSort,
                       limit: Int = 40,
                       token: String = "") async throws -> [MarketModel] {
        var c = URLComponents(string: "https://huggingface.co/api/models")!
        var items = [
            URLQueryItem(name: "filter", value: format.filterValue),
            // Without this the top results include embedding models and bare
            // chat-template repos — things you cannot load and chat with.
            URLQueryItem(name: "pipeline_tag", value: "text-generation"),
            URLQueryItem(name: "sort", value: sort.apiValue),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: String(limit)),
        ]
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { items.append(URLQueryItem(name: "search", value: trimmed)) }
        c.queryItems = items

        var req = URLRequest(url: c.url!)
        req.timeoutInterval = 20
        // Optional: a token only raises the rate limit and reveals gated repos.
        if !token.isEmpty { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }

        let (data, response) = try await URLSession.shared.data(for: req)
        guard let http = response as? HTTPURLResponse else { throw Failure.decoding("not an HTTP response") }
        guard http.statusCode == 200 else {
            throw Failure.badStatus(http.statusCode,
                                    String(decoding: data.prefix(300), as: UTF8.self))
        }
        return try parse(data)
    }

    /// Split out so it can be tested against fixed bytes.
    static func parse(_ data: Data) throws -> [MarketModel] {
        guard let raw = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw Failure.decoding("expected an array of models")
        }
        return raw.compactMap { entry in
            guard let id = entry["id"] as? String ?? entry["modelId"] as? String,
                  !id.isEmpty else { return nil }
            let parts = id.split(separator: "/", maxSplits: 1).map(String.init)
            // `gated` is `false` or a string like "auto"/"manual", never a plain true.
            let gated: Bool
            if let b = entry["gated"] as? Bool { gated = b }
            else if entry["gated"] is String { gated = true }
            else { gated = false }

            return MarketModel(
                id: id,
                author: entry["author"] as? String ?? parts.first ?? "",
                name: parts.count > 1 ? parts[1] : id,
                downloads: entry["downloads"] as? Int ?? 0,
                likes: entry["likes"] as? Int ?? 0,
                gated: gated,
                tags: entry["tags"] as? [String] ?? [],
                updated: (entry["lastModified"] as? String).flatMap(iso.date(from:)),
                pipelineTag: entry["pipeline_tag"] as? String
            )
        }
    }

    private static let iso: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    static func selfCheck() -> [String] {
        var f: [String] = []
        let sample = """
        [{"id":"unsloth/Qwen3-Coder-30B-GGUF","author":"unsloth","downloads":5114998,"likes":868,
          "gated":false,"tags":["gguf","uncensored","qwen3","license:apache-2.0","region:us"],
          "lastModified":"2026-01-02T03:04:05.000Z","pipeline_tag":"text-generation"},
         {"id":"someone/Gated-Model","gated":"manual","downloads":12,"likes":0,"tags":[]},
         {"noIdAtAll":true}]
        """.data(using: .utf8)!

        guard let models = try? parse(sample) else { return ["parse threw on valid input"] }
        if models.count != 2 { f.append("parse: expected 2 usable rows, got \(models.count)") }
        guard let first = models.first else { return f + ["no first row"] }
        if first.author != "unsloth" { f.append("author: \(first.author)") }
        if first.name != "Qwen3-Coder-30B-GGUF" { f.append("name: \(first.name)") }
        if first.downloadsLabel != "5.1M" { f.append("downloadsLabel: \(first.downloadsLabel)") }
        if first.updated == nil { f.append("lastModified did not parse") }
        // Noise tags and anything with a colon must be filtered out.
        if first.displayTags.contains(where: { $0.contains(":") || $0 == "gguf" }) {
            f.append("displayTags kept noise: \(first.displayTags)")
        }
        if !first.displayTags.contains("uncensored") { f.append("displayTags dropped a real tag") }
        // A string `gated` means gated.
        if models.count > 1, !models[1].gated { f.append("string gated value not treated as gated") }
        if models.first(where: { $0.id.isEmpty }) != nil { f.append("row without an id was kept") }

        if MarketSort.trending.apiValue != "trendingScore" { f.append("trending sort key wrong") }
        if (try? parse(Data("{}".utf8))) != nil { f.append("object instead of array should throw") }
        return f
    }
}
