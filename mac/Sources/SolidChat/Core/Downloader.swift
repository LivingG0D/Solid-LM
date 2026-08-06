import Foundation
import Observation

// MARK: - File-scope constants (private == file scoped, no collisions with other files)

private let dlHFBase = URL(string: "https://huggingface.co")!
private let dlUserAgent = "SolidChat/1.0 (macOS; +local)"

/// Extensions we are willing to pull down. Everything else in a repo is noise
/// (README images, .gitattributes, benchmarks, …).
private let dlKeptExtensions: Set<String> = [
    "json", "model", "txt", "safetensors", "gguf", "tiktoken", "jinja", "py",
]

private let dlSmallFileLimit: Int64 = 8 * 1024 * 1024   // one plain GET below this
private let dlParallelism = 8                            // concurrent ranged GETs per file
private let dlChunkSize: Int64 = 8 * 1024 * 1024        // sub-request size inside a segment
private let dlStreamBuffer = 512 * 1024                  // fallback stream flush size
private let dlMaxAttempts = 3

// MARK: - Errors

enum DownloaderError: LocalizedError, Sendable {
    case badRepoInput(String)
    case gated(String)
    case notFound(String)
    case httpStatus(Int, String)
    case noMatchingFiles(String, [String])
    case emptyRepo(String)
    case rangesUnsupported
    case fileSystem(String)
    case badResponse(String)

    var errorDescription: String? {
        switch self {
        case .badRepoInput(let s):
            return "\"\(s)\" is not a HuggingFace repo — use \"org/name\" or a huggingface.co model URL."
        case .gated(let r):
            return "\(r) is gated or private — add a HuggingFace token in Settings."
        case .notFound(let r):
            return "Repo not found: \(r)."
        case .httpStatus(let code, let what):
            return "HuggingFace returned HTTP \(code) for \(what)."
        case .noMatchingFiles(let quant, let available):
            if available.isEmpty { return "No downloadable files matched \"\(quant)\"." }
            return "No files matched quant \"\(quant)\". Available: \(available.joined(separator: ", "))."
        case .emptyRepo(let r):
            return "\(r) has no downloadable model files."
        case .rangesUnsupported:
            return "The server ignored the byte-range request."
        case .fileSystem(let m):
            return m
        case .badResponse(let m):
            return m
        }
    }

    var isRetryable: Bool {
        switch self {
        case .httpStatus(let code, _): return code == 408 || code == 429 || code >= 500
        case .badResponse: return true
        default: return false
        }
    }
}

// MARK: - Private plumbing types

/// Immutable, Sendable description of one download job's work.
private struct DownloadPlan: Sendable {
    var repo: String
    var token: String
    var files: [HFFile]
    var partDir: URL
    var finalDir: URL

    func fileURL(_ path: String) -> URL {
        dlHFBase.appendingPathComponent("\(repo)/resolve/main/\(path)")
    }
}

private struct ByteRange: Sendable {
    var lo: Int64
    var hi: Int64   // exclusive
}

private struct HeadInfo: Sendable {
    var length: Int64
    var acceptsRanges: Bool
}

private struct CounterSnapshot: Sendable {
    var bytes: Int64
    var file: String
    var installing: Bool
}

/// Byte accumulator shared by every worker of one job. Sampled from the main
/// actor ~4x/second so the UI never sees a per-chunk write storm.
private actor ByteCounter {
    private var bytes: Int64 = 0
    private var file: String = ""
    private var installing = false

    func add(_ n: Int64) { bytes += n }
    func set(_ n: Int64) { bytes = n }
    func value() -> Int64 { bytes }
    func setFile(_ s: String) { file = s }
    func beginInstall() { installing = true; file = "" }
    func snapshot() -> CounterSnapshot { CounterSnapshot(bytes: bytes, file: file, installing: installing) }
}

// MARK: - Downloader

@Observable
@MainActor
final class Downloader {

    var jobs: [DownloadJob] = []

    @ObservationIgnored private var tasks: [UUID: Task<Void, Never>] = [:]

    init() {}

    // MARK: Inspect

    /// Resolve a repo id / URL and ask the HuggingFace API what is inside it.
    func inspect(_ urlOrRepo: String, token: String) async throws -> HFRepo {
        let repo = try Self.parseRepo(urlOrRepo)
        return try await Self.fetchRepo(repo, token: token)
    }

    // MARK: Start / cancel

    func start(repo: HFRepo, quant: String, token: String, modelsRoot: URL) {
        let (publisher, name) = Self.split(repo: repo.repo)
        let publisherDir = modelsRoot.appendingPathComponent(publisher, isDirectory: true)
        let finalDir = publisherDir.appendingPathComponent(name, isDirectory: true)
        // The quant belongs in the staging path: two quants of one repo are separate
        // downloads, and without it they share a .part dir — cancelling or failing one
        // wipes the other's bytes. Staging also lives under a dot-directory so a
        // leftover .part is never scanned as an installed model.
        let stagingRoot = publisherDir.appendingPathComponent(".solidchat-staging", isDirectory: true)
        let stem = quant.isEmpty ? name : "\(name)@\(quant)"
        let partDir = stagingRoot.appendingPathComponent(stem + ".part", isDirectory: true)

        let selection: [HFFile]
        do {
            selection = try Self.filesToDownload(repo: repo, quant: quant)
        } catch {
            var failed = DownloadJob(repo: repo.repo, quant: quant, destination: finalDir, total: 0)
            failed.state = .failed
            failed.error = Self.message(error)
            jobs.append(failed)
            return
        }

        let total = selection.reduce(Int64(0)) { $0 + max(0, $1.size) }
        var job = DownloadJob(repo: repo.repo, quant: quant, destination: finalDir, total: total)
        job.state = .downloading
        jobs.append(job)

        let id = job.id
        let plan = DownloadPlan(repo: repo.repo, token: token, files: selection,
                                partDir: partDir, finalDir: finalDir)
        let counter = ByteCounter()
        let sampler = sampler(for: id, counter: counter)

        tasks[id] = Task { [weak self] in
            var thrown: Error?
            do {
                try await Downloader.perform(plan: plan, counter: counter)
            } catch {
                thrown = error
                if Downloader.isCancellation(error) {
                    await Downloader.removeDirectory(plan.partDir)
                }
            }
            sampler.cancel()
            guard let self else { return }
            self.finish(id: id, error: thrown, doneBytes: await counter.value())
        }
    }

    func cancel(_ id: UUID) {
        if let i = jobs.firstIndex(where: { $0.id == id }) {
            switch jobs[i].state {
            case .queued, .downloading, .installing:
                jobs[i].state = .cancelled
                jobs[i].bytesPerSec = 0
                jobs[i].currentFile = ""
            default:
                break
            }
        }
        tasks[id]?.cancel()
        tasks[id] = nil
    }

    // MARK: Main-actor job bookkeeping

    /// Samples the shared counter 4x/second and reports a rate averaged over a
    /// trailing window. A window rather than a per-tick average matters: bytes are
    /// credited a chunk at a time, so instantaneous samples are mostly zero and an
    /// EMA over them collapses to "0 KB/s" with an absurd ETA between chunks.
    private func sampler(for id: UUID, counter: ByteCounter) -> Task<Void, Never> {
        Task { [weak self] in
            var window: [(at: Date, bytes: Int64)] = []
            var speed: Double = 0
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 250_000_000)
                if Task.isCancelled { return }
                let snap = await counter.snapshot()
                let now = Date()

                window.append((now, snap.bytes))
                while window.count > 2, now.timeIntervalSince(window[0].at) > Self.rateWindow {
                    window.removeFirst()
                }
                if let oldest = window.first {
                    let dt = now.timeIntervalSince(oldest.at)
                    // Need at least a second of history before quoting a rate.
                    // The window average is already the smoother — layering an EMA
                    // on top just makes the number halve its way to zero in the
                    // quiet stretch between two chunk completions.
                    if dt >= 1.0 { speed = max(0, Double(snap.bytes - oldest.bytes) / dt) }
                }

                guard let self else { return }
                self.applySample(id: id, snapshot: snap, speed: speed)
            }
        }
    }

    /// Trailing window, in seconds, used to average download speed. Wide enough to
    /// span the quiet gap between two chunk completions on a slow link, so the UI
    /// shows a steady throughput figure instead of oscillating between the peak
    /// rate and "0 KB/s".
    nonisolated static let rateWindow: TimeInterval = 15

    private func applySample(id: UUID, snapshot snap: CounterSnapshot, speed: Double) {
        guard let i = jobs.firstIndex(where: { $0.id == id }) else { return }
        guard jobs[i].state == .downloading || jobs[i].state == .installing else { return }
        jobs[i].done = snap.bytes
        jobs[i].currentFile = snap.file
        if snap.installing {
            jobs[i].state = .installing
            jobs[i].bytesPerSec = 0
        } else {
            jobs[i].bytesPerSec = speed
        }
    }

    private func finish(id: UUID, error: Error?, doneBytes: Int64) {
        tasks[id] = nil
        guard let i = jobs.firstIndex(where: { $0.id == id }) else { return }
        jobs[i].bytesPerSec = 0
        jobs[i].currentFile = ""
        if let error {
            if Self.isCancellation(error) || jobs[i].state == .cancelled {
                jobs[i].state = .cancelled
                jobs[i].error = ""
            } else {
                jobs[i].state = .failed
                jobs[i].error = Self.message(error)
                jobs[i].done = doneBytes
            }
        } else {
            jobs[i].state = .done
            jobs[i].done = max(jobs[i].total, doneBytes)
            jobs[i].error = ""
            // Fires here, not from a view: a download that lands while the Downloads
            // tab is off screen must still make the new model appear.
            onModelInstalled?()
        }
    }

    /// Called after a download installs successfully. `AppStore` wires this to a rescan.
    var onModelInstalled: (@MainActor () -> Void)?

    // MARK: - Repo parsing (pure)

    /// Accepts "org/name", "https://huggingface.co/org/name",
    /// ".../org/name/tree/main", ".../org/name/blob/main/file.gguf" (and resolve/raw/commits),
    /// with surrounding whitespace, query strings and fragments.
    nonisolated static func parseRepo(_ raw: String) throws -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !s.isEmpty else { throw DownloaderError.badRepoInput(raw) }

        // Strip query / fragment.
        if let cut = s.firstIndex(where: { $0 == "?" || $0 == "#" }) { s = String(s[s.startIndex..<cut]) }

        // Strip scheme + host. Only huggingface.co / hf.co are accepted as hosts.
        if let schemeRange = s.range(of: "://") {
            s = String(s[schemeRange.upperBound...])
            guard let slash = s.firstIndex(of: "/") else { throw DownloaderError.badRepoInput(raw) }
            let host = String(s[s.startIndex..<slash]).lowercased()
            let bare = host.hasPrefix("www.") ? String(host.dropFirst(4)) : host
            guard bare == "huggingface.co" || bare == "hf.co" else { throw DownloaderError.badRepoInput(raw) }
            s = String(s[s.index(after: slash)...])
        } else {
            for host in ["huggingface.co/", "www.huggingface.co/", "hf.co/"] where s.lowercased().hasPrefix(host) {
                s = String(s.dropFirst(host.count))
                break
            }
        }
        if s.lowercased().hasPrefix("models/") { s = String(s.dropFirst("models/".count)) }

        // Split, allowing only leading/trailing empties (so "a//b" is rejected).
        var comps = s.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        while comps.first == "" { comps.removeFirst() }
        while comps.last == "" { comps.removeLast() }
        guard comps.count >= 2 else { throw DownloaderError.badRepoInput(raw) }

        if comps.count > 2 {
            let marker = comps[2].lowercased()
            let known: Set<String> = ["tree", "blob", "resolve", "raw", "commits", "commit"]
            guard known.contains(marker) else { throw DownloaderError.badRepoInput(raw) }
            comps = Array(comps.prefix(2))
        }

        let org = comps[0], name = comps[1]
        guard isRepoComponent(org), isRepoComponent(name) else { throw DownloaderError.badRepoInput(raw) }
        return "\(org)/\(name)"
    }

    nonisolated private static func isRepoComponent(_ s: String) -> Bool {
        guard !s.isEmpty, s != ".", s != ".." else { return false }
        return s.allSatisfy { c in
            c.isLetter || c.isNumber || c == "-" || c == "_" || c == "."
        }
    }

    nonisolated private static func split(repo: String) -> (publisher: String, name: String) {
        let parts = repo.split(separator: "/", maxSplits: 1).map(String.init)
        if parts.count == 2 { return (parts[0], parts[1]) }
        return ("unknown", repo.isEmpty ? "model" : repo)
    }

    // MARK: - Quant parsing (pure)

    /// Drops the ".gguf" extension and any "-00001-of-00009" shard suffix.
    nonisolated static func ggufStem(_ path: String) -> String {
        var s = path
        if s.lowercased().hasSuffix(".gguf") { s = String(s.dropLast(5)) }
        let parts = s.split(separator: "-", omittingEmptySubsequences: false)
        if parts.count >= 4 {
            let a = parts[parts.count - 3], b = parts[parts.count - 2], c = parts[parts.count - 1]
            let numeric = { (x: Substring) in !x.isEmpty && x.allSatisfy { $0.isNumber } }
            if b.lowercased() == "of", numeric(a), numeric(c) {
                s = parts.dropLast(3).joined(separator: "-")
            }
        }
        return s
    }

    /// Pulls the quant token out of a gguf path. Maximal munch, so
    /// "…-UD-Q3_K_XL.gguf" yields "Q3_K_XL" and never the truncated "Q3_K_".
    /// The *last* token in the path wins, because the quant tag is conventionally last.
    nonisolated static func parseQuant(fromPath path: String) -> String? {
        let chars = Array(ggufStem(path))
        var best: String?
        var i = 0
        while i < chars.count {
            let atBoundary = (i == 0) || !isAlnum(chars[i - 1])
            if atBoundary, let (token, end) = quantToken(chars, at: i) {
                best = token
                i = end
            } else {
                i += 1
            }
        }
        return best
    }

    nonisolated private static func isAlnum(_ c: Character) -> Bool { c.isLetter || c.isNumber }

    /// ASCII-only lowercase. `Character(c.lowercased())` would be the obvious
    /// spelling but it can trap on characters whose lowercase form is more than
    /// one grapheme cluster, and a model filename is attacker-adjacent input.
    nonisolated private static func lower(_ c: Character) -> Character {
        guard let a = c.asciiValue, a >= 65, a <= 90 else { return c }
        return Character(UnicodeScalar(a + 32))
    }

    /// Matches Q<digits>(_<alnum>)* / IQ<digits>(_<alnum>)* / BF16 / FP16 / F16 / F32 at `start`.
    nonisolated private static func quantToken(_ chars: [Character], at start: Int) -> (String, Int)? {
        func literal(_ word: String) -> Int? {
            let w = Array(word)
            guard start + w.count <= chars.count else { return nil }
            for k in 0..<w.count where lower(chars[start + k]) != lower(w[k]) {
                return nil
            }
            let end = start + w.count
            if end < chars.count, isAlnum(chars[end]) { return nil }
            return end
        }

        var i = start
        let head = lower(chars[i])
        if head == "i" {
            guard i + 1 < chars.count, lower(chars[i + 1]) == "q" else {
                return nil
            }
            i += 2
        } else if head == "q" {
            i += 1
        } else {
            for word in ["BF16", "FP16", "F16", "F32"] {
                if let end = literal(word) { return (word, end) }
            }
            return nil
        }

        // At least one digit must follow the Q / IQ prefix.
        let digitsStart = i
        while i < chars.count, chars[i].isNumber { i += 1 }
        guard i > digitsStart else { return nil }

        // Then any number of "_alnum+" groups, greedily.
        while i < chars.count, chars[i] == "_", i + 1 < chars.count, isAlnum(chars[i + 1]) {
            i += 1
            while i < chars.count, isAlnum(chars[i]) { i += 1 }
        }

        // Token may not be glued to another alphanumeric run.
        if i < chars.count, isAlnum(chars[i]) { return nil }
        return (String(chars[start..<i]).uppercased(), i)
    }

    nonisolated static func quantList(fromPaths paths: [String]) -> [String] {
        var seen = Set<String>()
        for p in paths where p.lowercased().hasSuffix(".gguf") && !isCompanionGGUF(p) {
            if let q = parseQuant(fromPath: p) { seen.insert(q) }
        }
        return seen.sorted()
    }

    /// A `.gguf` that ships beside a model rather than being one: a vision projector
    /// or a speculative-decoding draft head.
    ///
    /// These carry their own quant in the filename, so counting them invents quants the
    /// repo cannot actually serve. Real case: prism-ml/Ternary-Bonsai-27B-gguf offers
    /// "Q8_0" that exists only as `…-mmproj-Q8_0.gguf` — picking it downloaded a 600 MB
    /// projector instead of the 27B model, and the size shown was the projector's.
    nonisolated static func isCompanionGGUF(_ path: String) -> Bool {
        let name = ((path as NSString).lastPathComponent).lowercased()
        if name.contains("mmproj") { return true }                       // vision projector
        if name.hasPrefix("mtp-") || name.contains("-mtp-") { return true }   // MTP draft head
        // DSpark / DFlash draft models. Same trap in a different shape: in
        // prism-ml/Ternary-Bonsai-27B-gguf, "…-dspark-bf16.gguf" is 6.8 GB while the
        // real F16 is 50 GB — offering it as "BF16" hands over a draft, not a model.
        if name.contains("dspark") || name.contains("dflash") { return true }
        return false
    }

    // MARK: - File filtering / selection (pure)

    nonisolated static func isKeptFile(_ path: String) -> Bool {
        let comps = path.split(separator: "/").map(String.init)
        guard let last = comps.last, !last.isEmpty else { return false }
        if comps.contains("original") { return false }
        let ext = (last as NSString).pathExtension.lowercased()
        return dlKeptExtensions.contains(ext)
    }

    /// MLX repos take everything kept. GGUF repos take the shards of the chosen
    /// quant plus every non-gguf support file (config, tokenizer, chat template).
    nonisolated static func filesToDownload(repo: HFRepo, quant: String) throws -> [HFFile] {
        guard !repo.files.isEmpty else { throw DownloaderError.emptyRepo(repo.repo) }

        let ggufs = repo.files.filter { $0.path.lowercased().hasSuffix(".gguf") }
        let others = repo.files.filter { !$0.path.lowercased().hasSuffix(".gguf") }
        guard repo.kind == .gguf, !ggufs.isEmpty else { return repo.files }

        let want = quant.trimmingCharacters(in: .whitespaces).uppercased()
        if want.isEmpty && repo.quants.isEmpty {
            return others + ggufs.sorted { $0.path < $1.path }
        }

        // Companion files never satisfy a quant request — a projector quantised to Q8_0
        // is not a Q8_0 model.
        let matching = ggufs.filter {
            !isCompanionGGUF($0.path) && (parseQuant(fromPath: $0.path) ?? "") == want
        }
        guard !matching.isEmpty else { throw DownloaderError.noMatchingFiles(quant, repo.quants) }
        return others + matching.sorted { $0.path < $1.path }
    }

    // MARK: - HuggingFace API

    private struct APIModel: Decodable {
        var id: String?
        var isPrivate: Bool?
        var gated: Gated?
        var siblings: [Sibling]?

        enum CodingKeys: String, CodingKey {
            case id
            case isPrivate = "private"
            case gated
            case siblings
        }

        struct Sibling: Decodable {
            var rfilename: String
            var size: Int64?
            var lfs: LFS?
            struct LFS: Decodable { var size: Int64? }
            var bestSize: Int64 { lfs?.size ?? size ?? 0 }
        }

        /// `gated` is `false`, `"auto"` or `"manual"` depending on the repo.
        enum Gated: Decodable {
            case flag(Bool)
            case mode(String)

            init(from decoder: Decoder) throws {
                let c = try decoder.singleValueContainer()
                if let b = try? c.decode(Bool.self) { self = .flag(b); return }
                if let s = try? c.decode(String.self) { self = .mode(s); return }
                self = .flag(false)
            }

            var isGated: Bool {
                switch self {
                case .flag(let b): return b
                case .mode(let m): return m.lowercased() != "false"
                }
            }
        }
    }

    nonisolated static func fetchRepo(_ repo: String, token: String) async throws -> HFRepo {
        var comps = URLComponents(url: dlHFBase.appendingPathComponent("api/models/\(repo)"),
                                  resolvingAgainstBaseURL: false)
        comps?.queryItems = [URLQueryItem(name: "blobs", value: "true")]
        guard let url = comps?.url else { throw DownloaderError.badRepoInput(repo) }

        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        authorize(&req, token: token)

        let session = makeSession()
        defer { session.finishTasksAndInvalidate() }

        let (data, response) = try await session.data(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw DownloaderError.badResponse("No HTTP response from huggingface.co.")
        }
        switch http.statusCode {
        case 200: break
        case 401, 403: throw DownloaderError.gated(repo)
        case 404: throw DownloaderError.notFound(repo)
        default: throw DownloaderError.httpStatus(http.statusCode, repo)
        }

        let api = try JSONDecoder().decode(APIModel.self, from: data)
        let kept: [HFFile] = (api.siblings ?? []).compactMap { s in
            guard isKeptFile(s.rfilename) else { return nil }
            return HFFile(path: s.rfilename, size: s.bestSize)
        }
        guard !kept.isEmpty else { throw DownloaderError.emptyRepo(repo) }

        let hasGGUF = kept.contains { $0.path.lowercased().hasSuffix(".gguf") }
        return HFRepo(repo: repo,
                      kind: hasGGUF ? .gguf : .mlx,
                      gated: (api.gated?.isGated ?? false) || (api.isPrivate ?? false),
                      files: kept.sorted { $0.path < $1.path },
                      quants: quantList(fromPaths: kept.map(\.path)))
    }

    // MARK: - Networking helpers

    nonisolated private static func makeSession() -> URLSession {
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 60
        cfg.timeoutIntervalForResource = 60 * 60 * 12
        cfg.httpMaximumConnectionsPerHost = dlParallelism + 4
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        cfg.urlCache = nil
        cfg.waitsForConnectivity = true
        return URLSession(configuration: cfg)
    }

    nonisolated private static func authorize(_ req: inout URLRequest, token: String) {
        let t = token.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty { req.setValue("Bearer \(t)", forHTTPHeaderField: "Authorization") }
        req.setValue(dlUserAgent, forHTTPHeaderField: "User-Agent")
    }

    nonisolated private static func statusError(_ code: Int, _ url: URL) -> DownloaderError {
        switch code {
        case 401, 403: return .gated(url.lastPathComponent)
        case 404: return .notFound(url.lastPathComponent)
        default: return .httpStatus(code, url.lastPathComponent)
        }
    }

    nonisolated private static func retrying<T>(_ attempts: Int = dlMaxAttempts,
                                                _ body: () async throws -> T) async throws -> T {
        var delay: UInt64 = 400_000_000
        var attempt = 0
        while true {
            attempt += 1
            do {
                return try await body()
            } catch let error as DownloaderError where !error.isRetryable {
                throw error
            } catch let error as URLError where error.code == .cancelled {
                throw error
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                if attempt >= attempts { throw error }
                try await Task.sleep(nanoseconds: delay)
                delay *= 2
            }
        }
    }

    nonisolated private static func getData(url: URL, range: ByteRange?, token: String,
                                            session: URLSession) async throws -> Data {
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        authorize(&req, token: token)
        if let range {
            req.setValue("bytes=\(range.lo)-\(range.hi - 1)", forHTTPHeaderField: "Range")
        }
        let (data, response) = try await session.data(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw DownloaderError.badResponse("No HTTP response for \(url.lastPathComponent).")
        }
        if range != nil {
            if http.statusCode == 200 { throw DownloaderError.rangesUnsupported }
            guard http.statusCode == 206 else { throw statusError(http.statusCode, url) }
        } else {
            guard http.statusCode == 200 else { throw statusError(http.statusCode, url) }
        }
        return data
    }

    nonisolated private static func head(url: URL, token: String, session: URLSession) async throws -> HeadInfo {
        var req = URLRequest(url: url)
        req.httpMethod = "HEAD"
        authorize(&req, token: token)
        let (_, response) = try await session.data(for: req)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw DownloaderError.badResponse("HEAD failed for \(url.lastPathComponent).")
        }
        let accepts = (http.value(forHTTPHeaderField: "Accept-Ranges") ?? "").lowercased().contains("bytes")
        var length: Int64 = 0
        if let linked = http.value(forHTTPHeaderField: "x-linked-size"), let v = Int64(linked) {
            length = v
        } else if http.expectedContentLength > 0 {
            length = http.expectedContentLength
        }
        return HeadInfo(length: length, acceptsRanges: accepts)
    }

    // MARK: - Resume ledger

    /// Records which files inside a ".part" directory actually finished.
    ///
    /// Needed because `presize` gives a file its final length up front, so file size cannot
    /// distinguish "downloaded" from "allocated but interrupted". The ledger is the only
    /// thing a resume trusts; it is deleted before the model is installed.
    private final class CompletionLedger {
        private let url: URL
        private var done: Set<String>

        init(partDir: URL) {
            url = partDir.appendingPathComponent(".solidchat-resume.json")
            let decoded = (try? Data(contentsOf: url)).flatMap {
                try? JSONDecoder().decode([String].self, from: $0)
            }
            done = Set(decoded ?? [])
        }

        func isComplete(_ path: String) -> Bool { done.contains(path) }

        func markComplete(_ path: String) {
            done.insert(path)
            // Best effort: a lost write only costs a re-download, never corruption.
            if let data = try? JSONEncoder().encode(Array(done)) {
                try? data.write(to: url, options: .atomic)
            }
        }

        func discard() { try? FileManager.default.removeItem(at: url) }
    }

    // MARK: - The download itself (off the main actor)

    nonisolated private static func perform(plan: DownloadPlan, counter: ByteCounter) async throws {
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: plan.partDir, withIntermediateDirectories: true)
        } catch {
            throw DownloaderError.fileSystem("Could not create \(plan.partDir.path): \(error.localizedDescription)")
        }

        let ledger = CompletionLedger(partDir: plan.partDir)
        let session = makeSession()
        var succeeded = false
        defer {
            if succeeded { session.finishTasksAndInvalidate() } else { session.invalidateAndCancel() }
        }

        for file in plan.files {
            try Task.checkCancellation()
            await counter.setFile((file.path as NSString).lastPathComponent)

            let dest = plan.partDir.appendingPathComponent(file.path)
            do {
                try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            } catch {
                throw DownloaderError.fileSystem("Could not create \(dest.deletingLastPathComponent().path).")
            }

            // Resume: skip only files the ledger says finished.
            //
            // Size alone is NOT proof of completion: presize() truncates the file to its
            // full length before the first byte lands, so an interrupted transfer leaves a
            // full-size, mostly-zero file. Trusting size here silently installed a corrupt
            // multi-gigabyte model on the next attempt.
            if file.size > 0, let existing = fileSize(dest), existing == file.size,
               ledger.isComplete(file.path) {
                await counter.add(file.size)
                continue
            }

            let before = await counter.value()
            do {
                try await download(file: file, to: dest, plan: plan, session: session, counter: counter)
            } catch let error as DownloaderError where error.isRangeFallback {
                await counter.set(before)
                try await streamDownload(url: plan.fileURL(file.path), to: dest,
                                         token: plan.token, session: session, counter: counter)
            }
            ledger.markComplete(file.path)
        }

        try Task.checkCancellation()
        await counter.beginInstall()
        ledger.discard()          // never ships inside the installed model directory
        try install(from: plan.partDir, to: plan.finalDir)
        succeeded = true
    }

    nonisolated private static func download(file: HFFile, to dest: URL, plan: DownloadPlan,
                                             session: URLSession, counter: ByteCounter) async throws {
        let url = plan.fileURL(file.path)

        if file.size > 0 && file.size < dlSmallFileLimit {
            let data = try await retrying { try await getData(url: url, range: nil, token: plan.token, session: session) }
            try write(data, to: dest)
            await counter.add(Int64(data.count))
            return
        }

        var size = file.size
        var ranged = false
        if let info = try? await head(url: url, token: plan.token, session: session) {
            ranged = info.acceptsRanges
            if info.length > 0 { size = info.length }
        }
        try Task.checkCancellation()

        guard ranged, size > 0 else {
            try await streamDownload(url: url, to: dest, token: plan.token, session: session, counter: counter)
            return
        }

        try presize(dest, to: size)
        let segs = segments(total: size, count: dlParallelism)
        try await withThrowingTaskGroup(of: Void.self) { group in
            for seg in segs {
                group.addTask {
                    try await downloadSegment(url: url, dest: dest, segment: seg,
                                              token: plan.token, session: session, counter: counter)
                }
            }
            try await group.waitForAll()
        }
    }

    /// One of the concurrent workers: owns a contiguous byte range and its own
    /// file handle, so no two writers ever touch the same region.
    nonisolated private static func downloadSegment(url: URL, dest: URL, segment: ByteRange,
                                                    token: String, session: URLSession,
                                                    counter: ByteCounter) async throws {
        guard let handle = FileHandle(forWritingAtPath: dest.path) else {
            throw DownloaderError.fileSystem("Could not open \(dest.lastPathComponent) for writing.")
        }
        defer { try? handle.close() }

        var offset = segment.lo
        while offset < segment.hi {
            try Task.checkCancellation()
            let upper = min(offset + dlChunkSize, segment.hi)
            let start = offset
            let data = try await retrying {
                try await getData(url: url, range: ByteRange(lo: start, hi: upper),
                                  token: token, session: session)
            }
            guard !data.isEmpty else {
                throw DownloaderError.badResponse("Empty range response for \(dest.lastPathComponent).")
            }
            let written = min(Int64(data.count), segment.hi - offset)
            do {
                try handle.seek(toOffset: UInt64(offset))
                try handle.write(contentsOf: written == Int64(data.count) ? data : data.prefix(Int(written)))
            } catch {
                throw DownloaderError.fileSystem("Write failed for \(dest.lastPathComponent): \(error.localizedDescription)")
            }
            offset += written
            await counter.add(written)
        }
    }

    /// Single sequential stream — used when the server will not serve ranges.
    nonisolated private static func streamDownload(url: URL, to dest: URL, token: String,
                                                   session: URLSession, counter: ByteCounter) async throws {
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        authorize(&req, token: token)

        let (stream, response) = try await session.bytes(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw DownloaderError.badResponse("No HTTP response for \(url.lastPathComponent).")
        }
        guard http.statusCode == 200 else { throw statusError(http.statusCode, url) }

        try presize(dest, to: 0)
        guard let handle = FileHandle(forWritingAtPath: dest.path) else {
            throw DownloaderError.fileSystem("Could not open \(dest.lastPathComponent) for writing.")
        }
        defer { try? handle.close() }

        var buffer = [UInt8]()
        buffer.reserveCapacity(dlStreamBuffer)
        for try await byte in stream {
            buffer.append(byte)
            if buffer.count >= dlStreamBuffer {
                try Task.checkCancellation()
                try handle.write(contentsOf: Data(buffer))
                await counter.add(Int64(buffer.count))
                buffer.removeAll(keepingCapacity: true)
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: Data(buffer))
            await counter.add(Int64(buffer.count))
        }
    }

    // MARK: - Filesystem helpers

    nonisolated private static func segments(total: Int64, count: Int) -> [ByteRange] {
        guard total > 0, count > 0 else { return [] }
        let per = max(Int64(1), total / Int64(count))
        var out: [ByteRange] = []
        var lo: Int64 = 0
        for i in 0..<count {
            let hi = (i == count - 1) ? total : min(total, lo + per)
            if lo >= hi { break }
            out.append(ByteRange(lo: lo, hi: hi))
            lo = hi
        }
        if let last = out.last, last.hi < total {
            out[out.count - 1] = ByteRange(lo: last.lo, hi: total)
        }
        return out
    }

    nonisolated private static func fileSize(_ url: URL) -> Int64? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path) else { return nil }
        return (attrs[.size] as? NSNumber)?.int64Value
    }

    nonisolated private static func write(_ data: Data, to url: URL) throws {
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw DownloaderError.fileSystem("Could not write \(url.lastPathComponent): \(error.localizedDescription)")
        }
    }

    /// Creates the destination and gives it its final length up front so the
    /// parallel writers can seek anywhere inside it.
    nonisolated private static func presize(_ url: URL, to size: Int64) throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: url.path) {
            guard fm.createFile(atPath: url.path, contents: nil) else {
                throw DownloaderError.fileSystem("Could not create \(url.lastPathComponent).")
            }
        }
        do {
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.truncate(atOffset: UInt64(max(0, size)))
        } catch {
            throw DownloaderError.fileSystem("Could not size \(url.lastPathComponent): \(error.localizedDescription)")
        }
    }

    /// Moves the finished ".part" tree into place. If the model directory already
    /// exists (a second quant of the same repo) the files are merged into it.
    nonisolated private static func install(from partDir: URL, to finalDir: URL) throws {
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: finalDir.deletingLastPathComponent(), withIntermediateDirectories: true)
            if !fm.fileExists(atPath: finalDir.path) {
                try fm.moveItem(at: partDir, to: finalDir)
                return
            }
            guard let walker = fm.enumerator(at: partDir, includingPropertiesForKeys: [.isDirectoryKey]) else {
                throw DownloaderError.fileSystem("Could not read \(partDir.lastPathComponent).")
            }
            for case let src as URL in walker {
                let rel = relativePath(of: src, under: partDir)
                let dst = finalDir.appendingPathComponent(rel)
                let isDir = (try? src.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
                if isDir {
                    try fm.createDirectory(at: dst, withIntermediateDirectories: true)
                    continue
                }
                try fm.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
                if fm.fileExists(atPath: dst.path) { try fm.removeItem(at: dst) }
                try fm.moveItem(at: src, to: dst)
            }
            try? fm.removeItem(at: partDir)
        } catch let error as DownloaderError {
            throw error
        } catch {
            throw DownloaderError.fileSystem("Install failed: \(error.localizedDescription)")
        }
    }

    nonisolated private static func relativePath(of url: URL, under root: URL) -> String {
        let a = url.standardizedFileURL.path
        let b = root.standardizedFileURL.path
        if a.hasPrefix(b + "/") { return String(a.dropFirst(b.count + 1)) }
        return url.lastPathComponent
    }

    nonisolated private static func removeDirectory(_ url: URL) async {
        try? FileManager.default.removeItem(at: url)
    }

    // MARK: - Error helpers

    nonisolated static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        if let u = error as? URLError, u.code == .cancelled { return true }
        return false
    }

    nonisolated static func message(_ error: Error) -> String {
        if let d = error as? DownloaderError { return d.errorDescription ?? "Download failed." }
        return error.localizedDescription
    }

    // MARK: - Self check

    nonisolated static func selfCheck() -> [String] {
        var fails: [String] = []

        func repoOK(_ input: String, _ expected: String) {
            do {
                let got = try parseRepo(input)
                if got != expected { fails.append("parseRepo(\"\(input)\") = \"\(got)\", expected \"\(expected)\"") }
            } catch {
                fails.append("parseRepo(\"\(input)\") threw: \(message(error))")
            }
        }
        func repoRejected(_ input: String) {
            if let got = try? parseRepo(input) {
                fails.append("parseRepo(\"\(input)\") should be rejected, got \"\(got)\"")
            }
        }
        func quantIs(_ path: String, _ expected: String?) {
            let got = parseQuant(fromPath: path)
            if got != expected {
                fails.append("parseQuant(\"\(path)\") = \(got ?? "nil"), expected \(expected ?? "nil")")
            }
        }

        // parseRepo — accepted forms
        repoOK("unsloth/Qwen3-30B-A3B-GGUF", "unsloth/Qwen3-30B-A3B-GGUF")
        repoOK("  unsloth/Qwen3-30B-A3B-GGUF \n", "unsloth/Qwen3-30B-A3B-GGUF")
        repoOK("https://huggingface.co/org/name", "org/name")
        repoOK("https://huggingface.co/org/name/", "org/name")
        repoOK("https://huggingface.co/org/name/tree/main", "org/name")
        repoOK("https://huggingface.co/org/name/blob/main/model-Q4_K_M.gguf", "org/name")
        repoOK("https://huggingface.co/org/name/resolve/main/x.gguf?download=true", "org/name")
        repoOK("http://www.huggingface.co/org/name#files", "org/name")
        repoOK("hf.co/org/name", "org/name")
        repoOK("huggingface.co/org/name/tree/main", "org/name")

        // parseRepo — rejected forms
        repoRejected("")
        repoRejected("   ")
        repoRejected("justaname")
        repoRejected("/name")
        repoRejected("org/")
        repoRejected("org//name")
        repoRejected("org/name/extra")
        repoRejected("org/name/tree/main/deep/er/deeper".replacingOccurrences(of: "tree", with: "nope"))
        repoRejected("https://example.com/org/name")
        repoRejected("https://huggingface.co/")
        repoRejected("https://huggingface.co/org")
        repoRejected("a/b/c/d")

        // Quant parsing — the whole reason this is a hand-written scanner.
        quantIs("Qwen3-30B-A3B-UD-Q3_K_XL.gguf", "Q3_K_XL")
        quantIs("Qwen3-30B-A3B-Q3_K_S.gguf", "Q3_K_S")
        quantIs("model-Q4_K_M.gguf", "Q4_K_M")
        quantIs("model-q4_k_m.gguf", "Q4_K_M")
        quantIs("DeepSeek-R1-UD-IQ1_S.gguf", "IQ1_S")
        quantIs("foo-IQ4_XS.gguf", "IQ4_XS")
        quantIs("foo-Q8_0.gguf", "Q8_0")
        quantIs("foo-Q4_0-00001-of-00003.gguf", "Q4_0")
        quantIs("Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf", "Q4_K_M")
        quantIs("foo-BF16.gguf", "BF16")
        quantIs("foo-F16.gguf", "F16")
        quantIs("Qwen3-30B-A3B-Instruct.gguf", nil)
        quantIs("model.safetensors", nil)
        if ggufStem("foo-Q4_0-00002-of-00003.gguf") != "foo-Q4_0" {
            fails.append("ggufStem did not strip the shard suffix")
        }
        // Regression: prism-ml/Ternary-Bonsai-27B-gguf. Its only Q8_0 file is a vision
        // projector, so offering "Q8_0" as a quant meant downloading 600 MB of projector
        // instead of the 27B model.
        let realWorld = quantList(fromPaths: [
            "Ternary-Bonsai-27B-Q2_0.gguf",
            "Ternary-Bonsai-27B-F16.gguf",
            "Ternary-Bonsai-27B-mmproj-Q8_0.gguf",
            "Ternary-Bonsai-27B-mmproj-BF16.gguf",
            "Ternary-Bonsai-27B-dspark-bf16.gguf",
            "Ternary-Bonsai-27B-dspark-Q4_1.gguf",
            "mtp-gemma-4-12B-it.gguf",
        ])
        // BF16 and Q4_1 exist here only as dspark drafts — 6.8 GB and 1.8 GB against a
        // 50 GB real F16 — so neither is a quant of the model.
        if realWorld.contains("BF16") || realWorld.contains("Q4_1") {
            fails.append("quantList kept a dspark-draft-only quant: \(realWorld)")
        }
        if realWorld.contains("Q8_0") {
            fails.append("quantList kept a projector-only quant: \(realWorld)")
        }
        if !realWorld.contains("Q2_0") || !realWorld.contains("F16") {
            fails.append("quantList dropped a real quant: \(realWorld)")
        }
        for companion in ["x-mmproj-BF16.gguf", "mtp-gemma-4-12B-it.gguf", "a/b/MMPROJ-f16.gguf",
                          "m-dspark-bf16.gguf", "m-dflash-Q4_1.gguf"]
        where !isCompanionGGUF(companion) {
            fails.append("isCompanionGGUF missed \(companion)")
        }
        for model in ["Ternary-Bonsai-27B-Q2_0.gguf", "a-Q4_K_M-00001-of-00002.gguf"]
        where isCompanionGGUF(model) {
            fails.append("isCompanionGGUF wrongly flagged \(model)")
        }

        let quants = quantList(fromPaths: ["a-Q3_K_XL.gguf", "a-Q3_K_S.gguf", "a-Q3_K_XL-00001-of-00002.gguf", "README.md"])
        if quants != ["Q3_K_S", "Q3_K_XL"] {
            fails.append("quantList = \(quants), expected [Q3_K_S, Q3_K_XL]")
        }

        // Kept-file filter
        for good in ["config.json", "tokenizer.model", "vocab.txt", "model.safetensors",
                     "m-Q4_K_M.gguf", "tokenizer.tiktoken", "chat_template.jinja", "modeling_x.py"] {
            if !isKeptFile(good) { fails.append("isKeptFile(\"\(good)\") should be true") }
        }
        for bad in ["README.md", ".gitattributes", "original/consolidated.safetensors",
                    "original/params.json", "figures/loss.png", "nested/original/x.safetensors"] {
            if isKeptFile(bad) { fails.append("isKeptFile(\"\(bad)\") should be false") }
        }

        // File selection — gguf
        let ggufRepo = HFRepo(
            repo: "unsloth/Demo-GGUF",
            kind: .gguf,
            gated: false,
            files: [
                HFFile(path: "config.json", size: 1_000),
                HFFile(path: "tokenizer.json", size: 2_000),
                HFFile(path: "Demo-Q3_K_S.gguf", size: 10),
                HFFile(path: "Demo-Q3_K_XL-00001-of-00002.gguf", size: 20),
                HFFile(path: "Demo-Q3_K_XL-00002-of-00002.gguf", size: 30),
            ],
            quants: ["Q3_K_S", "Q3_K_XL"])
        do {
            let picked = try filesToDownload(repo: ggufRepo, quant: "Q3_K_XL").map(\.path)
            let expected = ["config.json", "tokenizer.json",
                            "Demo-Q3_K_XL-00001-of-00002.gguf", "Demo-Q3_K_XL-00002-of-00002.gguf"]
            if Set(picked) != Set(expected) {
                fails.append("filesToDownload(Q3_K_XL) = \(picked)")
            }
            if picked.contains("Demo-Q3_K_S.gguf") {
                fails.append("filesToDownload(Q3_K_XL) leaked the Q3_K_S shard")
            }
        } catch {
            fails.append("filesToDownload(Q3_K_XL) threw: \(message(error))")
        }
        do {
            _ = try filesToDownload(repo: ggufRepo, quant: "Q9_K_NOPE")
            fails.append("filesToDownload should throw for an unknown quant")
        } catch {
            // expected
        }

        // File selection — mlx takes everything
        let mlxRepo = HFRepo(
            repo: "mlx-community/Demo-4bit",
            kind: .mlx,
            gated: false,
            files: [HFFile(path: "config.json", size: 1),
                    HFFile(path: "model.safetensors", size: 2),
                    HFFile(path: "tokenizer.json", size: 3)],
            quants: [])
        do {
            let picked = try filesToDownload(repo: mlxRepo, quant: "")
            if picked.count != 3 { fails.append("filesToDownload(mlx) = \(picked.count) files, expected 3") }
        } catch {
            fails.append("filesToDownload(mlx) threw: \(message(error))")
        }

        // Segmenting must tile the file exactly once, with no gaps or overlap.
        let segs = segments(total: 1_000_003, count: dlParallelism)
        if segs.first?.lo != 0 || segs.last?.hi != 1_000_003 {
            fails.append("segments() does not cover the whole file")
        }
        for i in 1..<max(1, segs.count) where segs[i].lo != segs[i - 1].hi {
            fails.append("segments() has a gap or overlap at \(i)")
        }

        return fails
    }
}

private extension DownloaderError {
    /// Only the "server ignored Range" case is worth retrying as a plain stream.
    var isRangeFallback: Bool {
        if case .rangesUnsupported = self { return true }
        return false
    }
}
