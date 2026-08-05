import Foundation

// MARK: - ImageClient

/// Everything that talks to stable-diffusion.cpp: HTTP for generation and editing,
/// a one-shot `sd-cli` for upscaling.
///
/// Deliberately stateless. Nothing here consults engine state, which is what lets
/// `upscale` work with no diffusion model — and no server — in memory at all.
enum ImageClient {

    // MARK: Endpoint

    /// The port, the binary locations and the user's overrides for them all belong to
    /// `ImagePaths`; duplicating them here is how the two halves drift apart.
    static var baseURL: URL { ImagePaths.baseURL }

    /// How long we wait to learn whether anything is listening. Never applied to a
    /// running job — see `jobSession`.
    static let connectTimeout: TimeInterval = 30

    /// Effectively "no cap". A single 1024×1024 FLUX image on this machine is minutes of
    /// work, and there is no upper bound worth guessing at.
    private static let jobTimeout: TimeInterval = 7 * 24 * 60 * 60

    // MARK: Errors

    enum Failure: LocalizedError, Sendable {
        /// Nothing answered on the image port at all.
        case unreachable(String)
        /// Non-200 from sd-server. The body is its own explanation and is worth more
        /// than anything we could invent.
        case badStatus(code: Int, body: String)
        case notHTTP
        /// A 200 with no usable `images` array.
        case noImages(String)
        case badBase64(index: Int)
        case cliMissing(String)
        case inputMissing(String)
        case modelMissing(String)
        case notAnUpscaler(String)
        case cliFailed(code: Int32, output: String)
        case noOutput(String)
        case noModelLoaded

        var errorDescription: String? {
            switch self {
            case .unreachable(let detail):
                return "No image server at \(ImagePaths.baseURL.absoluteString) — \(detail)"
            case .badStatus(let code, let body):
                let text = ImageClient.humanMessage(from: body)
                return text.isEmpty ? "Image server returned HTTP \(code)."
                                    : "Image server returned HTTP \(code): \(text)"
            case .notHTTP:
                return "The image server did not return a valid HTTP response."
            case .noImages(let detail):
                return detail.isEmpty ? "The image server returned no images."
                                      : "The image server returned no images: \(detail)"
            case .badBase64(let index):
                return "Image \(index + 1) came back as unreadable base64."
            case .cliMissing(let path):
                return "sd-cli not found at \(path). Point \"\(ImagePaths.Key.sdCLI)\" at your build in Settings."
            case .inputMissing(let path):
                return "No such image: \(path)"
            case .modelMissing(let path):
                return "Upscaler model missing: \(path)"
            case .notAnUpscaler(let name):
                return "\(name) is not an ESRGAN upscaler."
            case .cliFailed(let code, let output):
                let why = code < 0 ? "was killed by signal \(-code)" : "exited with code \(code)"
                return output.isEmpty ? "sd-cli \(why)." : "sd-cli \(why):\n\(output)"
            case .noOutput(let output):
                return output.isEmpty ? "sd-cli wrote no image."
                                      : "sd-cli wrote no image:\n\(output)"
            case .noModelLoaded:
                return "The image server has no model loaded."
            }
        }
    }

    // MARK: Sessions

    /// Generation blocks: sd-server sends nothing whatsoever until the image is finished,
    /// so any idle timeout short enough to be useful would abort a legitimate job.
    /// Reachability is checked up front instead, by `ensureReachable`.
    private static let jobSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = jobTimeout
        config.timeoutIntervalForResource = jobTimeout
        config.waitsForConnectivity = false
        config.httpMaximumConnectionsPerHost = 2
        config.urlCache = nil
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: config)
    }()

    /// Metadata calls. sd-server answers GET routes on a separate thread — measured at
    /// 0.3 ms while a generation was in flight — so a short deadline here is honest even
    /// under load, and is the only place a connection timeout can live without lying.
    private static let infoSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = connectTimeout
        config.timeoutIntervalForResource = connectTimeout
        config.waitsForConnectivity = false
        config.urlCache = nil
        config.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        return URLSession(configuration: config)
    }()

    // MARK: Generation

    static func txt2img(params: ImageParams) async throws -> [Data] {
        try await generate(path: "sdapi/v1/txt2img", body: payload(for: params))
    }

    static func img2img(params: ImageParams, initImage: Data, mask: Data?) async throws -> [Data] {
        var body = payload(for: params)
        body["init_images"] = [initImage.base64EncodedString()]
        body["denoising_strength"] = params.strength
        if let mask, !mask.isEmpty {
            body["mask"] = mask.base64EncodedString()
        }
        return try await generate(path: "sdapi/v1/img2img", body: body)
    }

    /// The Automatic1111 request shape sd-server implements. Verified against a live
    /// server: it echoes exactly these keys back in the response's `parameters` object.
    private static func payload(for params: ImageParams) -> [String: Any] {
        var body: [String: Any] = [
            "prompt": params.prompt,
            "negative_prompt": params.negativePrompt,
            "width": params.width,
            "height": params.height,
            "steps": params.steps,
            "cfg_scale": params.cfgScale,
            "seed": params.seed,
            // *** This sd-server build reads the image count from batch_size and IGNORES
            // *** n_iter — verified: n_iter 3 returns one image, batch_size 3 returns three.
            // n_iter still goes out as 1 so the same body means the same thing to a stock
            // Automatic1111 backend, where the two fields multiply.
            "batch_size": max(1, params.batchCount),
            "n_iter": 1,
            "sampler_name": params.sampler,
            "scheduler": params.scheduler
        ]
        // sd.cpp reads any value <= 0 as "unspecified" and picks per architecture; sending
        // our sentinel would just be noise.
        if params.clipSkip >= 0 { body["clip_skip"] = params.clipSkip }
        return body
    }

    private static func generate(path: String, body: [String: Any]) async throws -> [Data] {
        try await ensureReachable()

        var request = URLRequest(url: baseURL.appending(path: path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        // Set on the request as well as the session: which of the two wins is not
        // something to leave to chance when the cost of being wrong is a killed job.
        request.timeoutInterval = jobTimeout
        request.httpBody = try JSONSerialization.data(withJSONObject: body, options: [])

        let response = try await perform(request, on: jobSession)
        return try images(from: response)
    }

    /// Fails fast and clearly when nothing is serving the image port.
    ///
    /// Any HTTP answer at all counts as reachable — even a 404. This gate exists to catch
    /// "nothing is there", not to police which routes a given sdcpp build exposes.
    private static func ensureReachable() async throws {
        var request = URLRequest(url: baseURL.appending(path: "sdapi/v1/options"))
        request.httpMethod = "GET"
        request.timeoutInterval = connectTimeout
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        do {
            _ = try await perform(request, on: infoSession)
        } catch is Failure {
            return
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw Failure.unreachable(error.localizedDescription)
        }
    }

    // MARK: Server info

    /// Name of the checkpoint sd-server currently has loaded, for display.
    static func serverInfo() async throws -> String {
        var models = URLRequest(url: baseURL.appending(path: "sdapi/v1/sd-models"))
        models.httpMethod = "GET"
        models.timeoutInterval = connectTimeout
        models.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData

        let listBody = try await perform(models, on: infoSession)
        if let list = (try? JSONSerialization.jsonObject(with: listBody)) as? [[String: Any]] {
            for entry in list {
                for key in ["model_name", "title", "filename"] {
                    if let name = entry[key] as? String, !name.isEmpty { return name }
                }
            }
        }

        // Split-file models report nothing under /sd-models; /options still names them.
        var options = URLRequest(url: baseURL.appending(path: "sdapi/v1/options"))
        options.httpMethod = "GET"
        options.timeoutInterval = connectTimeout
        options.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData

        let optionsBody = try await perform(options, on: infoSession)
        if let root = (try? JSONSerialization.jsonObject(with: optionsBody)) as? [String: Any],
           let name = root["sd_model_checkpoint"] as? String, !name.isEmpty {
            return name
        }
        throw Failure.noModelLoaded
    }

    // MARK: Upscaling

    /// Upscales one image with an ESRGAN network.
    ///
    /// Runs entirely through `sd-cli`: sd-server has no upscale route, and ESRGAN needs
    /// neither a diffusion model nor a loaded engine. That independence is the point —
    /// upscaling stays available when nothing is loaded.
    ///
    /// `ProcessRunner` does the spawning, so the work happens off this actor, the child
    /// gets its own process group, and it dies with the app no matter how we exit.
    /// Tile size for an ESRGAN pass, when the caller has no preference.
    ///
    /// There is no single best value — measured on this M5 with RealESRGAN x4:
    ///
    ///     512x512  input   tile 128 -> 15.5s    tile 512 -> 10.4s
    ///     1024x1024 input  tile 128 -> 74.7s    tile 512 -> 91.2s
    ///
    /// An image that fits inside one tile skips the tiling machinery altogether, which
    /// is the 1.5x win at 512. Past that the larger tile loses, so bigger inputs keep
    /// sd.cpp's 128. 512x512 is exactly what SD 1.5 produces here, so the common path
    /// is the one that got faster.
    ///
    /// `0` is not a valid value to pass through — sd-cli rejects it and prints usage.
    static func autoTileSize(for input: URL) -> Int {
        guard let data = try? Data(contentsOf: input, options: .mappedIfSafe),
              let size = pngSize(data)
        else { return 128 }
        return max(size.width, size.height) <= 512 ? 512 : 128
    }

    static func upscale(input: URL, model: ImageModel, repeats: Int, tileSize: Int) async throws -> Data {
        let fm = FileManager.default
        let cli = ImagePaths.cli

        guard fm.isExecutableFile(atPath: cli.path) else { throw Failure.cliMissing(cli.path) }
        guard fm.fileExists(atPath: input.path) else { throw Failure.inputMissing(input.path) }
        guard model.kind == .upscaler else { throw Failure.notAnUpscaler(model.name) }
        guard fm.fileExists(atPath: model.path) else { throw Failure.modelMissing(model.path) }

        // Our own name, never the caller's: sd-cli reads printf specifiers out of -o, so a
        // stray "%d" in a user's filename would silently become an image *sequence*.
        let output = fm.temporaryDirectory
            .appending(path: "solidchat-upscale-\(UUID().uuidString).png")
        defer { try? fm.removeItem(at: output) }

        var arguments = ["-M", "upscale",
                         "-i", input.path,
                         "--upscale-model", model.path,
                         "-o", output.path]
        if repeats > 1 { arguments += ["--upscale-repeats", String(repeats)] }
        // tileSize <= 0 means "choose for me" — sd.cpp's own default is a flat 128,
        // which is the wrong call for the size this app most often produces.
        let tile = tileSize > 0 ? tileSize : autoTileSize(for: input)
        arguments += ["--upscale-tile-size", String(tile)]

        let runner = ProcessRunner(executable: cli, arguments: arguments, environment: [:])
        let code = try await wait(for: runner)
        guard code == 0 else { throw Failure.cliFailed(code: code, output: tail(of: runner)) }

        guard let png = try? Data(contentsOf: output), !png.isEmpty else {
            throw Failure.noOutput(tail(of: runner))
        }
        return png
    }

    /// Suspends until the child exits, killing it if the calling Task is cancelled.
    private static func wait(for runner: ProcessRunner) async throws -> Int32 {
        let code = try await withTaskCancellationHandler { () async throws -> Int32 in
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Int32, Error>) in
                runner.onExit = { code in continuation.resume(returning: code) }
                do {
                    try runner.start()
                } catch {
                    // Nothing was spawned, so onExit will never fire. Drop it first or a
                    // stray callback could resume the continuation a second time.
                    runner.onExit = nil
                    continuation.resume(throwing: error)
                }
            }
        } onCancel: {
            // terminate() blocks up to 5s waiting for the child to die, and onCancel runs
            // on whichever thread cancelled us — frequently the main actor.
            Task.detached(priority: .utility) { runner.terminate() }
        }
        // A cancelled run reports -SIGTERM; surface it as cancellation, not as a failure.
        try Task.checkCancellation()
        return code
    }

    /// What went wrong, out of the merged stdout+stderr.
    ///
    /// sd-cli splits its diagnostics across both streams and buries the one line that
    /// matters under twenty lines of Metal banner, so error lines are pulled out first —
    /// a plain tail shows the user everything except the reason.
    private static func tail(of runner: ProcessRunner) -> String {
        let lines = runner.recentLines()
        let errors = lines.filter { $0.contains("[ERROR]") }
        if !errors.isEmpty { return errors.suffix(12).joined(separator: "\n") }
        return lines.suffix(20).joined(separator: "\n")
    }

    // MARK: Transport

    /// Runs one request to completion and cancels the underlying `URLSessionTask` the
    /// instant the calling `Task` is cancelled.
    ///
    /// The session task is created *before* the cancellation handler is installed, so
    /// there is no window in which a cancel has nothing to act on; entering
    /// `withTaskCancellationHandler` already-cancelled fires `onCancel` immediately.
    private static func perform(_ request: URLRequest, on session: URLSession) async throws -> Data {
        let (stream, sink) = AsyncThrowingStream<Data, Error>.makeStream()

        let task = session.dataTask(with: request) { data, response, error in
            if let error {
                sink.finish(throwing: error)
                return
            }
            guard let http = response as? HTTPURLResponse else {
                sink.finish(throwing: Failure.notHTTP)
                return
            }
            let body = data ?? Data()
            guard http.statusCode == 200 else {
                sink.finish(throwing: Failure.badStatus(code: http.statusCode, body: text(body)))
                return
            }
            sink.yield(body)
            sink.finish()
        }
        task.resume()

        do {
            return try await withTaskCancellationHandler { () async throws -> Data in
                for try await body in stream { return body }
                // Finished without yielding: URLSession gave us neither body nor error,
                // which only happens when the task was cancelled out from under us.
                throw CancellationError()
            } onCancel: {
                task.cancel()
            }
        } catch let error as URLError where error.code == .cancelled {
            // So callers can tell "the user stopped it" from "the engine broke".
            throw CancellationError()
        }
    }

    // MARK: Response decoding

    private static func images(from body: Data) throws -> [Data] {
        guard let root = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any],
              let raw = root["images"] as? [Any], !raw.isEmpty
        else {
            throw Failure.noImages(humanMessage(from: text(body)))
        }

        var out: [Data] = []
        out.reserveCapacity(raw.count)
        for (index, entry) in raw.enumerated() {
            guard let encoded = entry as? String,
                  let png = decodeBase64(encoded),
                  !png.isEmpty
            else {
                // Never hand back empty Data — a zero-byte "image" would be written to the
                // gallery and only fail much later, far from the cause.
                throw Failure.badBase64(index: index)
            }
            out.append(png)
        }
        return out
    }

    /// sd-server returns bare base64, but A1111 clients and proxies routinely wrap it in a
    /// `data:` URL. Accept both — decoding the header as payload corrupts the PNG.
    static func stripBase64Prefix(_ encoded: String) -> String {
        let trimmed = encoded.trimmingCharacters(in: .whitespacesAndNewlines)
        // ':' is not in the base64 alphabet, so this can never fire on real payload.
        guard trimmed.hasPrefix("data:"), let comma = trimmed.firstIndex(of: ",") else {
            return trimmed
        }
        return String(trimmed[trimmed.index(after: comma)...])
    }

    /// `.ignoreUnknownCharacters` because line-wrapped base64 is legal in a data URL. It
    /// also means junk decodes to *something*, which is why callers check for empty.
    static func decodeBase64(_ encoded: String) -> Data? {
        Data(base64Encoded: stripBase64Prefix(encoded), options: [.ignoreUnknownCharacters])
    }

    /// Bounded body-to-text. A server that returns megabytes of failure must not be able
    /// to balloon an error message.
    private static func text(_ data: Data) -> String {
        String(decoding: data.prefix(32_768), as: UTF8.self)
    }

    /// Pulls the readable part out of an sd-server error body.
    ///
    /// sd-server answers `{"error":"server_error","message":"<the real detail>"}`, so
    /// `message` must win over `error` — the other order yields only "server_error".
    static func humanMessage(from body: String) -> String {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }

        if let data = trimmed.data(using: .utf8),
           let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            for key in ["message", "detail", "error"] {
                if let message = root[key] as? String, !message.isEmpty { return message }
                if let nested = root[key] as? [String: Any],
                   let message = nested["message"] as? String, !message.isEmpty {
                    return message
                }
            }
        }
        return String(trimmed.prefix(400))
    }

    // MARK: PNG geometry

    /// PNG dimensions straight out of the IHDR header: an 8-byte signature, a 4-byte
    /// chunk length, the tag "IHDR", then width and height as big-endian u32s at 16 and 20.
    ///
    /// Reading the header beats decoding the image — a gallery of 1024×1024 PNGs would
    /// otherwise cost hundreds of megabytes just to label thumbnails.
    static func pngSize(_ data: Data) -> (width: Int, height: Int)? {
        guard data.count >= 24 else { return nil }
        // The raw-buffer overload, spelled out: the deprecated typed-pointer one is still
        // visible and makes the call ambiguous otherwise.
        return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> (width: Int, height: Int)? in
            let signature: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
            for i in 0..<8 where raw[i] != signature[i] { return nil }
            // IHDR is required to be the first chunk; anything else is not a PNG we can
            // read dimensions out of, whatever else it may be.
            guard raw[12] == 0x49, raw[13] == 0x48, raw[14] == 0x44, raw[15] == 0x52 else {
                return nil
            }
            func be32(_ offset: Int) -> Int {
                (Int(raw[offset]) << 24) | (Int(raw[offset + 1]) << 16)
                    | (Int(raw[offset + 2]) << 8) | Int(raw[offset + 3])
            }
            let width = be32(16)
            let height = be32(20)
            guard width > 0, height > 0 else { return nil }
            return (width, height)
        }
    }

    // MARK: Self-check

    /// Returns a list of failure descriptions; `[]` means every case passed.
    static func selfCheck() -> [String] {
        var failures: [String] = []

        // --- auto tile size (measured: 512 wins at 512px, loses at 1024px) ---
        func png(_ w: Int, _ h: Int) -> URL {
            // Minimal PNG: signature + IHDR is all autoTileSize reads.
            var d = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
            d.append(contentsOf: [0, 0, 0, 13])
            d.append(Data("IHDR".utf8))
            d.append(contentsOf: withUnsafeBytes(of: UInt32(w).bigEndian, Array.init))
            d.append(contentsOf: withUnsafeBytes(of: UInt32(h).bigEndian, Array.init))
            let u = FileManager.default.temporaryDirectory
                .appending(path: "solidchat-tile-check-\(w)x\(h).png")
            try? d.write(to: u)
            return u
        }
        for (w, h, want) in [(512, 512, 512), (256, 256, 512), (512, 384, 512),
                             (1024, 1024, 128), (768, 512, 128)] {
            let u = png(w, h)
            defer { try? FileManager.default.removeItem(at: u) }
            let got = autoTileSize(for: u)
            if got != want { failures.append("tile: \(w)x\(h) -> \(got), expected \(want)") }
        }
        // An unreadable path must fall back, not crash.
        if autoTileSize(for: URL(filePath: "/nonexistent/x.png")) != 128 {
            failures.append("tile: missing file should fall back to 128")
        }


        // A hand-built 29-byte PNG head: signature + a complete IHDR chunk header.
        func header(width: UInt32, height: UInt32, tag: [UInt8] = [0x49, 0x48, 0x44, 0x52]) -> Data {
            func be(_ v: UInt32) -> [UInt8] {
                [UInt8(truncatingIfNeeded: v >> 24), UInt8(truncatingIfNeeded: v >> 16),
                 UInt8(truncatingIfNeeded: v >> 8), UInt8(truncatingIfNeeded: v)]
            }
            var bytes: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
            bytes += [0x00, 0x00, 0x00, 0x0D]        // IHDR payload is always 13 bytes
            bytes += tag
            bytes += be(width) + be(height)
            bytes += [8, 6, 0, 0, 0]                 // 8-bit RGBA, no interlace
            return Data(bytes)
        }

        // MARK: prefix stripping

        func expectStrip(_ label: String, _ input: String, _ want: String) {
            let got = stripBase64Prefix(input)
            if got != want {
                failures.append("stripBase64Prefix \(label): got \"\(got)\", want \"\(want)\"")
            }
        }

        expectStrip("raw base64 untouched", "iVBORw0KGgo=", "iVBORw0KGgo=")
        expectStrip("png data url", "data:image/png;base64,iVBORw0KGgo=", "iVBORw0KGgo=")
        expectStrip("jpeg data url", "data:image/jpeg;base64,AAAA", "AAAA")
        expectStrip("bare data url", "data:,AAAA", "AAAA")
        expectStrip("surrounding whitespace", "  iVBORw0KGgo=\n", "iVBORw0KGgo=")
        expectStrip("data prefix with no comma", "data:image/png;base64", "data:image/png;base64")
        expectStrip("empty", "", "")

        // MARK: decoding

        let png = header(width: 640, height: 360)
        let encoded = png.base64EncodedString()
        for (label, candidate) in [("raw", encoded),
                                   ("data url", "data:image/png;base64," + encoded),
                                   ("line wrapped", encoded.split(separator: "").joined(separator: "\n"))] {
            guard let decoded = decodeBase64(candidate) else {
                failures.append("decodeBase64 \(label): returned nil")
                continue
            }
            if decoded != png {
                failures.append("decodeBase64 \(label): decoded \(decoded.count) bytes, want \(png.count)")
            }
        }
        // Junk must not masquerade as an image; `images(from:)` relies on it being empty.
        if let junk = decodeBase64("!!!!"), !junk.isEmpty {
            failures.append("decodeBase64 junk: produced \(junk.count) bytes, want none")
        }

        // MARK: pngSize

        func expectSize(_ label: String, _ data: Data, _ want: (width: Int, height: Int)?) {
            let got = pngSize(data)
            if got?.width != want?.width || got?.height != want?.height {
                failures.append("pngSize \(label): got \(String(describing: got)), want \(String(describing: want))")
            }
        }

        expectSize("640x360", png, (640, 360))
        expectSize("1x1", header(width: 1, height: 1), (1, 1))
        expectSize("large", header(width: 65_535, height: 4_096), (65_535, 4_096))
        expectSize("zero width", header(width: 0, height: 8), nil)
        expectSize("zero height", header(width: 8, height: 0), nil)
        expectSize("truncated", png.prefix(23), nil)
        expectSize("empty", Data(), nil)
        expectSize("first chunk is not IHDR",
                   header(width: 8, height: 8, tag: [0x49, 0x44, 0x41, 0x54]), nil)

        var corrupt = png
        corrupt[0] = 0x88
        expectSize("bad signature", corrupt, nil)

        // A Data slice does not start at index 0. Indexing it as if it did would read the
        // wrong bytes — the exact bug this case exists to catch.
        var padded = Data([0xFF])
        padded.append(png)
        expectSize("offset slice", padded.dropFirst(), (640, 360))

        // MARK: base64 -> pngSize, end to end

        if let round = decodeBase64("data:image/png;base64," + encoded),
           let size = pngSize(round) {
            if size.width != 640 || size.height != 360 {
                failures.append("round trip: got \(size.width)x\(size.height), want 640x360")
            }
        } else {
            failures.append("round trip: prefixed base64 did not survive to pngSize")
        }

        // MARK: error extraction

        // Verbatim shape of a real sd-server 500, captured from the running binary.
        let serverError = "{\"error\":\"server_error\",\"message\":\"parse error at line 1\"}"
        if humanMessage(from: serverError) != "parse error at line 1" {
            failures.append("humanMessage: \"message\" must win over \"error\", got \"\(humanMessage(from: serverError))\"")
        }
        if humanMessage(from: "plain text failure") != "plain text failure" {
            failures.append("humanMessage: non-JSON body should pass through")
        }
        if !humanMessage(from: "   ").isEmpty {
            failures.append("humanMessage: blank body should be empty")
        }

        return failures
    }
}
