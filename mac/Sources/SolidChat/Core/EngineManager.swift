import Darwin
import Foundation
import Observation

// MARK: - Paths

/// Every filesystem location the app depends on, in one place.
///
/// Each path can be repointed by the user through `UserDefaults` (see `Paths.Key`);
/// when no override is set the machine defaults below are used.
enum Paths {
    /// UserDefaults keys a Settings screen can bind to with `@AppStorage`.
    enum Key {
        static let llamaServer = "paths.llamaServer"
        static let venvPython = "paths.venvPython"
        static let modelsRoot = "paths.modelsRoot"
    }


    /// Resolved against the *current* user's home directory rather than a baked-in
    /// absolute path, so the app works for whoever runs it — and so no developer's
    /// username ends up in the published source.
    private static func inHome(_ relative: String) -> String {
        NSHomeDirectory() + "/" + relative
    }

    static var defaultLlamaServer: String { inHome("prism-llama/prism/llama-prism-b9599-9ca265a/llama-server") }
    static var defaultVenvPython: String { inHome("github/Solid-LM/.venv/bin/python") }

    /// One engine at a time, always on this port.
    static let enginePort = 8181
    static let engineBaseURL = URL(string: "http://127.0.0.1:\(enginePort)")!

    static var llamaServer: URL {
        override(Key.llamaServer) ?? URL(filePath: defaultLlamaServer)
    }

    static var venvPython: URL {
        override(Key.venvPython) ?? URL(filePath: defaultVenvPython)
    }

    static var modelsRoot: URL {
        override(Key.modelsRoot)
            ?? FileManager.default.homeDirectoryForCurrentUser.appending(path: "Downloads/LM_Studio_Models")
    }

    /// llama.cpp's dylibs sit beside the binary, so the child needs `DYLD_LIBRARY_PATH` pointed here.
    static var llamaServerDirectory: URL { llamaServer.deletingLastPathComponent() }

    /// `~/Library/Application Support/SolidChat`, created on first access.
    static var appSupport: URL {
        let fm = FileManager.default
        let base = (try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                appropriateFor: nil, create: true))
            ?? fm.homeDirectoryForCurrentUser.appending(path: "Library/Application Support")
        let dir = base.appending(path: "SolidChat")
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    /// The engine child's stdout + stderr. Truncated on every load.
    static var engineLog: URL { appSupport.appending(path: "engine.log") }

    private static func override(_ key: String) -> URL? {
        guard let raw = UserDefaults.standard.string(forKey: key) else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return URL(filePath: (trimmed as NSString).expandingTildeInPath)
    }
}

// MARK: - Private support types

/// A fully validated command line. Built before anything is killed.
private struct LaunchSpec {
    var executable: URL
    var arguments: [String]
    var environment: [String: String]

    /// argv as a user would type it, for the log header.
    var commandLine: String {
        ([executable.path] + arguments)
            .map { $0.contains(" ") ? "\"\($0)\"" : $0 }
            .joined(separator: " ")
    }
}

private struct EngineLaunchError: Error {
    var message: String
    init(_ message: String) { self.message = message }
}

/// The result of a child swap, tagged with the generation that produced it: releasing the
/// swap gate is a suspension point, so a queued load may already have moved on.
private enum EngineSwapOutcome {
    case spawned(generation: Int)
    case failed(generation: Int, message: String)
}

private struct EnginePortHolder {
    var pid: pid_t
    var command: String

    /// A leftover engine from a previous run of this app is ours to clean up.
    ///
    /// Deliberately narrow: this decides what we SIGKILL. A bare "python" match would
    /// kill any unrelated interpreter that happens to hold the port — someone's Jupyter
    /// kernel or dev server — so we require a marker only our own invocations carry.
    var looksLikeOurEngine: Bool {
        let c = command.lowercased()
        guard c.contains("llama-server")
                || c.contains("mlx_lm")
                || c.contains("vllm.entrypoints")
        else { return false }
        // …and it must be pointed at the port we actually drive.
        return c.contains(String(Paths.enginePort))
    }
}

/// Thin `Process` wrapper. MainActor-isolated so the non-Sendable `Process` never escapes.
@MainActor
private final class EngineChildProcess {
    private let process = Process()
    private let sink: FileHandle

    init(executable: URL, arguments: [String], environment: [String: String], logURL: URL) throws {
        // Truncate whatever the previous run left behind, then hand the child the fd.
        FileManager.default.createFile(atPath: logURL.path, contents: Data())
        sink = try FileHandle(forWritingTo: logURL)
        process.executableURL = executable
        process.arguments = arguments
        process.environment = environment
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = sink
        process.standardError = sink
    }

    func run() throws { try process.run() }

    var isRunning: Bool { process.isRunning }
    var pid: pid_t { process.processIdentifier }
    /// Only meaningful once `isRunning` is false — Foundation traps if asked too early.
    var terminationStatus: Int32 { process.isRunning ? 0 : process.terminationStatus }

    /// SIGTERM, then SIGKILL if it is still alive 5 s later. Safe to call twice.
    func terminate() async {
        guard process.isRunning else { closeSink(); return }
        process.terminate()
        for _ in 0..<50 {
            if !process.isRunning { break }
            try? await Task.sleep(for: .milliseconds(100))
        }
        if process.isRunning {
            _ = kill(process.processIdentifier, SIGKILL)
            for _ in 0..<20 {
                if !process.isRunning { break }
                try? await Task.sleep(for: .milliseconds(100))
            }
        }
        closeSink()
    }

    /// Blocking variant for `applicationWillTerminate`, where there is no time to await.
    /// Bounded to roughly 3 s so quitting never hangs.
    func terminateBlocking() {
        guard process.isRunning else { closeSink(); return }
        process.terminate()
        for _ in 0..<60 {
            if !process.isRunning { break }
            Thread.sleep(forTimeInterval: 0.05)
        }
        if process.isRunning {
            _ = kill(process.processIdentifier, SIGKILL)
            process.waitUntilExit()
        }
        closeSink()
    }

    private func closeSink() { try? sink.close() }
}

// MARK: - EngineManager

/// The single owner of the child engine process. Exactly one model is loaded at a time:
/// 24 GB of unified memory does not stretch to two.
@Observable
@MainActor
final class EngineManager {

    // MARK: Observable state

    var state: EngineState = .idle
    var model: LocalModel?
    var engine: EngineKind?
    var config: LoadConfig = LoadConfig()
    var loadedAt: Date?
    /// Engine stdout + stderr, newest last, capped at 500 lines.
    var logLines: [String] = []
    /// Set when a load was rejected *before* anything was killed (missing binary, missing
    /// model, port taken). If a model was already serving it is still serving.
    var lastLoadError: String?

    // MARK: Constants

    nonisolated static let port = Paths.enginePort
    nonisolated static let baseURL = Paths.engineBaseURL

    private static let maxLogLines = 500
    private static let healthTimeout: TimeInterval = 300
    private static let failureTailLines = 15

    // MARK: Private state

    @ObservationIgnored private var child: EngineChildProcess?
    /// Bumped by every load()/unload(). A poll loop whose generation is stale exits
    /// without touching state — otherwise a slow load would clobber a newer one.
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private var tailTask: Task<Void, Never>?
    /// Held for the length of a child swap. Two loads racing would otherwise each spawn a
    /// server and one would be orphaned, still holding the port and its weight in RAM.
    @ObservationIgnored private var swapping = false
    @ObservationIgnored private var logURL: URL = Paths.engineLog
    @ObservationIgnored private var logOffset: UInt64 = 0
    @ObservationIgnored private var logRemainder = ""

    init() {}

    // MARK: - Loading

    /// Start `model` on `engine` and return once it is serving, has failed, or was superseded.
    ///
    /// The command line is validated first: a request that cannot possibly work must not
    /// cost the user the model they already have loaded.
    func load(_ model: LocalModel, engine: EngineKind, config: LoadConfig) async {
        lastLoadError = nil

        let spec: LaunchSpec
        do {
            spec = try Self.launchSpec(model: model, engine: engine, config: config)
        } catch {
            let message = Self.describe(error)
            lastLoadError = message
            // Nothing has been killed. A running engine keeps serving; only an idle or
            // already-failed engine takes on the error.
            if !state.isReady { state = .failed(message) }
            return
        }

        switch await swapIn(spec: spec, model: model, engine: engine, config: config) {
        case .failed(let gen, let message):
            guard gen == generation else { return }   // a newer load already took over
            lastLoadError = message
            state = .failed(message)
        case .spawned(let gen):
            await waitUntilHealthy(gen: gen)
        }
    }

    /// Kill the old child and start the new one, with no other swap interleaved.
    /// Bounded work only — the health wait deliberately happens outside the exclusion.
    private func swapIn(spec: LaunchSpec,
                        model: LocalModel,
                        engine: EngineKind,
                        config: LoadConfig) async -> EngineSwapOutcome {
        await beginSwap()
        defer { endSwap() }

        await terminateChild()
        tailTask?.cancel()
        tailTask = nil
        logLines.removeAll()
        logOffset = 0
        logRemainder = ""
        logURL = Paths.engineLog
        self.model = nil
        self.engine = nil
        loadedAt = nil
        state = .loading

        generation &+= 1
        let gen = generation

        // A stale engine still bound to 8181 would make the new child exit on bind, or —
        // worse — leave us happily chatting with the wrong model.
        if let portProblem = await claimPort() {
            return .failed(generation: gen, message: portProblem)
        }

        append([spec.commandLine])
        do {
            let started = try EngineChildProcess(executable: spec.executable,
                                                 arguments: spec.arguments,
                                                 environment: spec.environment,
                                                 logURL: logURL)
            try started.run()
            child = started
        } catch {
            return .failed(generation: gen,
                           message: "could not start \(engine.label): \(Self.describe(error))")
        }

        self.model = model
        self.engine = engine
        self.config = config
        startLogTail(gen: gen)
        return .spawned(generation: gen)
    }

    /// Cooperative gate. Both the check and the claim run without an intervening suspension
    /// point, so on a single actor this cannot be raced.
    private func beginSwap() async {
        while swapping {
            try? await Task.sleep(for: .milliseconds(50))
        }
        swapping = true
    }

    private func endSwap() { swapping = false }

    /// Stop the child and go back to idle. Safe to call when nothing is loaded.
    func unload() async {
        await beginSwap()
        defer { endSwap() }

        generation &+= 1
        tailTask?.cancel()
        tailTask = nil
        await terminateChild()
        drainLog()
        model = nil
        engine = nil
        loadedAt = nil
        lastLoadError = nil
        state = .idle
    }

    /// Bounded, blocking shutdown for `applicationWillTerminate` — an orphaned engine would
    /// keep both the port and several GB of memory.
    func shutdownBlocking() {
        generation &+= 1
        tailTask?.cancel()
        tailTask = nil
        let running = child
        child = nil
        running?.terminateBlocking()
        state = .idle
    }

    func clearLog() {
        logLines.removeAll()
    }

    private func terminateChild() async {
        guard let running = child else { return }
        // Detach first: a concurrent unload() must not wait on the same corpse.
        child = nil
        await running.terminate()
    }

    // MARK: - Health

    /// Poll until the engine answers, the child dies, or five minutes pass. Big models
    /// mmap-page slowly on a cold first load, which is what the long timeout is for.
    private func waitUntilHealthy(gen: Int) async {
        let deadline = Date.now.addingTimeInterval(Self.healthTimeout)
        while Date.now < deadline {
            guard gen == generation else { return }

            if let running = child, !running.isRunning {
                drainLog()
                guard gen == generation else { return }
                let status = running.terminationStatus
                child = nil
                tailTask?.cancel()
                tailTask = nil
                await running.terminate()   // already dead: just closes the log fd
                // The last lines are where llama.cpp explains itself.
                let tail = logLines.suffix(Self.failureTailLines).joined(separator: "\n")
                let label = engine?.label ?? "engine"
                state = .failed(tail.isEmpty ? "\(label) exited immediately (status \(status))" : tail)
                loadedAt = nil
                return
            }

            if await Self.engineAnswers() {
                guard gen == generation else { return }
                loadedAt = .now
                state = .ready
                return
            }

            guard gen == generation else { return }
            try? await Task.sleep(for: .seconds(1))
        }

        guard gen == generation else { return }
        drainLog()
        // The child is still alive and holding the model's memory — a wedged engine we
        // gave up on must be killed, or it keeps the RAM and port 8181 forever.
        await terminateChild()
        tailTask?.cancel()
        tailTask = nil
        state = .failed("health check timed out after \(Int(Self.healthTimeout / 60)) minutes — see the engine log")
    }

    /// llama.cpp serves `/health`; mlx_lm and vLLM only have `/v1/models`.
    private nonisolated static func engineAnswers() async -> Bool {
        if await status(of: "/health") == 200 { return true }
        return await status(of: "/v1/models") == 200
    }

    private nonisolated static func status(of path: String) async -> Int? {
        var request = URLRequest(url: baseURL.appending(path: path))
        request.httpMethod = "GET"
        request.timeoutInterval = 2
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode
        } catch {
            return nil
        }
    }

    // MARK: - Port

    /// Returns `nil` once 127.0.0.1:8181 is bindable, or a human-readable reason it is not.
    private func claimPort() async -> String? {
        if Self.portIsFree() { return nil }

        // A child we just SIGTERMed may need a moment to let go of the socket.
        for _ in 0..<10 {
            try? await Task.sleep(for: .milliseconds(200))
            if Self.portIsFree() { return nil }
        }

        let holders = Self.portHolders()
        let ours = holders.filter(\.looksLikeOurEngine)
        if !holders.isEmpty, ours.count == holders.count {
            for holder in ours {
                append(["reclaiming port \(Self.port) from leftover \(holder.command) (pid \(holder.pid))"])
                _ = kill(holder.pid, SIGTERM)
            }
            for _ in 0..<15 {
                try? await Task.sleep(for: .milliseconds(200))
                if Self.portIsFree() { return nil }
            }
            for holder in ours { _ = kill(holder.pid, SIGKILL) }
            for _ in 0..<10 {
                try? await Task.sleep(for: .milliseconds(200))
                if Self.portIsFree() { return nil }
            }
        }

        if holders.isEmpty {
            return "port \(Self.port) is already in use by another process. Quit it and load again."
        }
        let who = holders.map { "\($0.command) (pid \($0.pid))" }.joined(separator: ", ")
        return "port \(Self.port) is held by \(who) and would not give it up. Quit it and load again."
    }

    /// Bind-test the engine port. `SO_REUSEADDR` keeps sockets lingering in TIME_WAIT from
    /// reading as "in use", so only a live listener returns false.
    private nonisolated static func portIsFree() -> Bool {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return true }  // cannot tell — let the child try
        defer { Darwin.close(fd) }

        var reuse: Int32 = 1
        _ = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(port).bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")

        let bound = withUnsafePointer(to: &addr) { raw in
            raw.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                Darwin.bind(fd, sa, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return bound == 0
    }

    /// Best-effort: who is listening on the engine port. Only ever called on the failure
    /// path, where a short synchronous `lsof` is cheaper than a confusing error.
    private nonisolated static func portHolders() -> [EnginePortHolder] {
        let lsof = URL(filePath: "/usr/sbin/lsof")
        guard FileManager.default.isExecutableFile(atPath: lsof.path) else { return [] }

        let process = Process()
        process.executableURL = lsof
        process.arguments = ["-nP", "+c", "0", "-iTCP:\(port)", "-sTCP:LISTEN", "-Fpc"]
        let out = Pipe()
        process.standardOutput = out
        process.standardError = FileHandle.nullDevice
        guard (try? process.run()) != nil else { return [] }
        let data = (try? out.fileHandleForReading.readToEnd()) ?? Data()
        process.waitUntilExit()

        var holders: [EnginePortHolder] = []
        var seen = Set<pid_t>()
        var pid: pid_t?
        for line in String(decoding: data, as: UTF8.self).split(separator: "\n") {
            if line.hasPrefix("p") {
                pid = pid_t(line.dropFirst())
            } else if line.hasPrefix("c"), let current = pid, !seen.contains(current) {
                seen.insert(current)
                holders.append(EnginePortHolder(pid: current, command: String(line.dropFirst())))
            }
        }
        return holders
    }

    // MARK: - Command line

    private nonisolated static func launchSpec(model: LocalModel,
                                               engine: EngineKind,
                                               config: LoadConfig) throws -> LaunchSpec {
        let fm = FileManager.default
        guard !model.path.isEmpty, fm.fileExists(atPath: model.path) else {
            throw EngineLaunchError("\(model.name) is not on disk — nothing at \(model.path)")
        }
        guard config.ctx > 0 else {
            throw EngineLaunchError("context length must be at least 1 token")
        }

        var environment = ProcessInfo.processInfo.environment

        switch engine {
        case .llama:
            let binary = Paths.llamaServer
            guard fm.isExecutableFile(atPath: binary.path) else {
                throw EngineLaunchError("llama-server not found at \(binary.path). Point \"\(Paths.Key.llamaServer)\" at your build in Settings.")
            }
            // The dylibs live next to the binary and are not in the default search path.
            environment["DYLD_LIBRARY_PATH"] = Paths.llamaServerDirectory.path
            var arguments = ["-m", model.path,
                             "--host", "127.0.0.1",
                             "--port", String(Paths.enginePort),
                             "-ngl", String(config.gpuLayers),   // full offload: 3.99 vs 37.9 tok/s
                             "-c", String(config.ctx),
                             "--jinja"]
            if config.flashAttention { arguments += ["-fa", "on"] }
            if config.quantizeKVCache { arguments += ["-ctk", "q8_0", "-ctv", "q8_0"] }

            // Speculative decoding, when the model shipped an MTP draft head.
            //
            // Generation is memory-bandwidth-bound: tok/s is weights-read-per-token
            // divided into the bus, and no flag changes that — measured here, KV
            // quantisation, thread count and mmap all move it by under 3%. A draft
            // head does change it, because the big model verifies several drafted
            // tokens per weight read.
            //
            // Measured, Gemma4-12B Q4_K_M, 200 tokens, mean of 3:
            //     no draft                14.14 tok/s
            //     MTP draft, n-max 3      25.51 tok/s   <- 1.80x
            //     n-max 2 / 4 / 5 / 6     22.0 / 21.6 / 17.7 / 15.5
            // Depth 3 wins: past that the acceptance rate falls faster than the
            // batching gains, and every rejected token is a wasted verify pass.
            //
            // `-fit off` is required: the memory-fitting probe fails to build a
            // context for the draft ("requires ctx_other to be set") and the server
            // then refuses every request with a 503.
            if config.speculativeDecoding, let draft = model.draftPath,
               FileManager.default.fileExists(atPath: draft) {
                arguments += ["-fit", "off",
                              "--spec-type", "draft-mtp",
                              "-md", draft,
                              "-ngld", "999",
                              "--spec-draft-n-max", String(config.draftTokens)]
            }
            return LaunchSpec(executable: binary, arguments: arguments, environment: environment)

        case .mlx:
            let python = try venvPython()
            // Python block-buffers stdout once it is not a TTY, which would strand the log
            // view several KB behind the engine.
            environment["PYTHONUNBUFFERED"] = "1"
            // NOTE for whoever writes the chat client: mlx_lm.server reads the request's
            // "model" field as a HuggingFace repo id and will silently re-download a model
            // that is already on disk. Send `model.path`, never "publisher/name".
            return LaunchSpec(executable: python,
                              arguments: ["-m", "mlx_lm", "server",
                                          "--model", model.path,
                                          "--host", "127.0.0.1",
                                          "--port", String(Paths.enginePort)],
                              environment: environment)

        case .vllm:
            let python = try venvPython()
            environment["PYTHONUNBUFFERED"] = "1"
            if vllmIsMissing(python: python) {
                throw EngineLaunchError("""
                    vLLM is not installed in \(venvRoot(for: python).path), and on Apple Silicon it \
                    has no Metal backend — it would run on the CPU. Install it only if you know why:
                    uv pip install --python \(python.path) vllm
                    """)
            }
            return LaunchSpec(executable: python,
                              arguments: ["-m", "vllm.entrypoints.openai.api_server",
                                          "--model", model.path,
                                          "--host", "127.0.0.1",
                                          "--port", String(Paths.enginePort),
                                          "--max-model-len", String(config.ctx)],
                              environment: environment)
        }
    }

    private nonisolated static func venvPython() throws -> URL {
        let python = Paths.venvPython
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            throw EngineLaunchError("python not found at \(python.path). Point \"\(Paths.Key.venvPython)\" at your virtualenv in Settings.")
        }
        return python
    }

    private nonisolated static func venvRoot(for python: URL) -> URL {
        python.deletingLastPathComponent().deletingLastPathComponent()
    }

    /// True only when site-packages was positively inspected and vLLM is not in it. An
    /// unfamiliar layout returns false so the child gets to speak for itself — it will exit
    /// within a second with "No module named vllm", which the load path already surfaces.
    private nonisolated static func vllmIsMissing(python: URL) -> Bool {
        let fm = FileManager.default
        let lib = venvRoot(for: python).appending(path: "lib")
        guard let versions = try? fm.contentsOfDirectory(atPath: lib.path) else { return false }

        var inspected = false
        for version in versions where version.hasPrefix("python") {
            let sitePackages = lib.appending(path: version).appending(path: "site-packages")
            guard let entries = try? fm.contentsOfDirectory(atPath: sitePackages.path) else { continue }
            inspected = true
            if entries.contains(where: { $0 == "vllm" || $0.hasPrefix("vllm-") || $0.hasPrefix("vllm.") }) {
                return false
            }
        }
        return inspected
    }

    private nonisolated static func describe(_ error: Error) -> String {
        if let launch = error as? EngineLaunchError { return launch.message }
        return (error as NSError).localizedDescription
    }

    // MARK: - Log tail

    private func startLogTail(gen: Int) {
        tailTask?.cancel()
        tailTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self, gen == self.generation else { return }
                self.drainLog()
                try? await Task.sleep(for: .milliseconds(400))
            }
        }
    }

    /// Pull whatever the child has written since the last pass. Reads are a few KB at a
    /// time, so doing them here rather than off-actor keeps the state handling trivial.
    private func drainLog() {
        guard let handle = try? FileHandle(forReadingFrom: logURL) else { return }
        defer { try? handle.close() }

        do {
            let end = try handle.seekToEnd()
            if end < logOffset {        // truncated behind our back
                logOffset = 0
                logRemainder = ""
            }
            guard end > logOffset else { return }
            try handle.seek(toOffset: logOffset)
            guard let data = try handle.readToEnd(), !data.isEmpty else { return }
            logOffset += UInt64(data.count)

            var text = logRemainder + String(decoding: data, as: UTF8.self)
            text = text.replacingOccurrences(of: "\r\n", with: "\n")
            text = text.replacingOccurrences(of: "\r", with: "\n")   // llama.cpp redraws progress with \r
            var parts = text.components(separatedBy: "\n")
            logRemainder = parts.removeLast()
            append(parts.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty })
        } catch {
            return
        }
    }

    private func append(_ lines: [String]) {
        guard !lines.isEmpty else { return }
        logLines.append(contentsOf: lines)
        if logLines.count > Self.maxLogLines {
            logLines.removeFirst(logLines.count - Self.maxLogLines)
        }
    }
}
