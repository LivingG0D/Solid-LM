import Darwin
import Foundation
import Observation
import Synchronization

// MARK: - Paths

/// Where the stable-diffusion.cpp binaries and the image models live.
///
/// Deliberately separate from `Paths`: the image side is optional, and a user who
/// never generates an image should never be asked where sd.cpp is. Every entry can
/// be repointed through `UserDefaults` (see `ImagePaths.Key`).
enum ImagePaths {
    /// UserDefaults keys a Settings screen can bind to with `@AppStorage`.
    enum Key {
        static let sdServer = "paths.sdServer"
        static let sdCLI = "paths.sdCLI"
        static let imageModelsRoot = "paths.imageModelsRoot"
    }


    /// Home-relative so the published source carries no developer username and the
    /// app resolves correctly for whoever runs it. Override in Settings > Paths.
    private static func inHome(_ relative: String) -> String { NSHomeDirectory() + "/" + relative }

    static var defaultServer: String { inHome("prism-llama/sdcpp/bin/sd-server") }
    static var defaultCLI: String { inHome("prism-llama/sdcpp/bin/sd-cli") }

    /// 8181 belongs to the LLM engine. Both servers can be up at once, so this must
    /// never move onto that port — a collision would break chat and images together.
    static let port: Int = 8182
    static let baseURL = URL(string: "http://127.0.0.1:\(port)")!

    static var server: URL {
        override(Key.sdServer) ?? URL(filePath: defaultServer)
    }

    static var cli: URL {
        override(Key.sdCLI) ?? URL(filePath: defaultCLI)
    }

    /// `~/Library/Application Support/SolidChat/gallery`, created on first access.
    /// Every generated PNG lands here; `GeneratedImage` stores only the file name.
    static var galleryDir: URL {
        let dir = Paths.appSupport.appending(path: "gallery")
        let fm = FileManager.default
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    /// Folders the image scanner walks. The override accepts several roots separated
    /// by newlines or commas, because people keep checkpoints in more than one place.
    ///
    /// The default is the *shared* LM Studio root: image checkpoints live in its
    /// `_image-models` subfolder, which the LLM scanner skips.
    static var imageModelRoots: [URL] {
        if let raw = UserDefaults.standard.string(forKey: Key.imageModelsRoot) {
            let parts = raw.split(whereSeparator: { $0 == "\n" || $0 == "," })
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                .map { ($0 as NSString).expandingTildeInPath }
            var seen = Set<String>()
            let roots = parts.filter { seen.insert($0).inserted }.map { URL(filePath: $0) }
            if !roots.isEmpty { return roots }
        }
        // Scoped to the image subfolder, NOT the shared models root. A diffusion
        // checkpoint and an LLM checkpoint are both multi-gigabyte .gguf files with
        // no reliable way to tell them apart by name or size — pointed at the shared
        // root the scanner listed all 48 local LLMs as image models, with a 26B Gemma
        // labelled "FLUX". Loading one would start sd-server against a text model.
        let scoped = Paths.modelsRoot.appending(path: "_image-models", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: scoped, withIntermediateDirectories: true)
        return [scoped]
    }

    private static func override(_ key: String) -> URL? {
        guard let raw = UserDefaults.standard.string(forKey: key) else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return URL(filePath: (trimmed as NSString).expandingTildeInPath)
    }
}

// MARK: - Private support types

/// A fully validated command line. Built before anything is killed.
private struct ImageLaunchSpec {
    var executable: URL
    var arguments: [String]

    /// argv as a user would type it, for the log header.
    var commandLine: String {
        ([executable.path] + arguments)
            .map { $0.contains(" ") ? "\"\($0)\"" : $0 }
            .joined(separator: " ")
    }
}

private struct ImageLaunchError: Error {
    var message: String
    init(_ message: String) { self.message = message }
}

/// The result of a child swap, tagged with the generation that produced it: releasing
/// the swap gate is a suspension point, so a queued load may already have moved on.
private enum ImageSwapOutcome {
    case spawned(generation: Int)
    case failed(generation: Int, message: String)
}

private struct ImagePortHolder {
    var pid: pid_t
    var command: String

    /// A leftover sd-server from a previous run of this app is ours to clean up.
    ///
    /// Deliberately narrow: this decides what we SIGKILL. It must be sd-server *and*
    /// pointed at the port we drive, so an unrelated process is never touched.
    var looksLikeOurServer: Bool {
        let c = command.lowercased()
        return c.contains("sd-server") && c.contains(String(ImagePaths.port))
    }
}

// MARK: - ImageEngine

/// The single owner of the sd-server child process — the mirror image of
/// `EngineManager`, on port 8182 instead of 8181.
///
/// One checkpoint at a time. SDXL is ~7 GB and FLUX up to 24 GB; with the LLM engine
/// also resident there is no room to keep a second one warm.
@Observable
@MainActor
final class ImageEngine {

    // MARK: Observable state

    var state: EngineState = .idle
    var model: ImageModel?
    var loadedAt: Date?
    /// sd-server stdout + stderr merged, newest last, capped at 500 lines.
    var logLines: [String] = []
    /// Set when a load was rejected *before* anything was killed (missing binary,
    /// missing checkpoint, port taken). If a model was already serving it still is.
    var lastLoadError: String?

    // MARK: Constants

    nonisolated static let port = ImagePaths.port
    nonisolated static let baseURL = ImagePaths.baseURL

    nonisolated private static let maxLogLines = 500
    /// Slack between drains. sd.cpp with `-v` is chatty; anything past this is noise
    /// we would throw away 400 ms later anyway.
    nonisolated private static let maxPending = 2_000
    private static let healthTimeout: TimeInterval = 300
    private static let failureTailLines = 15

    // MARK: Private state

    /// Lines the child has written but the UI has not seen. Written from the pipe
    /// reader's queue, drained on the main actor — hence the mutex rather than a
    /// per-line actor hop, which would be free to reorder the log.
    nonisolated private let pending = Mutex<[String]>([])

    @ObservationIgnored private var runner: ProcessRunner?
    /// Bumped by every load()/unload(). A poll loop or callback whose generation is
    /// stale exits without touching state — otherwise a slow load clobbers a newer one.
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private var tailTask: Task<Void, Never>?
    /// Held for the length of a child swap. Two loads racing would otherwise each
    /// spawn a server, and the orphan would keep the port and its weight in RAM.
    @ObservationIgnored private var swapping = false

    init() {}

    // MARK: - Loading

    /// Start `model` in sd-server and return once it is serving, has failed, or was
    /// superseded.
    ///
    /// The command line is validated first: a request that cannot possibly work must
    /// not cost the user the checkpoint they already have loaded.
    func load(_ model: ImageModel) async {
        lastLoadError = nil

        let spec: ImageLaunchSpec
        do {
            spec = try Self.launchSpec(model: model)
        } catch {
            let message = Self.describe(error)
            lastLoadError = message
            // Nothing has been killed. A running server keeps serving; only an idle
            // or already-failed engine takes on the error.
            if !state.isReady { state = .failed(message) }
            return
        }

        switch await swapIn(spec: spec, model: model) {
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
    private func swapIn(spec: ImageLaunchSpec, model: ImageModel) async -> ImageSwapOutcome {
        await beginSwap()
        defer { endSwap() }

        await terminateRunner()
        tailTask?.cancel()
        tailTask = nil
        pending.withLock { $0.removeAll(keepingCapacity: true) }
        logLines.removeAll()
        self.model = nil
        loadedAt = nil
        state = .loading

        generation &+= 1
        let gen = generation

        // sd-server binds *after* the checkpoint is in memory, so a stale server on
        // 8182 would cost a full model load before the new child failed on bind.
        if let portProblem = await claimPort() {
            return .failed(generation: gen, message: portProblem)
        }

        append([spec.commandLine])
        let started = ProcessRunner(executable: spec.executable,
                                    arguments: spec.arguments,
                                    environment: [:])
        // Wired before start() so the checkpoint-loading lines are never missed —
        // when a load fails, those lines are the entire diagnosis.
        started.onLogLine = { [weak self] line in self?.enqueue(line) }
        started.onExit = { [weak self] code in
            Task { @MainActor in self?.childExited(gen: gen, code: code) }
        }
        do {
            try started.start()
        } catch {
            started.onLogLine = nil
            started.onExit = nil
            return .failed(generation: gen,
                           message: "could not start sd-server: \(Self.describe(error))")
        }
        runner = started

        self.model = model
        startLogTail(gen: gen)
        return .spawned(generation: gen)
    }

    /// Cooperative gate. Both the check and the claim run without an intervening
    /// suspension point, so on a single actor this cannot be raced.
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
        await terminateRunner()
        drainLog()
        model = nil
        loadedAt = nil
        lastLoadError = nil
        state = .idle
    }

    /// Bounded, blocking shutdown for `applicationWillTerminate` — an orphaned
    /// sd-server would keep port 8182 and several GB of memory.
    func shutdownBlocking() {
        generation &+= 1
        tailTask?.cancel()
        tailTask = nil
        let running = runner
        runner = nil
        if let running {
            running.onLogLine = nil
            running.onExit = nil
            Self.killGroupBlocking(running)
        }
        state = .idle
    }

    func clearLog() {
        logLines.removeAll()
    }

    /// SIGTERM the child's whole process group, SIGKILL it if it is still alive.
    /// Blocking by design — call it off the main actor.
    private func terminateRunner() async {
        guard let running = runner else { return }
        // Detach first: a concurrent unload() must not wait on the same corpse, and a
        // dying child's last lines must not land in the next child's log.
        runner = nil
        running.onLogLine = nil
        running.onExit = nil
        await Task.detached(priority: .userInitiated) { running.terminate() }.value
    }

    /// `applicationWillTerminate` has no time to await, and `ProcessRunner.terminate()`
    /// waits a full 5 s before escalating — too long to hold up a quit.
    nonisolated private static func killGroupBlocking(_ running: ProcessRunner) {
        guard let pid = running.processIdentifier else { return }
        signalGroup(SIGTERM, pid: pid)
        for _ in 0..<60 {
            if !running.isRunning { return }
            Thread.sleep(forTimeInterval: 0.05)
        }
        signalGroup(SIGKILL, pid: pid)
    }

    /// `ProcessRunner` spawns with `POSIX_SPAWN_SETPGROUP` and pgroup 0, so the child's
    /// pgid *is* its pid. The `getpgrp()` guard is the seatbelt: signalling our own
    /// group would take down the app.
    nonisolated private static func signalGroup(_ sig: Int32, pid: pid_t) {
        if pid > 1, pid != getpgrp(), kill(-pid, sig) == 0 { return }
        _ = kill(pid, sig)
    }

    // MARK: - Health

    /// Poll until sd-server answers, the child dies, or five minutes pass. A cold FLUX
    /// checkpoint is 24 GB off disk, which is what the long timeout is for.
    private func waitUntilHealthy(gen: Int) async {
        let deadline = Date.now.addingTimeInterval(Self.healthTimeout)
        while Date.now < deadline {
            guard gen == generation else { return }
            // childExited() already reported the failure and cleared the runner.
            guard let running = runner else { return }

            if !running.isRunning {
                drainLog()
                guard gen == generation else { return }
                runner = nil
                tailTask?.cancel()
                tailTask = nil
                reportExit(code: running.lastExitCode ?? -1)
                return
            }

            if await Self.serverAnswers() {
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
        // The child is still alive and holding the checkpoint's memory — a wedged
        // server we gave up on must be killed, or it keeps the RAM and 8182 forever.
        await terminateRunner()
        tailTask?.cancel()
        tailTask = nil
        state = .failed("sd-server did not answer within \(Int(Self.healthTimeout / 60)) minutes — see the image log")
    }

    /// The child died on its own. Reached from `ProcessRunner.onExit`, which covers the
    /// case the health loop cannot: a crash *after* the server went ready, where every
    /// later request would otherwise fail against a state that still reads "ready".
    private func childExited(gen: Int, code: Int32) {
        guard gen == generation, runner != nil else { return }
        drainLog()
        runner = nil
        tailTask?.cancel()
        tailTask = nil
        reportExit(code: code)
    }

    /// The last lines are where sd.cpp explains itself — a missing VAE, an unsupported
    /// weight type, an out-of-memory Metal allocation.
    private func reportExit(code: Int32) {
        let tail = logLines.suffix(Self.failureTailLines).joined(separator: "\n")
        let how = code < 0 ? "killed by signal \(-code)" : "status \(code)"
        state = .failed(tail.isEmpty ? "sd-server exited (\(how))" : tail)
        loadedAt = nil
    }

    /// sd-server has no `/health`; the model list is the cheapest route that only
    /// answers once the checkpoint is resident.
    private nonisolated static func serverAnswers() async -> Bool {
        var request = URLRequest(url: baseURL.appending(path: "/sdapi/v1/sd-models"))
        request.httpMethod = "GET"
        request.timeoutInterval = 2
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    // MARK: - Port

    /// Returns `nil` once 127.0.0.1:8182 is bindable, or a human-readable reason it is not.
    private func claimPort() async -> String? {
        if Self.portIsFree() { return nil }

        // A child we just SIGTERMed may need a moment to let go of the socket.
        for _ in 0..<10 {
            try? await Task.sleep(for: .milliseconds(200))
            if Self.portIsFree() { return nil }
        }

        let holders = Self.portHolders()
        let ours = holders.filter(\.looksLikeOurServer)
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

    /// Bind-test the image port. `SO_REUSEADDR` keeps sockets lingering in TIME_WAIT
    /// from reading as "in use", so only a live listener returns false.
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

    /// Best-effort: who is listening on 8182. Only ever called on the failure path,
    /// where a short synchronous `lsof` is cheaper than a confusing error.
    private nonisolated static func portHolders() -> [ImagePortHolder] {
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

        var holders: [ImagePortHolder] = []
        var seen = Set<pid_t>()
        var pid: pid_t?
        for line in String(decoding: data, as: UTF8.self).split(separator: "\n") {
            if line.hasPrefix("p") {
                pid = pid_t(line.dropFirst())
            } else if line.hasPrefix("c"), let current = pid, !seen.contains(current) {
                seen.insert(current)
                holders.append(ImagePortHolder(pid: current, command: String(line.dropFirst())))
            }
        }
        return holders
    }

    // MARK: - Command line

    private nonisolated static func launchSpec(model: ImageModel) throws -> ImageLaunchSpec {
        let fm = FileManager.default

        guard model.kind == .diffusion else {
            throw ImageLaunchError("\(model.name) is an upscaler. Upscaling runs sd-cli directly and needs no server.")
        }
        guard !model.path.isEmpty, fm.fileExists(atPath: model.path) else {
            throw ImageLaunchError("\(model.name) is not on disk — nothing at \(model.path)")
        }

        let binary = ImagePaths.server
        guard fm.isExecutableFile(atPath: binary.path) else {
            throw ImageLaunchError("sd-server not found at \(binary.path). Point \"\(ImagePaths.Key.sdServer)\" at your stable-diffusion.cpp build in Settings.")
        }

        return ImageLaunchSpec(executable: binary,
                               arguments: try modelArguments(for: model, fm: fm)
                                           + ["--listen-ip", "127.0.0.1",
                                              "--listen-port", String(ImagePaths.port),
                                              // Flash attention in the diffusion model. Measured on this
                                              // M5 with SD 1.5 Q4_0, 512x512, 20 steps: 132.4s without,
                                              // 20.8s with — a 6.4x speedup for identical output. This is
                                              // the image-side equivalent of the LLM's full GPU offload,
                                              // and it is never worth turning off.
                                              "--diffusion-fa",
                                              "-v"])
    }

    /// How the checkpoint itself is passed, which differs by family.
    ///
    /// SD 1.5 and SDXL are self-contained: `-m <file>` and nothing else, which is all
    /// this did before. Newer families split the transformer, VAE and text encoders
    /// across separate files, so each one has to be located and named on the command
    /// line — see `ImageComponents`.
    private nonisolated static func modelArguments(for model: ImageModel,
                                                   fm: FileManager) throws -> [String] {
        let roles = model.arch.componentRoles
        guard !roles.isEmpty else { return ["-m", model.path] }

        let found = ImageComponents.resolve(roles: roles, modelPath: model.path, fm: fm)

        // Nothing at all beside it means this is almost certainly an all-in-one
        // checkpoint — several FLUX and SD 3.5 redistributions bake the encoders in.
        // `-m` is what loads those, and it is also exactly the old behaviour, so a
        // model that worked before still works.
        guard !found.isEmpty else { return ["-m", model.path] }

        // Some but not all: a split layout with a hole in it. sd-server's own failure
        // here is a tensor-shaped complaint that never names the missing file, so
        // name it here instead.
        let missing = roles.filter { found[$0] == nil }
        guard missing.isEmpty else {
            let wanted = missing.map(\.label).joined(separator: ", ")
            let searched = ImageComponents.searchDirectories(forModelAt: model.path, fm: fm)
                .map(\.path).joined(separator: "\n  ")
            throw ImageLaunchError("""
                \(model.name) is a \(model.arch.label) model, which sd.cpp loads from several files, \
                and \(missing.count == 1 ? "one is" : "\(missing.count) are") missing: \(wanted).
                Put \(missing.count == 1 ? "it" : "them") in one of these folders:
                  \(searched)
                """)
        }

        var args = [model.arch.usesStandaloneDiffusionFlag ? "--diffusion-model" : "-m", model.path]
        // Ordered by `roles` rather than dictionary order, so the log header is stable.
        for role in roles {
            guard let path = found[role] else { continue }
            args += [role.flag, path]
        }
        return args
    }

    private nonisolated static func describe(_ error: Error) -> String {
        if let launch = error as? ImageLaunchError { return launch.message }
        if let runner = error as? ProcessRunnerError { return runner.errorDescription ?? "\(runner)" }
        return (error as NSError).localizedDescription
    }

    // MARK: - Log

    /// Off-actor, from the pipe reader's queue.
    nonisolated private func enqueue(_ line: String) {
        pending.withLock { buffer in
            buffer.append(line)
            if buffer.count > Self.maxPending {
                buffer.removeFirst(buffer.count - Self.maxPending)
            }
        }
    }

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

    /// Move everything the child has written since the last pass into `logLines`.
    /// Batched rather than per-line so a verbose loader cannot drive 500 view updates
    /// a second.
    private func drainLog() {
        let lines: [String] = pending.withLock { buffer in
            guard !buffer.isEmpty else { return [] }
            let taken = buffer
            buffer.removeAll(keepingCapacity: true)
            return taken
        }
        guard !lines.isEmpty else { return }
        append(lines.map(Self.sanitize).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty })
    }

    private func append(_ lines: [String]) {
        guard !lines.isEmpty else { return }
        logLines.append(contentsOf: lines)
        if logLines.count > Self.maxLogLines {
            logLines.removeFirst(logLines.count - Self.maxLogLines)
        }
    }

    /// sd.cpp draws its tensor-loading progress bar with ANSI erase codes, which a
    /// SwiftUI `Text` renders as literal garbage ("…0.14MB/s[K"). Strip CSI sequences
    /// and leave everything else alone.
    private nonisolated static func sanitize(_ line: String) -> String {
        guard line.contains("\u{1B}") else { return line }

        enum Scan { case text, escape, csi }
        var scan = Scan.text
        var out = ""
        out.reserveCapacity(line.count)

        for character in line {
            switch scan {
            case .text:
                if character == "\u{1B}" { scan = .escape } else { out.append(character) }
            case .escape:
                // "ESC [" opens a control sequence; any other ESC pair is two chars long.
                scan = character == "[" ? .csi : .text
            case .csi:
                // Parameters and intermediates run 0x20–0x3F; the first byte in
                // 0x40–0x7E ends the sequence.
                if let scalar = character.unicodeScalars.first, (0x40...0x7E).contains(scalar.value) {
                    scan = .text
                }
            }
        }
        return out
    }
}
