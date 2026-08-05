import AVFoundation
import Darwin
import Foundation
import Observation
import Synchronization

// MARK: - Paths

/// Where the Kokoro worker and its interpreter live, and where its audio lands.
///
/// Separate from `Paths` and `ImagePaths` for the same reason those are separate
/// from each other: speech is optional, and a user who never presses Speak should
/// never be asked where a Python script is. Every entry is repointable through
/// `UserDefaults` (see `SpeechPaths.Key`).
enum SpeechPaths {
    /// UserDefaults keys a Settings screen can bind to with `@AppStorage`.
    enum Key {
        static let ttsWorker = "paths.ttsWorker"
        static let ttsPython = "paths.ttsPython"
        static let ttsScratch = "paths.ttsScratch"
    }


    /// Home-relative: see ImagePaths. Override in Settings > Paths.
    private static func inHome(_ relative: String) -> String { NSHomeDirectory() + "/" + relative }

    static var defaultWorkerScript: String { inHome("prism-llama/kokoro/tts_worker.py") }

    static var workerScript: URL {
        override(Key.ttsWorker) ?? URL(filePath: defaultWorkerScript)
    }

    /// The worker imports `mlx_audio` out of the same virtualenv that serves MLX
    /// chat models, so it follows `Paths.venvPython` rather than keeping a second
    /// copy of the path. The dedicated key covers the one case where that is wrong:
    /// a second venv that has mlx_audio when the first does not.
    static var python: URL {
        override(Key.ttsPython) ?? Paths.venvPython
    }

    /// The venv root, derived from the interpreter instead of stored twice.
    ///
    /// Not decoration: `mlx_audio` shells out to `uv`, which aborts with
    /// "No virtual environment found" unless `VIRTUAL_ENV` names this directory.
    /// Measured on this machine — the worker will not speak a word without it.
    static var virtualEnv: URL { virtualEnv(for: python) }

    /// `…/.venv/bin/python` → `…/.venv`. Pure, so `selfCheck()` can pin it down.
    static func virtualEnv(for python: URL) -> URL {
        python.deletingLastPathComponent().deletingLastPathComponent()
    }

    /// One WAV per utterance, deleted the moment it stops playing.
    ///
    /// The temp directory, not Application Support: nothing here is worth surviving
    /// a reboot, and a crash mid-sentence should not leave audio in the user's
    /// library folder for the OS to never clean up.
    static var audioScratch: URL {
        let dir = override(Key.ttsScratch)
            ?? FileManager.default.temporaryDirectory
                .appending(path: "SolidChatSpeech", directoryHint: .isDirectory)
        let fm = FileManager.default
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    private static func override(_ key: String) -> URL? {
        guard let raw = UserDefaults.standard.string(forKey: key) else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return URL(filePath: (trimmed as NSString).expandingTildeInPath)
    }
}

// MARK: - Errors

/// Everything this file throws. One type, because every failure ends up in the
/// same place: `SpeechState.failed`, rendered as the tooltip on a speak button.
private struct SpeechFailure: Error, LocalizedError {
    var message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

// MARK: - Queue

/// The pending utterance chunks. A value type with no behaviour beyond ordering,
/// so `selfCheck()` can prove the ordering without a worker, a voice or audio.
private struct SpeechUtteranceQueue {
    private var items: [String] = []

    var isEmpty: Bool { items.isEmpty }
    var count: Int { items.count }

    /// `speak()` semantics: whatever was queued is abandoned.
    mutating func replace(with chunks: [String]) { items = chunks }

    /// `enqueue()` semantics: appended behind what is already waiting.
    mutating func append(_ chunks: [String]) { items.append(contentsOf: chunks) }

    mutating func next() -> String? {
        items.isEmpty ? nil : items.removeFirst()
    }

    mutating func removeAll() { items.removeAll() }
}

// MARK: - What a run was started with

/// The engine, voice and knob positions captured when an utterance began.
///
/// Snapshotted rather than read live: the settings sliders are bound straight to
/// `SpeechEngine.settings`, and a drag mid-sentence would otherwise change the
/// voice between two chunks of the same reply.
private struct SpeechPlan: Sendable {
    var engine: SpeechEngineKind
    var voiceID: String
    var rate: Double
    var pitch: Double
    var volume: Double
    var kokoroSpeed: Double
}

// MARK: - Kokoro wire protocol

/// One request line. `Encodable` rather than hand-built JSON so a quote or a
/// newline inside a chat reply cannot break the line framing the worker reads.
private struct KokoroRequest: Encodable, Sendable {
    var text: String
    var voice: String
    var speed: Double
    var out: String

    /// The exact bytes written to the worker's stdin, newline included.
    static func line(for request: KokoroRequest) -> String? {
        let encoder = JSONEncoder()
        // Sorted for reproducibility (selfCheck compares against it); slashes left
        // alone so a path in the log reads like a path.
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(request),
              let text = String(data: data, encoding: .utf8) else { return nil }
        return text + "\n"
    }
}

/// A line the worker wrote back.
private enum KokoroMessage: Equatable {
    case ready(voices: [String], sampleRate: Int)
    case ok(path: String, seconds: Double)
    case failure(String)

    /// `nil` for anything that is not one of the three shapes above.
    ///
    /// stdout is supposed to be pure protocol — the worker redirects library chatter
    /// to stderr — but treating a stray print as the answer would desynchronise every
    /// later reply by one, so unknown lines are skipped rather than trusted.
    static func parse(_ line: String) -> KokoroMessage? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("{"), let data = trimmed.data(using: .utf8) else { return nil }
        guard let any = try? JSONSerialization.jsonObject(with: data),
              let object = any as? [String: Any] else { return nil }

        if object["ready"] as? Bool == true {
            let voices = object["voices"] as? [String] ?? []
            let rate = (object["sampleRate"] as? NSNumber)?.intValue ?? 24_000
            return .ready(voices: voices, sampleRate: rate)
        }
        if let ok = object["ok"] as? Bool {
            guard ok else {
                return .failure(object["error"] as? String ?? "the Kokoro worker rejected the request")
            }
            let path = object["path"] as? String ?? ""
            let seconds = (object["seconds"] as? NSNumber)?.doubleValue ?? 0
            return .ok(path: path, seconds: seconds)
        }
        return nil
    }
}

// MARK: - Crash-safe child registry

/// Process groups of live speech workers, so `atexit` takes them down with the app.
/// `ProcessRunner` keeps its own private registry for its own children; this is the
/// same idea, not a shared one, because that storage is file-private over there.
private let liveSpeechChildren = Mutex<Set<pid_t>>([])

private let speechReaperInstalled: Bool = {
    _ = atexit {
        let groups = liveSpeechChildren.withLock { $0 }
        guard !groups.isEmpty else { return }
        for pgid in groups where pgid > 1 { kill(-pgid, SIGTERM) }
        usleep(200_000)
        for pgid in groups where pgid > 1 { kill(-pgid, SIGKILL) }
    }
    return true
}()

// MARK: - Kokoro worker

private struct SpeechChild {
    var pid: pid_t
    var stdinFD: Int32
    var stdoutFD: Int32
    var stderrFD: Int32
}

/// The persistent Kokoro child: one JSON line in, one JSON line out, forever.
///
/// Not built on `ProcessRunner`. That type opens `/dev/null` on the child's stdin and
/// merges stdout with stderr — correct for a server you only ever read from, fatal for
/// a worker you have to *talk* to and whose stdout is a framed protocol.
///
/// Every request is funnelled through one serial queue. The worker answers exactly one
/// line per request with no id in it, so two overlapping writes would produce two
/// replies with no way to say which belonged to which.
private final class KokoroWorker: Sendable {

    private struct Store {
        var pid: pid_t
        var stdinFD: Int32
        var stdoutFD: Int32
        var stderrFD: Int32
        /// Bytes read from stdout that do not yet make a whole line.
        var buffer: [UInt8] = []
        /// Last few stderr lines. The whole diagnosis when the worker dies during
        /// import — a missing mlx_audio or the `uv` complaint about VIRTUAL_ENV.
        var stderrTail: [String] = []
        var voices: [String] = []
        var sampleRate: Int = 24_000
        var exited = false
        var shuttingDown = false
    }

    private let store: Mutex<Store>
    /// Serialises request/response. Blocking `poll`/`read` are fine here and only here.
    private let io: DispatchQueue

    nonisolated static let stderrTailLines = 24
    /// A cold start also pays for `snapshot_download` if the model is not cached yet.
    nonisolated static let readyTimeout: TimeInterval = 300
    /// Measured: ~1.8 s for a 350-character chunk on this M5. Anything past this is wedged.
    nonisolated static let replyTimeout: TimeInterval = 120

    var voices: [String] { store.withLock { $0.voices } }
    var sampleRate: Int { store.withLock { $0.sampleRate } }
    var isAlive: Bool { store.withLock { !$0.exited && !$0.shuttingDown } }

    private init(child: SpeechChild, io: DispatchQueue) {
        self.io = io
        store = Mutex(Store(pid: child.pid,
                            stdinFD: child.stdinFD,
                            stdoutFD: child.stdoutFD,
                            stderrFD: child.stderrFD))
        startStderrDrain(fd: child.stderrFD)
        startReaper(pid: child.pid)
    }

    deinit {
        // Last resort. A leaked python holds the whole Kokoro model in unified memory.
        let (pid, exited) = store.withLock { ($0.pid, $0.exited) }
        if !exited, pid > 1 {
            Self.signalGroup(SIGTERM, pid: pid)
            usleep(100_000)
            Self.signalGroup(SIGKILL, pid: pid)
            liveSpeechChildren.withLock { _ = $0.remove(pid) }
        }
        closeDescriptors()
    }

    // MARK: Start

    /// Spawn the worker and return once it has announced its voices.
    ///
    /// `nonisolated async`, so the ~4.4 s wait for the model to load happens off the
    /// main actor even though every caller is on it.
    static func start(python: URL, script: URL, virtualEnv: URL) async throws -> KokoroWorker {
        let child = try spawn(python: python, script: script, virtualEnv: virtualEnv)
        let worker = KokoroWorker(child: child,
                                  io: DispatchQueue(label: "com.solidchat.speech.kokoro",
                                                    qos: .userInitiated))
        do {
            try await worker.handshake()
        } catch {
            worker.shutdown()
            throw error
        }
        return worker
    }

    /// Read the single `{"ready": …}` line the worker emits before it will take work.
    private func handshake() async throws {
        let message = try await onIO { [self] in
            try readMessage(timeout: Self.readyTimeout, what: "start up")
        }
        guard case .ready(let voices, let sampleRate) = message else {
            throw SpeechFailure("the Kokoro worker answered something other than its ready line. \(stderrSummary())")
        }
        store.withLock {
            $0.voices = voices
            $0.sampleRate = sampleRate
        }
    }

    // MARK: Request

    /// Synthesise one chunk and return the path the worker actually wrote.
    func synthesise(_ request: KokoroRequest) async throws -> String {
        guard let line = KokoroRequest.line(for: request) else {
            throw SpeechFailure("could not encode the speech request.")
        }
        let message = try await onIO { [self] in
            try writeLine(line)
            return try readMessage(timeout: Self.replyTimeout, what: "answer")
        }
        switch message {
        case .ok(let path, _):
            return path.isEmpty ? request.out : path
        case .failure(let reason):
            throw SpeechFailure("Kokoro could not speak that — \(reason)")
        case .ready:
            // The worker restarted underneath us; the reply we were waiting for is gone.
            throw SpeechFailure("the Kokoro worker restarted mid-sentence.")
        }
    }

    /// Hop onto the serial queue and back. Every fd touch in this class goes through here.
    private func onIO<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            io.async {
                do { continuation.resume(returning: try work()) }
                catch { continuation.resume(throwing: error) }
            }
        }
    }

    /// Read lines until one parses, the worker dies, or `timeout` elapses.
    private func readMessage(timeout: TimeInterval, what: String) throws -> KokoroMessage {
        let deadline = Date.now.addingTimeInterval(timeout)
        while true {
            guard let line = try readLine(deadline: deadline, timeout: timeout, what: what) else {
                throw SpeechFailure("the Kokoro worker stopped. \(stderrSummary())")
            }
            if let message = KokoroMessage.parse(line) { return message }
        }
    }

    /// One line of stdout, or `nil` at EOF. Blocking, on the io queue only.
    ///
    /// `timeout` is carried alongside `deadline` only so the error can name the budget
    /// the worker blew; by the time it is thrown the deadline itself is in the past.
    private func readLine(deadline: Date, timeout: TimeInterval, what: String) throws -> String? {
        if let buffered = takeBufferedLine() { return buffered }

        let fd = store.withLock { $0.stdoutFD }
        guard fd >= 0 else { return nil }

        var chunk = [UInt8](repeating: 0, count: 8192)
        while true {
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else {
                throw SpeechFailure("the Kokoro worker did not \(what) within \(Int(timeout)) seconds. \(stderrSummary())")
            }
            // Capped slices rather than one long block, so a worker that dies without
            // closing its pipe (a surviving grandchild holds the write end) still gets
            // noticed at the deadline instead of hanging the queue forever.
            var descriptor = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
            let ready = poll(&descriptor, 1, Int32(min(remaining * 1000, 250)))
            if ready < 0 {
                if errno == EINTR { continue }
                throw SpeechFailure("lost the connection to the Kokoro worker (errno \(errno)).")
            }
            if ready == 0 { continue }

            let count = chunk.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress, $0.count) }
            if count < 0 {
                if errno == EINTR { continue }
                throw SpeechFailure("could not read from the Kokoro worker (errno \(errno)).")
            }
            if count == 0 {
                store.withLock {
                    if $0.stdoutFD >= 0 { close($0.stdoutFD); $0.stdoutFD = -1 }
                }
                return nil
            }
            store.withLock { $0.buffer.append(contentsOf: chunk[0..<count]) }
            if let line = takeBufferedLine() { return line }
        }
    }

    private func takeBufferedLine() -> String? {
        store.withLock { state in
            guard let index = state.buffer.firstIndex(of: 0x0A) else { return nil }
            let line = String(decoding: state.buffer[..<index], as: UTF8.self)
            state.buffer.removeSubrange(0...index)
            return line
        }
    }

    private func writeLine(_ line: String) throws {
        let fd = store.withLock { $0.stdinFD }
        guard fd >= 0 else { throw SpeechFailure("the Kokoro worker is no longer running.") }

        let bytes = Array(line.utf8)
        guard !bytes.isEmpty else { return }
        var offset = 0
        while offset < bytes.count {
            let written = bytes.withUnsafeBytes { raw -> Int in
                guard let base = raw.baseAddress else { return -1 }
                return Darwin.write(fd, base.advanced(by: offset), raw.count - offset)
            }
            if written > 0 { offset += written; continue }
            if written < 0, errno == EINTR { continue }
            // EPIPE, not a crash: the fd carries F_SETNOSIGPIPE so a dead worker
            // cannot take the whole app down with SIGPIPE.
            throw SpeechFailure("could not reach the Kokoro worker (errno \(errno)). \(stderrSummary())")
        }
    }

    // MARK: Diagnostics

    /// The worker's dying words, ready to append to an error message.
    private func stderrSummary() -> String {
        let tail = store.withLock { $0.stderrTail }
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard let last = tail.last else { return "It printed nothing to explain why." }
        return "Its last output was: \(last)"
    }

    private func startStderrDrain(fd: Int32) {
        let queue = DispatchQueue(label: "com.solidchat.speech.kokoro.stderr", qos: .utility)
        queue.async { [weak self] in
            var pending: [UInt8] = []
            var chunk = [UInt8](repeating: 0, count: 4096)
            while true {
                let count = chunk.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress, $0.count) }
                if count < 0, errno == EINTR { continue }
                if count <= 0 { break }
                guard let worker = self else { break }
                pending.append(contentsOf: chunk[0..<count])
                while let index = pending.firstIndex(of: 0x0A) {
                    worker.recordStderr(String(decoding: pending[..<index], as: UTF8.self))
                    pending.removeSubrange(0...index)
                }
                // A pathologically long unterminated line must not grow without bound.
                if pending.count > 16 * 1024 { pending.removeFirst(pending.count - 16 * 1024) }
            }
            close(fd)
            self?.store.withLock { if $0.stderrFD == fd { $0.stderrFD = -1 } }
        }
    }

    private func recordStderr(_ line: String) {
        store.withLock { state in
            state.stderrTail.append(line)
            if state.stderrTail.count > Self.stderrTailLines {
                state.stderrTail.removeFirst(state.stderrTail.count - Self.stderrTailLines)
            }
        }
    }

    // MARK: Exit

    private func startReaper(pid: pid_t) {
        let queue = DispatchQueue(label: "com.solidchat.speech.kokoro.reaper", qos: .utility)
        queue.async { [weak self] in
            var status: Int32 = 0
            var reaped: pid_t = -1
            repeat { reaped = waitpid(pid, &status, 0) } while reaped < 0 && errno == EINTR
            liveSpeechChildren.withLock { _ = $0.remove(pid) }
            self?.store.withLock { $0.exited = true }
        }
    }

    /// SIGTERM the worker's process group, then SIGKILL it. Bounded to ~600 ms so
    /// `applicationWillTerminate` can call it without making Quit feel broken.
    func shutdown() {
        let (pid, already) = store.withLock { state -> (pid_t, Bool) in
            let was = state.shuttingDown
            state.shuttingDown = true
            return (state.pid, was)
        }
        guard !already, pid > 1 else { return }

        // Deliberately no fd closing here: the io queue may be parked in poll() on
        // stdout right now, and pulling a descriptor out from under it invites a
        // read against a number the kernel has already handed to someone else.
        // The child dying closes its ends, which is what the reader is waiting for.
        Self.signalGroup(SIGTERM, pid: pid)
        for _ in 0..<12 {
            if store.withLock({ $0.exited }) { return }
            Thread.sleep(forTimeInterval: 0.05)
        }
        Self.signalGroup(SIGKILL, pid: pid)
    }

    private func closeDescriptors() {
        store.withLock { state in
            for fd in [state.stdinFD, state.stdoutFD, state.stderrFD] where fd >= 0 { close(fd) }
            state.stdinFD = -1
            state.stdoutFD = -1
            state.stderrFD = -1
        }
    }

    /// Spawned with `POSIX_SPAWN_SETPGROUP` and pgroup 0, so the child's pgid *is* its
    /// pid. The `getpgrp()` guard is the seatbelt: signalling our own group would take
    /// down the app.
    private static func signalGroup(_ sig: Int32, pid: pid_t) {
        if pid > 1, pid != getpgrp(), kill(-pid, sig) == 0 { return }
        _ = kill(pid, sig)
    }

    // MARK: Spawn

    private static func spawn(python: URL, script: URL, virtualEnv: URL) throws -> SpeechChild {
        let fm = FileManager.default
        guard fm.isExecutableFile(atPath: python.path) else {
            throw SpeechFailure("No Python interpreter at \(python.path). Set \"\(SpeechPaths.Key.ttsPython)\" to your venv's python, or switch the speech engine to System.")
        }
        guard fm.fileExists(atPath: script.path) else {
            throw SpeechFailure("The Kokoro worker script is missing — nothing at \(script.path). Set \"\(SpeechPaths.Key.ttsWorker)\" to your tts_worker.py, or switch the speech engine to System.")
        }

        var toChild: [Int32] = [-1, -1]     // child reads [0], we write [1]
        var fromChild: [Int32] = [-1, -1]   // child writes [1], we read [0]
        var errFromChild: [Int32] = [-1, -1]
        guard pipe(&toChild) == 0 else {
            throw SpeechFailure("could not open a pipe to the speech worker (errno \(errno)).")
        }
        guard pipe(&fromChild) == 0 else {
            close(toChild[0]); close(toChild[1])
            throw SpeechFailure("could not open a pipe from the speech worker (errno \(errno)).")
        }
        guard pipe(&errFromChild) == 0 else {
            close(toChild[0]); close(toChild[1]); close(fromChild[0]); close(fromChild[1])
            throw SpeechFailure("could not open a pipe from the speech worker (errno \(errno)).")
        }

        // None of these raw descriptors should survive the exec; the dup2s below
        // re-open 0, 1 and 2 for the child (dup2 always clears FD_CLOEXEC).
        for fd in [toChild[0], toChild[1], fromChild[0], fromChild[1], errFromChild[0], errFromChild[1]] {
            _ = fcntl(fd, F_SETFD, FD_CLOEXEC)
        }
        // Writing to a dead worker's stdin would otherwise raise SIGPIPE and kill the
        // whole app. EPIPE is something we can turn into a sentence the user can read.
        _ = fcntl(toChild[1], F_SETNOSIGPIPE, 1)

        var actions: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&actions)
        defer { posix_spawn_file_actions_destroy(&actions) }
        posix_spawn_file_actions_adddup2(&actions, toChild[0], 0)
        posix_spawn_file_actions_adddup2(&actions, fromChild[1], 1)
        posix_spawn_file_actions_adddup2(&actions, errFromChild[1], 2)

        var attributes: posix_spawnattr_t?
        posix_spawnattr_init(&attributes)
        defer { posix_spawnattr_destroy(&attributes) }
        // Own process group so the whole tree can be signalled: python spawns `uv`,
        // and `uv` spawns more.
        posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETPGROUP))
        posix_spawnattr_setpgroup(&attributes, 0)

        let environment = childEnvironment(virtualEnv: virtualEnv)
        let command: [String] = [python.path, script.path]
        var argv: [UnsafeMutablePointer<CChar>?] = command.map { strdup($0) }
        argv.append(nil)
        var envp: [UnsafeMutablePointer<CChar>?] = environment.map { strdup("\($0.key)=\($0.value)") }
        envp.append(nil)
        defer {
            for slot in argv { free(slot) }
            for slot in envp { free(slot) }
        }

        var pid: pid_t = 0
        let status = posix_spawn(&pid, python.path, &actions, &attributes, argv, envp)

        // The parent must drop the child's ends, or those pipes never see EOF.
        close(toChild[0])
        close(fromChild[1])
        close(errFromChild[1])

        guard status == 0 else {
            close(toChild[1]); close(fromChild[0]); close(errFromChild[0])
            throw SpeechFailure("could not launch \(python.path) (errno \(status)).")
        }

        _ = speechReaperInstalled
        liveSpeechChildren.withLock { _ = $0.insert(pid) }
        return SpeechChild(pid: pid,
                           stdinFD: toChild[1],
                           stdoutFD: fromChild[0],
                           stderrFD: errFromChild[0])
    }

    private static func childEnvironment(virtualEnv: URL) -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        // The one variable that is not optional. Verified, not guessed.
        environment["VIRTUAL_ENV"] = virtualEnv.path
        // The worker flushes after every line, but a library that prints on its way
        // out should not sit in a block buffer either.
        environment["PYTHONUNBUFFERED"] = "1"

        // A GUI-launched app inherits a bare PATH (/usr/bin:/bin:/usr/sbin:/sbin) and
        // mlx_audio shells out to `uv`, which lives in none of those. Appended rather
        // than prepended, so a PATH the user actually set always wins.
        let home = FileManager.default.homeDirectoryForCurrentUser
        let extras = [virtualEnv.appending(path: "bin").path,
                      "/opt/homebrew/bin",
                      "/usr/local/bin",
                      home.appending(path: ".local/bin").path,
                      home.appending(path: ".cargo/bin").path]
        let current = environment["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        let known = Set(current.split(separator: ":").map(String.init))
        environment["PATH"] = ([current] + extras.filter { !known.contains($0) }).joined(separator: ":")
        return environment
    }
}

// MARK: - AVSpeechSynthesizer delegate

/// Bridges `AVSpeechSynthesizer`'s callbacks — which arrive on AVFoundation's own
/// queue, not the main actor — back to the engine.
///
/// A separate object rather than a conformance on `SpeechEngine`: that class is
/// `@MainActor`, and these methods are not. The handlers are re-armed per utterance
/// so each one carries the generation it belongs to and a stale callback cannot
/// advance a queue that has already moved on.
private final class SystemSpeechDelegate: NSObject, AVSpeechSynthesizerDelegate {

    private struct Handlers {
        var onStart: (@Sendable () -> Void)?
        var onEnd: (@Sendable () -> Void)?
    }

    private let handlers = Mutex(Handlers())

    func arm(onStart: @escaping @Sendable () -> Void, onEnd: @escaping @Sendable () -> Void) {
        handlers.withLock { $0 = Handlers(onStart: onStart, onEnd: onEnd) }
    }

    func disarm() {
        handlers.withLock { $0 = Handlers() }
    }

    /// Takes the end handler and clears it in one step, so didFinish followed by a
    /// late didCancel cannot end the same chunk twice.
    private func takeEnd() -> (@Sendable () -> Void)? {
        handlers.withLock { state in
            defer { state = Handlers() }
            return state.onEnd
        }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        handlers.withLock { $0.onStart }?()
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        takeEnd()?()
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        takeEnd()?()
    }
}

// MARK: - SpeechEngine

/// The whole speech subsystem: two backends, the Kokoro worker's lifecycle, an
/// utterance queue and playback.
///
/// Both backends are main-actor bound on purpose. `AVSpeechSynthesizer` and
/// `AVAudioPlayer` are not `Sendable`, and neither is allowed to leave this actor;
/// the only things that cross are the worker (immutable behind a mutex), plain
/// strings and file paths.
@Observable
@MainActor
final class SpeechEngine {

    // MARK: Observable state

    var state: SpeechState = .idle
    /// Owned here so views can bind to it directly; `AppStore` watches and persists it.
    var settings = SpeechSettings()
    var systemVoices: [SpeechVoice] = []
    /// Empty until the worker has announced itself — the 54 names come from the model
    /// snapshot on disk, and nothing else knows them.
    var kokoroVoices: [SpeechVoice] = []
    /// Which chat message is being read, so one row shows a stop button and the rest
    /// do not.
    var speakingMessageID: UUID?
    var workerReady = false
    /// The last failure, kept after `state` goes back to idle so Settings can show it.
    var lastError: String?

    // MARK: Constants

    nonisolated static let previewLine =
        "This is how I sound reading a sentence of a reply out loud."
    /// Kokoro renders a chunk before a single sample plays, so the first one decides
    /// how long the spinner is on screen.
    nonisolated private static let leadChunkThreshold = 200
    /// How long to wait for AVSpeechSynthesizer to actually begin before deciding it
    /// silently refused the utterance.
    nonisolated private static let systemStartGrace: TimeInterval = 2

    // MARK: Private state

    @ObservationIgnored private let synthesizer = AVSpeechSynthesizer()
    @ObservationIgnored private let systemDelegate = SystemSpeechDelegate()
    @ObservationIgnored private var player: AVAudioPlayer?

    @ObservationIgnored private var queue = SpeechUtteranceQueue()
    /// The plan the running utterance was started with, so `enqueue()` extends it on
    /// the same terms. Nil whenever nothing is running.
    @ObservationIgnored private var activePlan: SpeechPlan?
    /// Bumped by speak/stop/shutdown. Anything holding an older value must not touch
    /// state, start playback or delete a file — the same guard `ImageEngine` uses for
    /// a superseded load.
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private var driveTask: Task<Void, Never>?
    /// Set by `SystemSpeechDelegate` when the current utterance ends.
    @ObservationIgnored private var systemChunkEnded = false

    @ObservationIgnored private var worker: KokoroWorker?
    /// In-flight worker start, shared by every caller that arrives during the ~4.4 s.
    @ObservationIgnored private var startTask: Task<KokoroWorker, Error>?
    @ObservationIgnored private var didBootstrap = false

    init() {
        synthesizer.delegate = systemDelegate
    }

    // MARK: - Lifecycle

    /// Load the installed system voices and clear the scratch directory.
    ///
    /// Deliberately does not start the Kokoro worker: it costs ~4.4 s and holds the
    /// model in memory, and most launches never speak a word.
    func bootstrap() {
        guard !didBootstrap else { return }
        didBootstrap = true

        systemVoices = Self.installedSystemVoices()
        // A saved id survives the voice being uninstalled in System Settings, and a
        // picker whose selection does not exist renders blank. Falling back to "" puts
        // it on "System Default", which is true rather than merely quiet.
        if !settings.systemVoiceID.isEmpty,
           !systemVoices.contains(where: { $0.id == settings.systemVoiceID }) {
            settings.systemVoiceID = ""
        }
        Self.purgeScratch()
    }

    /// Start the Kokoro worker now so the first reply does not wait for it.
    func warmUpKokoro() {
        guard worker?.isAlive != true else { return }
        // Never over a live utterance: warming up is background work and must not
        // repaint a speak button that is mid-sentence. The generation snapshot is what
        // decides that at the *end* too — four seconds is long enough for the user to
        // have started, and finished, speaking something in the meantime.
        let announce = !state.isActive
        let generationAtStart = generation
        if announce { state = .preparing }
        Task { [weak self] in
            guard let self else { return }
            do {
                _ = try await self.ensureWorker()
                if announce, generationAtStart == self.generation, self.state == .preparing {
                    self.state = .idle
                }
            } catch {
                let message = Self.describe(error)
                self.lastError = message
                if announce, generationAtStart == self.generation {
                    self.state = .failed(message)
                }
            }
        }
    }

    /// Bounded, blocking teardown for `applicationWillTerminate`.
    func shutdownBlocking() {
        generation &+= 1
        driveTask?.cancel()
        driveTask = nil
        startTask?.cancel()
        startTask = nil
        activePlan = nil
        queue.removeAll()

        synthesizer.stopSpeaking(at: .immediate)
        systemDelegate.disarm()
        player?.stop()
        player = nil

        let running = worker
        worker = nil
        workerReady = false
        running?.shutdown()

        speakingMessageID = nil
        state = .idle
        Self.purgeScratch()
    }

    // MARK: - Speaking

    var isSpeaking: Bool { state.isActive }

    /// Replace whatever is queued and read `text` aloud.
    func speak(_ text: String, messageID: UUID?) {
        let plan = currentPlan()
        let chunks = Self.chunks(for: text, engine: plan.engine)
        // An empty or markdown-only message must not silence something that is
        // already speaking — hence the guard before the stop.
        guard !chunks.isEmpty else { return }

        stop()
        lastError = nil
        queue.replace(with: chunks)
        speakingMessageID = messageID
        start(plan: plan)
    }

    /// Append to the running queue. For a reply that is still streaming in.
    func enqueue(_ text: String, messageID: UUID?) {
        // A different message means the caller moved on; splicing two replies into one
        // utterance would read the tail of the old one after the new one started.
        guard driveTask != nil, let plan = activePlan, messageID == speakingMessageID else {
            speak(text, messageID: messageID)
            return
        }
        // Chunked against the *running* plan, not the current settings: a Kokoro run
        // splits its opening sentence out, and switching engines mid-reply would
        // otherwise change how the rest of it is cut up.
        let chunks = Self.chunks(for: text, engine: plan.engine)
        guard !chunks.isEmpty else { return }
        queue.append(chunks)
    }

    /// Speak, or stop if this message is the one already being spoken.
    func toggle(_ text: String, messageID: UUID?) {
        if speakingMessageID == messageID, state.isActive {
            stop()
        } else {
            speak(text, messageID: messageID)
        }
    }

    /// Cancel everything: the queue, the synthesizer, playback and any in-flight
    /// synthesis. Safe to call when nothing is speaking.
    func stop() {
        generation &+= 1
        queue.removeAll()
        driveTask?.cancel()
        driveTask = nil
        activePlan = nil

        synthesizer.stopSpeaking(at: .immediate)
        systemDelegate.disarm()
        systemChunkEnded = true

        player?.stop()
        player = nil

        speakingMessageID = nil
        state = .idle
    }

    /// Speak a fixed sample line in `voice`, whichever engine it belongs to.
    func previewVoice(_ voice: SpeechVoice) {
        var plan = currentPlan()
        plan.engine = voice.engine
        plan.voiceID = voice.id

        let chunks = Self.chunks(for: Self.previewLine, engine: plan.engine)
        guard !chunks.isEmpty else { return }

        stop()
        lastError = nil
        queue.replace(with: chunks)
        speakingMessageID = nil
        start(plan: plan)
    }

    func voices(for engine: SpeechEngineKind) -> [SpeechVoice] {
        engine == .kokoro ? kokoroVoices : systemVoices
    }

    // MARK: - Drive

    private func start(plan: SpeechPlan) {
        generation &+= 1
        let generationAtStart = generation
        activePlan = plan
        state = .preparing
        driveTask = Task { [weak self] in
            guard let self else { return }
            switch plan.engine {
            case .system: await self.runSystem(gen: generationAtStart, plan: plan)
            case .kokoro: await self.runKokoro(gen: generationAtStart, plan: plan)
            }
        }
    }

    private func finish(gen: Int) {
        guard gen == generation else { return }
        driveTask = nil
        activePlan = nil
        queue.removeAll()
        speakingMessageID = nil
        state = .idle
    }

    /// End the run with a message the user can act on.
    ///
    /// `speakingMessageID` is left alone on purpose: the row that asked for speech is
    /// the one that should explain why it got none. A `.failed` state is not active,
    /// so the button reverts to "speak" with the reason in its tooltip.
    private func fail(_ message: String, gen: Int) {
        guard gen == generation else { return }
        driveTask = nil
        activePlan = nil
        queue.removeAll()
        player?.stop()
        player = nil
        lastError = message
        state = .failed(message)
    }

    private func currentPlan() -> SpeechPlan {
        SpeechPlan(engine: settings.engine,
                   voiceID: settings.voiceID(for: settings.engine),
                   rate: settings.rate,
                   pitch: settings.pitch,
                   volume: settings.volume,
                   kokoroSpeed: settings.kokoroSpeed)
    }

    // MARK: - System backend

    private func runSystem(gen: Int, plan: SpeechPlan) async {
        while gen == generation, let text = queue.next() {
            await speakSystemChunk(text, plan: plan, gen: gen)
        }
        finish(gen: gen)
    }

    /// Speak one chunk and return when it has finished, been cancelled, or turned out
    /// never to have started.
    private func speakSystemChunk(_ text: String, plan: SpeechPlan, gen: Int) async {
        let utterance = AVSpeechUtterance(string: text)
        if !plan.voiceID.isEmpty, let voice = AVSpeechSynthesisVoice(identifier: plan.voiceID) {
            utterance.voice = voice
        }
        utterance.rate = Float(Self.clamp(plan.rate, 0, 1))
        utterance.pitchMultiplier = Float(Self.clamp(plan.pitch, 0.5, 2))
        utterance.volume = Float(Self.clamp(plan.volume, 0, 1))

        systemChunkEnded = false
        systemDelegate.arm(
            onStart: { [weak self] in
                Task { @MainActor in self?.systemDidStart(gen: gen) }
            },
            onEnd: { [weak self] in
                Task { @MainActor in self?.systemDidEnd(gen: gen) }
            })

        synthesizer.speak(utterance)

        // Delegate-driven, with `isSpeaking` as the backstop. AVSpeechSynthesizer can
        // quietly decline an utterance — an unavailable voice, a string it decides is
        // empty — and then never call back at all, and a queue waiting forever on a
        // callback that is not coming is worse than a dropped chunk.
        var didBegin = false
        let grace = Date.now.addingTimeInterval(Self.systemStartGrace)
        while gen == generation, !systemChunkEnded {
            if synthesizer.isSpeaking {
                didBegin = true
            } else if didBegin {
                break                       // finished, but the callback never arrived
            } else if Date.now > grace {
                break                       // never started
            }
            try? await Task.sleep(for: .milliseconds(20))
        }
        systemDelegate.disarm()
    }

    private func systemDidStart(gen: Int) {
        guard gen == generation else { return }
        state = .speaking
    }

    private func systemDidEnd(gen: Int) {
        guard gen == generation else { return }
        systemChunkEnded = true
    }

    // MARK: - Kokoro backend

    private func runKokoro(gen: Int, plan: SpeechPlan) async {
        let active: KokoroWorker
        do {
            active = try await ensureWorker()
        } catch {
            // Deliberately no fallback to the system voice: a silent switch to a
            // Compact voice is a quality change the user did not ask for and cannot
            // explain.
            fail(Self.describe(error), gen: gen)
            return
        }
        guard gen == generation else { return }

        var current = synthesise(queue.next(), on: active, plan: plan, gen: gen)
        var upcoming: Task<URL, Error>?

        while let job = current {
            guard gen == generation else { break }

            let url: URL
            do {
                url = try await job.value
            } catch is CancellationError {
                break
            } catch {
                upcoming?.cancel()
                fail(Self.describe(error), gen: gen)
                return
            }
            guard gen == generation else { Self.remove(url); break }

            do {
                try startPlayback(url, volume: plan.volume)
            } catch {
                Self.remove(url)
                upcoming?.cancel()
                fail(Self.describe(error), gen: gen)
                return
            }
            state = .speaking

            // Render the next chunk while this one plays. The worker takes one request
            // at a time, so this overlap is the whole difference between continuous
            // speech and a ~1.8 s hole between every sentence.
            if upcoming == nil {
                upcoming = synthesise(queue.next(), on: active, plan: plan, gen: gen)
            }

            await awaitPlayback(gen: gen)
            player?.stop()
            player = nil
            Self.remove(url)
            guard gen == generation else { break }

            current = upcoming
            upcoming = nil
            // enqueue() may have appended while that chunk was playing.
            if current == nil {
                current = synthesise(queue.next(), on: active, plan: plan, gen: gen)
            }
        }

        upcoming?.cancel()
        finish(gen: gen)
    }

    /// Ask the worker for one chunk's audio. `nil` when there is nothing left to say.
    private func synthesise(_ text: String?,
                            on worker: KokoroWorker,
                            plan: SpeechPlan,
                            gen: Int) -> Task<URL, Error>? {
        guard let text else { return nil }
        let voice = plan.voiceID.isEmpty ? SpeechSettings().kokoroVoiceID : plan.voiceID
        let destination = SpeechPaths.audioScratch
            .appending(path: "utterance-\(UUID().uuidString).wav", directoryHint: .notDirectory)
        let request = KokoroRequest(text: text,
                                    voice: voice,
                                    speed: Self.clamp(plan.kokoroSpeed, 0.5, 2),
                                    out: destination.path)

        return Task { [weak self] in
            let produced = try await worker.synthesise(request)
            let url = URL(filePath: produced)
            // The user may have pressed stop while this sat inside the worker.
            // Nothing will ever play it, so it is cleaned up here rather than left
            // behind in the scratch directory.
            guard let self, self.generation == gen else {
                SpeechEngine.remove(url)
                throw CancellationError()
            }
            return url
        }
    }

    /// Start or restart the worker. Concurrent callers share one start.
    private func ensureWorker() async throws -> KokoroWorker {
        if let existing = worker, existing.isAlive { return existing }
        // A worker that died mid-session leaves the app permanently mute otherwise.
        // The utterance that hit the death still fails; the next one gets a fresh child.
        worker = nil
        workerReady = false

        if let pending = startTask { return try await pending.value }

        let python = SpeechPaths.python
        let script = SpeechPaths.workerScript
        let virtualEnv = SpeechPaths.virtualEnv
        let task = Task<KokoroWorker, Error> {
            try await KokoroWorker.start(python: python, script: script, virtualEnv: virtualEnv)
        }
        startTask = task

        do {
            let started = try await task.value
            startTask = nil
            worker = started
            kokoroVoices = started.voices.map {
                SpeechVoice(id: $0, name: $0, language: Self.kokoroLanguage($0),
                            engine: .kokoro, quality: .premium)
            }
            workerReady = true
            return started
        } catch {
            startTask = nil
            workerReady = false
            throw error
        }
    }

    // MARK: - Playback

    private func startPlayback(_ url: URL, volume: Double) throws {
        let created = try AVAudioPlayer(contentsOf: url)
        created.volume = Float(Self.clamp(volume, 0, 1))
        created.prepareToPlay()
        guard created.play() else {
            throw SpeechFailure("macOS would not play the synthesised audio.")
        }
        player = created
    }

    /// Return once the current clip has finished, or the run has been superseded.
    private func awaitPlayback(gen: Int) async {
        guard let playing = player else { return }
        // One long sleep for the body of the clip, then a fine poll for the tail:
        // `duration` is exact, but ending on the estimate alone clips the last word.
        let remaining = playing.duration - playing.currentTime
        if remaining > 0.1 {
            try? await Task.sleep(for: .seconds(remaining - 0.05))
        }
        while gen == generation, player?.isPlaying == true {
            try? await Task.sleep(for: .milliseconds(20))
        }
    }

    // MARK: - Voices

    private nonisolated static func installedSystemVoices() -> [SpeechVoice] {
        AVSpeechSynthesisVoice.speechVoices().map { voice in
            SpeechVoice(id: voice.identifier,
                        name: voice.name,
                        language: voice.language,
                        engine: .system,
                        quality: quality(of: voice.quality))
        }
    }

    /// AVFoundation calls the small robotic voices `.default`; this app calls them
    /// Compact, because that is the word System Settings puts next to them.
    private nonisolated static func quality(of value: AVSpeechSynthesisVoiceQuality) -> VoiceQuality {
        switch value {
        case .premium: return .premium
        case .enhanced: return .enhanced
        default: return .compact
        }
    }

    /// Kokoro encodes the locale in the first letter of the voice name. Settings groups
    /// its picker on `language.hasPrefix("en")`, so a/b must come back as English.
    nonisolated static func kokoroLanguage(_ name: String) -> String {
        switch name.first {
        case "a": return "en-US"
        case "b": return "en-GB"
        case "e": return "es"
        case "f": return "fr"
        case "h": return "hi"
        case "i": return "it"
        case "j": return "ja"
        case "p": return "pt"
        case "z": return "zh"
        default: return ""
        }
    }

    // MARK: - Chunking

    private nonisolated static func chunks(for text: String, engine: SpeechEngineKind) -> [String] {
        let all = SpeechChunker.chunks(text)
        return engine == .kokoro ? splitLead(all) : all
    }

    /// Peel the opening sentence off a long first chunk.
    ///
    /// Kokoro renders a whole chunk before one sample plays — ~1.8 s for 350
    /// characters — so the first chunk is the entire perceived latency. Splitting on a
    /// sentence boundary (never mid-sentence, which would put a pause in the middle of
    /// a clause) roughly halves it, and the full-size chunk behind it finishes
    /// rendering long before the opener stops playing.
    private nonisolated static func splitLead(_ chunks: [String]) -> [String] {
        guard let head = chunks.first, head.count > leadChunkThreshold else { return chunks }
        let parts = SpeechChunker.sentences(head)
        guard parts.count > 1, let opener = parts.first, opener.count >= 20 else { return chunks }
        let rest = parts.dropFirst().joined(separator: " ")
        return [opener, rest] + chunks.dropFirst()
    }

    // MARK: - Files

    private nonisolated static func remove(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    /// Clear the scratch directory. Called at bootstrap and at quit — a crash
    /// mid-sentence is the only way a WAV outlives its playback.
    private nonisolated static func purgeScratch() {
        let fm = FileManager.default
        let dir = SpeechPaths.audioScratch
        guard let names = try? fm.contentsOfDirectory(atPath: dir.path(percentEncoded: false)) else { return }
        for name in names where name.hasSuffix(".wav") {
            try? fm.removeItem(at: dir.appending(path: name, directoryHint: .notDirectory))
        }
    }

    // MARK: - Helpers

    private nonisolated static func clamp(_ value: Double, _ low: Double, _ high: Double) -> Double {
        guard value.isFinite else { return low }
        return Swift.min(Swift.max(value, low), high)
    }

    private nonisolated static func describe(_ error: Error) -> String {
        if let failure = error as? SpeechFailure { return failure.message }
        if error is CancellationError { return "Speech was cancelled." }
        return (error as NSError).localizedDescription
    }

    // MARK: - Self check

    /// Pure logic only: queue ordering, request framing, reply parsing, the voice and
    /// path derivations. Returns [] when everything holds.
    nonisolated static func selfCheck() -> [String] {
        var failures: [String] = []

        // Queue ordering — speak() replaces, enqueue() appends behind.
        var queue = SpeechUtteranceQueue()
        queue.replace(with: ["one", "two"])
        queue.append(["three"])
        if queue.next() != "one" { failures.append("queue: first in should come out first") }
        queue.append(["four"])
        if queue.next() != "two" { failures.append("queue: a late append jumped the line") }
        if queue.next() != "three" || queue.next() != "four" {
            failures.append("queue: appended chunks lost their order")
        }
        if queue.next() != nil { failures.append("queue: a drained queue should hand back nothing") }
        queue.replace(with: ["a", "b"])
        queue.replace(with: ["c"])
        if queue.next() != "c" || queue.next() != nil {
            failures.append("queue: replace must discard what was already queued")
        }

        // Request framing — one line, and text that would break the framing escaped.
        let request = KokoroRequest(text: "Line one.\nLine \"two\".",
                                    voice: "af_heart",
                                    speed: 1.25,
                                    out: "/tmp/a b.wav")
        if let line = KokoroRequest.line(for: request) {
            if !line.hasSuffix("\n") { failures.append("request: must end with a newline") }
            if line.filter({ $0 == "\n" }).count != 1 {
                failures.append("request: a raw newline in the text would desynchronise the worker")
            }
            let body = String(line.dropLast())
            if let data = body.data(using: .utf8),
               let any = try? JSONSerialization.jsonObject(with: data),
               let object = any as? [String: Any] {
                if object["text"] as? String != request.text { failures.append("request: text did not survive encoding") }
                if object["voice"] as? String != request.voice { failures.append("request: voice did not survive encoding") }
                if object["out"] as? String != request.out { failures.append("request: out did not survive encoding") }
                if (object["speed"] as? NSNumber)?.doubleValue != request.speed {
                    failures.append("request: speed did not survive encoding")
                }
            } else {
                failures.append("request: did not encode to a JSON object")
            }
        } else {
            failures.append("request: could not be encoded at all")
        }

        // Reply parsing.
        if KokoroMessage.parse(#"{"ready": true, "voices": ["af_heart"], "sampleRate": 24000}"#)
            != .ready(voices: ["af_heart"], sampleRate: 24_000) {
            failures.append("reply: the ready line did not parse")
        }
        if KokoroMessage.parse(#"{"ok": true, "path": "/tmp/x.wav", "seconds": 2.45}"#)
            != .ok(path: "/tmp/x.wav", seconds: 2.45) {
            failures.append("reply: a success line did not parse")
        }
        if KokoroMessage.parse(#"{"ok": false, "error": "boom"}"#) != .failure("boom") {
            failures.append("reply: a failure line did not parse")
        }
        if KokoroMessage.parse("Fetching 6 files: 100%|####| 6/6") != nil {
            failures.append("reply: a stray library print must be skipped, not parsed")
        }
        if KokoroMessage.parse("") != nil { failures.append("reply: an empty line must be skipped") }
        if KokoroMessage.parse("{not json}") != nil { failures.append("reply: broken JSON must be skipped") }

        // Voice locales — Settings groups its picker on hasPrefix("en").
        if !kokoroLanguage("af_heart").hasPrefix("en") { failures.append("voices: af_ must read as English") }
        if !kokoroLanguage("bm_george").hasPrefix("en") { failures.append("voices: bm_ must read as English") }
        if kokoroLanguage("jf_alpha") != "ja" { failures.append("voices: jf_ must read as Japanese") }
        if kokoroLanguage("zm_yunjian") != "zh" { failures.append("voices: zm_ must read as Chinese") }

        // Clamping — the sliders allow values the AV APIs do not.
        if clamp(0.5, 0, 1) != 0.5 { failures.append("clamp: an in-range value must pass through") }
        if clamp(-3, 0, 1) != 0 { failures.append("clamp: below range must land on the floor") }
        if clamp(9, 0.5, 2) != 2 { failures.append("clamp: above range must land on the ceiling") }
        if clamp(.nan, 0.5, 2) != 0.5 { failures.append("clamp: NaN must not reach AVFoundation") }

        // VIRTUAL_ENV derivation — the worker will not run without exactly this.
        let venv = SpeechPaths.virtualEnv(for: URL(filePath: "/Users/x/repo/.venv/bin/python"))
        if venv.path != "/Users/x/repo/.venv" {
            failures.append("paths: VIRTUAL_ENV came out as \(venv.path)")
        }

        // Lead-splitting must never cut inside a sentence.
        let long = String(repeating: "This is a full sentence of about forty characters. ", count: 6)
        let lead = splitLead(SpeechChunker.chunks(long))
        if let first = lead.first, first.count > leadChunkThreshold {
            failures.append("chunks: the opening chunk was not shortened")
        }
        if lead.contains(where: { $0.trimmingCharacters(in: .whitespaces).isEmpty }) {
            failures.append("chunks: lead-splitting produced an empty chunk")
        }
        let single = ["One very long unbroken clause that simply keeps going and going without ever reaching a terminator so there is nothing at all to split it on anywhere in here at all"]
        if splitLead(single) != single {
            failures.append("chunks: a single sentence must never be cut in half")
        }

        return failures
    }
}
