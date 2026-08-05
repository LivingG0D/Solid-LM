import Foundation
import Synchronization
import Darwin

// MARK: - Errors

enum ProcessRunnerError: Error, LocalizedError, Sendable {
    case alreadyRunning
    case executableMissing(String)
    case pipeFailed(Int32)
    case spawnFailed(Int32, String)

    var errorDescription: String? {
        switch self {
        case .alreadyRunning:
            return "That process is already running."
        case .executableMissing(let path):
            return "Not an executable file: \(path)"
        case .pipeFailed(let code):
            return "Could not create an output pipe — \(posixMessage(code))"
        case .spawnFailed(let code, let path):
            return "Could not launch \(path) — \(posixMessage(code))"
        }
    }
}

private func posixMessage(_ code: Int32) -> String {
    guard let text = strerror(code) else { return "errno \(code)" }
    return "\(String(cString: text)) (errno \(code))"
}

// MARK: - Crash-safe child registry

/// Process groups of every live child, so `atexit` can take them down with the app.
/// Children are spawned into their *own* process group, so `kill(-pgid, …)` can never
/// reach back into this app.
private let liveChildGroups = Mutex<Set<pid_t>>([])

/// Installed exactly once, lazily, by the first successful spawn.
private let atexitReaperInstalled: Bool = {
    _ = atexit {
        let groups = liveChildGroups.withLock { $0 }
        guard !groups.isEmpty else { return }
        for pgid in groups where pgid > 1 { kill(-pgid, SIGTERM) }
        usleep(200_000)
        for pgid in groups where pgid > 1 { kill(-pgid, SIGKILL) }
    }
    return true
}()

// MARK: - Ring buffer

/// Fixed-capacity, overwrite-oldest line buffer.
private struct LineRing: Sendable {
    let capacity: Int
    private var storage: [String] = []
    private var next = 0

    init(capacity: Int) {
        self.capacity = max(1, capacity)
        storage.reserveCapacity(self.capacity)
    }

    mutating func append(_ line: String) {
        if storage.count < capacity {
            storage.append(line)
        } else {
            storage[next] = line
            next = (next + 1) % capacity
        }
    }

    var ordered: [String] {
        guard storage.count == capacity, next > 0 else { return storage }
        return Array(storage[next...]) + Array(storage[..<next])
    }
}

// MARK: - ProcessRunner

/// Owns one child process: spawns it in its own process group, streams merged
/// stdout+stderr line by line, and guarantees the child dies with the app.
///
/// Not main-actor bound — every mutable field lives behind a `Mutex`, because pipe
/// and reaper callbacks arrive on arbitrary queues.
///
/// Launching goes through `posix_spawn` rather than `Foundation.Process` for one
/// reason: `Process` cannot put the child in a new process group, and without that a
/// `kill(-pgid, …)` aimed at llama-server's or python's grandchildren would signal
/// this app's own group instead.
final class ProcessRunner: Sendable {

    /// Number of log lines retained for `recentLines()`.
    static let logCapacity = 500

    let executable: URL
    let arguments: [String]
    /// Merged on top of the app's own environment when spawning.
    let environment: [String: String]

    private struct State {
        var pid: pid_t?
        var pgid: pid_t?
        var launching = false
        var lastExitCode: Int32?
        /// Strong reference to the pipe reader — see `startReading`.
        var reader: FileHandle?
        var pending: [UInt8] = []
        var ring = LineRing(capacity: ProcessRunner.logCapacity)
        var onLogLine: (@Sendable (String) -> Void)?
        var onExit: (@Sendable (Int32) -> Void)?
    }

    private let state = Mutex(State())
    private let reaperQueue = DispatchQueue(label: "com.solidchat.processrunner.reaper", qos: .utility)

    init(executable: URL, arguments: [String], environment: [String: String]) {
        self.executable = executable
        self.arguments = arguments
        self.environment = environment
    }

    deinit {
        // Last resort: never leave an orphan behind, and never leak the pipe fd.
        closeReader()
        let (pid, pgid) = state.withLock { ($0.pid, $0.pgid) }
        guard let pid else { return }
        Self.signal(SIGTERM, pid: pid, pgid: pgid)
        usleep(100_000)
        Self.signal(SIGKILL, pid: pid, pgid: pgid)
        if let pgid { liveChildGroups.withLock { _ = $0.remove(pgid) } }
    }

    // MARK: Callbacks

    /// Called once per line of merged stdout/stderr, off the main actor.
    var onLogLine: (@Sendable (String) -> Void)? {
        get { state.withLock { $0.onLogLine } }
        set { state.withLock { $0.onLogLine = newValue } }
    }

    /// Called once when the child is reaped. Negative values are `-signal`.
    var onExit: (@Sendable (Int32) -> Void)? {
        get { state.withLock { $0.onExit } }
        set { state.withLock { $0.onExit = newValue } }
    }

    // MARK: Status

    var isRunning: Bool {
        state.withLock { $0.pid != nil || $0.launching }
    }

    var processIdentifier: pid_t? {
        state.withLock { $0.pid }
    }

    /// Exit code of the most recent run, once it has been reaped.
    var lastExitCode: Int32? {
        state.withLock { $0.lastExitCode }
    }

    /// Up to the last `logCapacity` lines of output, oldest first.
    func recentLines() -> [String] {
        state.withLock { $0.ring.ordered }
    }

    // MARK: Launch

    func start() throws {
        try state.withLock { s throws(ProcessRunnerError) in
            guard s.pid == nil, !s.launching else { throw ProcessRunnerError.alreadyRunning }
            s.launching = true
            s.pending.removeAll(keepingCapacity: true)
            s.lastExitCode = nil
        }

        do {
            let (pid, readFD) = try spawn()
            state.withLock {
                $0.pid = pid
                $0.pgid = pid          // POSIX_SPAWN_SETPGROUP with pgroup 0 ⇒ pgid == pid
                $0.launching = false
            }
            _ = atexitReaperInstalled
            liveChildGroups.withLock { _ = $0.insert(pid) }
            startReading(readFD)
            startReaping(pid: pid)
        } catch {
            state.withLock { $0.launching = false }
            throw error
        }
    }

    /// Fires up the child and hands back its pid plus the read end of the merged pipe.
    private func spawn() throws -> (pid_t, Int32) {
        let path = executable.path
        guard FileManager.default.isExecutableFile(atPath: path) else {
            throw ProcessRunnerError.executableMissing(path)
        }

        var fds: [Int32] = [-1, -1]
        guard pipe(&fds) == 0 else { throw ProcessRunnerError.pipeFailed(errno) }
        let readFD = fds[0]
        let writeFD = fds[1]
        // Neither raw descriptor should survive the exec; the dup2s below re-open
        // 1 and 2 for the child (dup2 always clears FD_CLOEXEC on the new fd).
        _ = fcntl(readFD, F_SETFD, FD_CLOEXEC)
        _ = fcntl(writeFD, F_SETFD, FD_CLOEXEC)

        var actions: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&actions)
        defer { posix_spawn_file_actions_destroy(&actions) }
        posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0)
        posix_spawn_file_actions_adddup2(&actions, writeFD, 1)
        posix_spawn_file_actions_adddup2(&actions, writeFD, 2)

        var attr: posix_spawnattr_t?
        posix_spawnattr_init(&attr)
        defer { posix_spawnattr_destroy(&attr) }
        // Own process group (pgid == child pid) so we can signal the whole tree —
        // llama-server and python both spawn helpers of their own.
        posix_spawnattr_setflags(&attr, Int16(POSIX_SPAWN_SETPGROUP))
        posix_spawnattr_setpgroup(&attr, 0)

        var merged = ProcessInfo.processInfo.environment
        for (key, value) in environment { merged[key] = value }

        var argv: [UnsafeMutablePointer<CChar>?] = ([path] + arguments).map { strdup($0) }
        argv.append(nil)
        var envp: [UnsafeMutablePointer<CChar>?] = merged.map { strdup("\($0.key)=\($0.value)") }
        envp.append(nil)
        defer {
            for slot in argv { free(slot) }
            for slot in envp { free(slot) }
        }

        var pid: pid_t = 0
        let status = posix_spawn(&pid, path, &actions, &attr, argv, envp)

        // The parent must drop its copy of the write end, or the pipe never sees EOF.
        close(writeFD)

        guard status == 0 else {
            close(readFD)
            throw ProcessRunnerError.spawnFailed(status, path)
        }
        return (pid, readFD)
    }

    // MARK: Output

    /// Non-blocking, back-pressure-free drain of the merged pipe.
    ///
    /// The handle is parked in `state` deliberately. Nothing else owns it, and a
    /// `FileHandle` that deallocates silently stops delivering `readabilityHandler`
    /// callbacks — which would leave the child wedged the instant it filled the
    /// 64 KB pipe buffer. llama-server hits that within its first second of logs.
    private func startReading(_ readFD: Int32) {
        let handle = FileHandle(fileDescriptor: readFD, closeOnDealloc: false)
        state.withLock { $0.reader = handle }
        handle.readabilityHandler = { [weak self] source in
            let chunk = source.availableData
            if chunk.isEmpty {          // EOF: every writer is gone
                self?.closeReader()
                self?.flushPending()
                return
            }
            self?.ingest(chunk)
        }
    }

    /// Tears the pipe reader down exactly once. Called on EOF and, as a backstop,
    /// from `deinit`; never from `terminate()`, where the handler may be mid-read.
    private func closeReader() {
        let handle: FileHandle? = state.withLock { s in
            defer { s.reader = nil }
            return s.reader
        }
        guard let handle else { return }
        handle.readabilityHandler = nil
        try? handle.close()
    }

    /// True once the pipe has hit EOF and been torn down.
    private var readerFinished: Bool { state.withLock { $0.reader == nil } }

    private func ingest(_ chunk: Data) {
        var produced: [String] = []
        let sink: (@Sendable (String) -> Void)? = state.withLock { s in
            s.pending.append(contentsOf: chunk)
            // Treat CR as a terminator too: llama.cpp draws progress with \r.
            while let idx = s.pending.firstIndex(where: { $0 == 0x0A || $0 == 0x0D }) {
                let line = String(decoding: s.pending[..<idx], as: UTF8.self)
                s.pending.removeSubrange(0...idx)
                if !line.isEmpty {
                    s.ring.append(line)
                    produced.append(line)
                }
            }
            // A pathologically long unterminated line must not grow without bound.
            if s.pending.count > 64 * 1024 {
                let line = String(decoding: s.pending, as: UTF8.self)
                s.pending.removeAll(keepingCapacity: true)
                s.ring.append(line)
                produced.append(line)
            }
            return s.onLogLine
        }
        guard let sink else { return }
        for line in produced { sink(line) }   // never call out while holding the lock
    }

    private func flushPending() {
        var produced: [String] = []
        let sink: (@Sendable (String) -> Void)? = state.withLock { s in
            guard !s.pending.isEmpty else { return s.onLogLine }
            let line = String(decoding: s.pending, as: UTF8.self)
            s.pending.removeAll(keepingCapacity: true)
            if !line.isEmpty {
                s.ring.append(line)
                produced.append(line)
            }
            return s.onLogLine
        }
        guard let sink else { return }
        for line in produced { sink(line) }
    }

    // MARK: Exit

    private func startReaping(pid: pid_t) {
        reaperQueue.async { [self] in
            var status: Int32 = 0
            var reaped: pid_t = -1
            repeat {
                reaped = waitpid(pid, &status, 0)
            } while reaped < 0 && errno == EINTR

            let code: Int32
            if reaped < 0 {
                code = -1
            } else if (status & 0x7F) == 0 {
                code = (status >> 8) & 0xFF              // normal exit
            } else {
                code = -(status & 0x7F)                  // killed by signal N ⇒ -N
            }

            // Let the pipe finish draining before announcing the exit, so a listener
            // that calls recentLines() from onExit sees the child's dying words —
            // that tail is the whole diagnosis when an engine fails to load.
            // Bounded, because a surviving grandchild can hold the write end open.
            let deadline = Date().addingTimeInterval(1)
            while !readerFinished && Date() < deadline { usleep(20_000) }

            handleExit(code: code, pgid: pid)
        }
    }

    private func handleExit(code: Int32, pgid: pid_t) {
        liveChildGroups.withLock { _ = $0.remove(pgid) }
        let callback: (@Sendable (Int32) -> Void)? = state.withLock { s in
            s.pid = nil
            s.pgid = nil
            s.launching = false
            s.lastExitCode = code
            return s.onExit
        }
        callback?(code)
    }

    // MARK: Termination

    /// SIGTERM the child's whole process group, wait up to 5s, then SIGKILL it.
    /// Blocking by design — call it off the main thread.
    func terminate() {
        let (pid, pgid) = state.withLock { ($0.pid, $0.pgid) }
        guard let pid else { return }

        Self.signal(SIGTERM, pid: pid, pgid: pgid)

        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if !isRunning { return }
            usleep(50_000)
        }

        guard isRunning else { return }
        Self.signal(SIGKILL, pid: pid, pgid: pgid)
    }

    /// Signals the child's process group when it genuinely has its own, and never
    /// this app's group; falls back to the single pid otherwise.
    private static func signal(_ sig: Int32, pid: pid_t, pgid: pid_t?) {
        if let pgid, pgid > 1, pgid != getpgrp() {
            if kill(-pgid, sig) == 0 { return }
        }
        _ = kill(pid, sig)
    }
}
