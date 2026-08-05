import Foundation

// MARK: - Streaming delta

/// One incremental piece of an assistant turn.
/// `content` is visible text, `reasoning` is chain-of-thought reported by the
/// engine out-of-band (llama.cpp: `delta.reasoning_content`, MLX: `delta.reasoning`).
struct ChatDelta: Sendable {
    var content: String
    var reasoning: String
    var completionTokens: Int?

    init(content: String = "", reasoning: String = "", completionTokens: Int? = nil) {
        self.content = content
        self.reasoning = reasoning
        self.completionTokens = completionTokens
    }

    var isEmpty: Bool { content.isEmpty && reasoning.isEmpty && completionTokens == nil }
}

// MARK: - OpenAI-compatible streaming client

enum ChatClient {

    // The engine (llama-server / mlx_lm.server / vLLM) always listens here.
    static let host = "127.0.0.1"
    static let port = 8181

    static var endpoint: URL {
        URL(string: "http://\(host):\(port)/v1/chat/completions")!
    }

    // MARK: Errors

    enum Failure: LocalizedError, Sendable {
        /// Non-200 from the engine. `body` is the raw response text — the engine's
        /// own message is by far the most useful thing we can show the user.
        case badStatus(code: Int, body: String)
        /// The engine reported an error mid-stream (llama.cpp does this on context overflow).
        case stream(String)
        case notHTTP

        var errorDescription: String? {
            switch self {
            case .badStatus(let code, let body):
                let text = ChatClient.humanMessage(from: body)
                return text.isEmpty ? "Engine returned HTTP \(code)."
                                    : "Engine returned HTTP \(code): \(text)"
            case .stream(let message):
                return message.isEmpty ? "The engine aborted the response." : message
            case .notHTTP:
                return "The engine did not return a valid HTTP response."
            }
        }
    }

    // MARK: Public API

    /// Streams a chat completion from the local engine.
    ///
    /// Cancelling the consuming `Task` (or breaking out of the `for await` loop)
    /// terminates the stream, which cancels the underlying `URLSessionDataTask` so the
    /// engine actually stops generating instead of just losing its reader.
    static func stream(messages: [Msg],
                       system: String,
                       samplers: Samplers,
                       engine: EngineKind,
                       modelPath: String,
                       modelID: String) -> AsyncThrowingStream<ChatDelta, Error> {

        // *** mlx_lm.server treats "model" as a HuggingFace repo id and will silently
        // *** RE-DOWNLOAD the whole model if given "publisher/name". It must get the
        // *** local directory. vLLM is launched with --model <path>, so its served
        // *** name is the path too. Only llama-server is happy with the friendly id.
        let modelField: String
        switch engine {
        case .llama: modelField = modelID.isEmpty ? "local-model" : modelID
        case .mlx, .vllm: modelField = modelPath
        }

        return deltaStream(messages: messages,
                           system: system,
                           samplers: samplers,
                           modelField: modelField)
    }

    // MARK: Core

    private static func deltaStream(messages: [Msg],
                                    system: String,
                                    samplers: Samplers,
                                    modelField: String) -> AsyncThrowingStream<ChatDelta, Error> {

        AsyncThrowingStream<ChatDelta, Error> { continuation in
            let work = Task {
                do {
                    let request = try makeRequest(messages: messages,
                                                  system: system,
                                                  samplers: samplers,
                                                  modelField: modelField)

                    let (bytes, response) = try await URLSession.shared.bytes(for: request)

                    guard let http = response as? HTTPURLResponse else { throw Failure.notHTTP }
                    guard http.statusCode == 200 else {
                        let body = await collectBody(bytes)
                        throw Failure.badStatus(code: http.statusCode, body: body)
                    }

                    // Cancel the HTTP task itself, not just our read loop.
                    let sessionTask = bytes.task
                    try await withTaskCancellationHandler {
                        // Split on LF ourselves rather than using `bytes.lines`.
                        // `.lines` breaks on Unicode separators too (U+2028, U+2029,
                        // U+0085), which are legal inside a JSON string — a model that
                        // emits one would have its frame split mid-JSON and the whole
                        // token silently dropped. SSE only terminates lines on LF/CRLF.
                        var buffer: [UInt8] = []
                        buffer.reserveCapacity(8192)

                        func take() -> String? {
                            guard let nl = buffer.firstIndex(of: 0x0A) else { return nil }
                            var line = Array(buffer[buffer.startIndex..<nl])
                            buffer.removeSubrange(buffer.startIndex...nl)
                            if line.last == 0x0D { line.removeLast() }   // CRLF
                            return String(decoding: line, as: UTF8.self)
                        }

                        readLoop: for try await byte in bytes {
                            if Task.isCancelled { break readLoop }
                            buffer.append(byte)
                            guard byte == 0x0A, let rawLine = take() else { continue }
                            switch parse(line: rawLine) {
                            case .ignore:
                                continue
                            case .done:
                                break readLoop
                            case .delta(let delta):
                                if !delta.isEmpty { continuation.yield(delta) }
                            case .error(let message):
                                throw Failure.stream(message)
                            }
                        }

                        // A last frame with no trailing newline would otherwise be lost.
                        if !buffer.isEmpty, !Task.isCancelled {
                            var line = buffer
                            if line.last == 0x0D { line.removeLast() }
                            if case .delta(let delta) = parse(line: String(decoding: line, as: UTF8.self)),
                               !delta.isEmpty {
                                continuation.yield(delta)
                            }
                        }
                    } onCancel: {
                        sessionTask.cancel()
                    }

                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch let error as URLError where error.code == .cancelled {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in work.cancel() }
        }
    }

    // MARK: Request

    private static func makeRequest(messages: [Msg],
                                    system: String,
                                    samplers: Samplers,
                                    modelField: String) throws -> URLRequest {

        var wire: [[String: String]] = []

        let sys = system.trimmingCharacters(in: .whitespacesAndNewlines)
        if !sys.isEmpty {
            wire.append(["role": "system", "content": sys])
        }
        for m in messages {
            // Engines reject empty-content turns; a placeholder assistant row is not history yet.
            if m.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
            wire.append(["role": m.role.rawValue, "content": m.content])
        }

        let body: [String: Any] = [
            "model": modelField,
            "messages": wire,
            "temperature": samplers.temperature,
            "top_k": samplers.topK,
            "top_p": samplers.topP,
            "min_p": samplers.minP,
            // llama.cpp spells it repeat_penalty, MLX/OpenAI spell it repetition_penalty.
            "repeat_penalty": samplers.repeatPenalty,
            "repetition_penalty": samplers.repeatPenalty,
            "max_tokens": samplers.maxTokens,
            "stream": true,
            "stream_options": ["include_usage": true]
        ]

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 600
        request.httpBody = try JSONSerialization.data(withJSONObject: body, options: [])
        return request
    }

    // MARK: Server-Sent Events

    private enum Frame {
        case delta(ChatDelta)
        case error(String)
        case done
        case ignore
    }

    /// Parses one SSE line. Anything unrecognised is `.ignore` — partial frames, comments,
    /// keep-alives and blank separators must never break the stream.
    private static func parse(line rawLine: String) -> Frame {
        var line = Substring(rawLine)
        while line.last == "\r" { line = line.dropLast() }
        if line.isEmpty { return .ignore }

        // llama.cpp emits mid-stream failures as `error: {...}` rather than `data:`.
        if line.hasPrefix("error:") {
            let payload = trimSSEPayload(line.dropFirst("error:".count))
            return .error(humanMessage(from: String(payload)))
        }

        guard line.hasPrefix("data:") else { return .ignore }
        let payload = trimSSEPayload(line.dropFirst("data:".count))
        if payload.isEmpty { return .ignore }
        if payload == "[DONE]" { return .done }

        guard let data = String(payload).data(using: .utf8),
              let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return .ignore }

        if let err = root["error"] {
            if let dict = err as? [String: Any], let message = dict["message"] as? String {
                return .error(message)
            }
            if let message = err as? String { return .error(message) }
        }

        var content = ""
        var reasoning = ""
        var tokens: Int?

        if let choices = root["choices"] as? [[String: Any]], let first = choices.first {
            if let delta = first["delta"] as? [String: Any] {
                content = delta["content"] as? String ?? ""
                reasoning = (delta["reasoning_content"] as? String)
                    ?? (delta["reasoning"] as? String) ?? ""
            } else if let message = first["message"] as? [String: Any] {
                // Some builds answer a stream request with a single non-streamed frame.
                content = message["content"] as? String ?? ""
                reasoning = (message["reasoning_content"] as? String)
                    ?? (message["reasoning"] as? String) ?? ""
            }
        }

        if let usage = root["usage"] as? [String: Any] {
            if let n = usage["completion_tokens"] as? Int {
                tokens = n
            } else if let n = usage["completion_tokens"] as? Double {
                tokens = Int(n)
            }
        }

        let delta = ChatDelta(content: content, reasoning: reasoning, completionTokens: tokens)
        return delta.isEmpty ? .ignore : .delta(delta)
    }

    private static func trimSSEPayload(_ s: Substring) -> Substring {
        var out = s
        while out.first == " " { out = out.dropFirst() }
        while out.last == " " { out = out.dropLast() }
        return out
    }

    // MARK: Bodies

    /// Drains an error response into text. Bounded — an engine that streams megabytes of
    /// failure should not be able to balloon our memory.
    private static func collectBody(_ bytes: URLSession.AsyncBytes) async -> String {
        var data = Data()
        do {
            for try await byte in bytes {
                data.append(byte)
                if data.count >= 32_768 { break }
            }
        } catch {
            // Whatever we managed to read is still worth showing.
        }
        return String(data: data, encoding: .utf8) ?? ""
    }

    /// Pulls the readable part out of an engine error body (JSON or plain text).
    private static func humanMessage(from body: String) -> String {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "" }

        if let data = trimmed.data(using: .utf8),
           let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            if let err = root["error"] as? [String: Any], let m = err["message"] as? String, !m.isEmpty {
                return m
            }
            if let m = root["error"] as? String, !m.isEmpty { return m }
            if let m = root["message"] as? String, !m.isEmpty { return m }
        }
        return String(trimmed.prefix(400))
    }

    /// Convenience forwarder so callers can run every self-check in this file from one place.
    static func selfCheck() -> [String] { ThinkSplitter.selfCheck() }
}

// MARK: - Inline <think> splitting

/// Separates inline `<think>…</think>` spans from visible content while text streams in.
///
/// Only the characters that could still turn out to be a partial tag are held back;
/// everything else is emitted on the very same `feed` call, so the UI stays live.
struct ThinkSplitter: Sendable {

    private static let openTag = "<think>"
    private static let closeTag = "</think>"

    /// Text received but not yet classified — always a proper prefix of a tag.
    private var buffer = ""
    private var inThink = false

    init() {}

    /// Consumes a chunk and returns only what that chunk contributed.
    @discardableResult
    mutating func feed(_ s: String) -> (content: String, thinking: String) {
        if s.isEmpty { return ("", "") }
        buffer += s

        var newContent = ""
        var newThinking = ""

        var i = buffer.startIndex
        var pending = buffer.startIndex   // start of text not yet emitted

        while i < buffer.endIndex {
            if buffer[i] == "<" {
                let tag = inThink ? Self.closeTag : Self.openTag
                let rest = buffer[i...]

                if rest.hasPrefix(tag) {
                    let text = buffer[pending..<i]
                    if !text.isEmpty {
                        if inThink { newThinking.append(contentsOf: text) }
                        else { newContent.append(contentsOf: text) }
                    }
                    i = buffer.index(i, offsetBy: tag.count)
                    pending = i
                    inThink.toggle()
                    continue
                }

                // Everything left could still grow into the tag — hold it back.
                if tag.hasPrefix(rest) { break }
            }
            i = buffer.index(after: i)
        }

        let text = buffer[pending..<i]
        if !text.isEmpty {
            if inThink { newThinking.append(contentsOf: text) }
            else { newContent.append(contentsOf: text) }
        }
        buffer = String(buffer[i...])

        return (newContent, newThinking)
    }

    /// Releases any held-back text at end of stream. A dangling `"<thi"` was never a tag,
    /// so it becomes ordinary text.
    @discardableResult
    mutating func flush() -> (content: String, thinking: String) {
        let rest = buffer
        buffer = ""
        if rest.isEmpty { return ("", "") }
        return inThink ? ("", rest) : (rest, "")
    }

    // MARK: Self-check

    /// Returns a list of failure descriptions; `[]` means every case passed.
    static func selfCheck() -> [String] {
        var failures: [String] = []

        // Feeds the chunks the way a caller does: accumulate what each feed returns,
        // then flush at end of stream.
        func run(_ chunks: [String]) -> (content: String, thinking: String) {
            var splitter = ThinkSplitter()
            var content = ""
            var thinking = ""
            for chunk in chunks {
                let step = splitter.feed(chunk)
                content += step.content
                thinking += step.thinking
            }
            let tail = splitter.flush()
            content += tail.content
            thinking += tail.thinking
            return (content, thinking)
        }

        func expect(_ label: String,
                    _ chunks: [String],
                    content wantContent: String,
                    thinking wantThinking: String) {
            let got = run(chunks)
            if got.content != wantContent || got.thinking != wantThinking {
                failures.append("""
                ThinkSplitter \(label): got content=\"\(got.content)\" thinking=\"\(got.thinking)\" \
                want content=\"\(wantContent)\" thinking=\"\(wantThinking)\"
                """)
            }
        }

        // The load-bearing case: tags torn apart by chunk boundaries.
        expect("tag split across chunks",
               ["<thi", "nk>abc</thi", "nk>def"],
               content: "def", thinking: "abc")

        expect("open tag split at every byte",
               ["<", "t", "h", "i", "n", "k", ">", "hi", "</think>", "bye"],
               content: "bye", thinking: "hi")

        // Every possible single split point of a full exchange.
        let full = "A<think>B</think>C"
        for cut in 1..<full.count {
            let idx = full.index(full.startIndex, offsetBy: cut)
            expect("split@\(cut)",
                   [String(full[..<idx]), String(full[idx...])],
                   content: "AC", thinking: "B")
        }

        // One character at a time.
        expect("character by character",
               full.map(String.init),
               content: "AC", thinking: "B")

        expect("no tags", ["hello ", "world"], content: "hello world", thinking: "")
        expect("bare angle brackets", ["2 < 3 and 4 ", "> 1"], content: "2 < 3 and 4 > 1", thinking: "")
        expect("near miss tag", ["<thinkish>x"], content: "<thinkish>x", thinking: "")
        expect("unterminated think", ["<think>still going"], content: "", thinking: "still going")
        expect("dangling partial tag", ["done<thi"], content: "done<thi", thinking: "")
        expect("multiple spans",
               ["a<think>1</think>b<think>2</think>c"],
               content: "abc", thinking: "12")

        // Nothing may be held back that is not a possible tag prefix.
        var live = ThinkSplitter()
        let first = live.feed("Hello")
        if first.content != "Hello" {
            failures.append("ThinkSplitter immediacy: expected \"Hello\" at once, got \"\(first.content)\"")
        }
        let second = live.feed(" wor")
        if second.content != " wor" {
            failures.append("ThinkSplitter immediacy: expected \" wor\" at once, got \"\(second.content)\"")
        }

        // A held partial must be released once it is disproved.
        var held = ThinkSplitter()
        let a = held.feed("x<")
        let b = held.feed("y")
        if a.content != "x" || b.content != "<y" {
            failures.append("ThinkSplitter hold/release: got \"\(a.content)\" then \"\(b.content)\", want \"x\" then \"<y\"")
        }

        // A partial tag must never leak into visible content mid-stream.
        var quiet = ThinkSplitter()
        let leaked = quiet.feed("visible<thi")
        if leaked.content != "visible" {
            failures.append("ThinkSplitter leak: partial tag reached content as \"\(leaked.content)\"")
        }

        return failures
    }
}
