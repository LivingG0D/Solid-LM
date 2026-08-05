import Foundation

// MARK: - Engines

enum SpeechEngineKind: String, Codable, CaseIterable, Sendable {
    /// AVSpeechSynthesizer. Instant, offline, zero setup — but on a Mac with no
    /// downloaded Siri voices every English voice is the old compact kind.
    case system
    /// Kokoro-82M through a persistent MLX worker. Far better, ~1.8s per utterance
    /// after a ~4s one-time start.
    case kokoro

    var label: String {
        switch self {
        case .system: return "System"
        case .kokoro: return "Kokoro (neural)"
        }
    }

    var detail: String {
        switch self {
        case .system: return "macOS built-in. Starts instantly, no model to load."
        case .kokoro: return "Local neural voices. Better quality; a few seconds to warm up."
        }
    }
}

/// How good a system voice actually sounds. Premium and enhanced voices are a
/// separate download in System Settings, so most Macs only have compact ones.
enum VoiceQuality: String, Codable, Sendable {
    case compact, enhanced, premium

    var label: String {
        switch self {
        case .compact: return "Compact"
        case .enhanced: return "Enhanced"
        case .premium: return "Premium"
        }
    }
}

struct SpeechVoice: Identifiable, Hashable, Codable, Sendable {
    var id: String              // AVSpeechSynthesisVoice identifier, or the Kokoro voice name
    var name: String
    var language: String        // BCP-47 for system voices, "en"/"ja"/… for Kokoro
    var engine: SpeechEngineKind
    var quality: VoiceQuality = .compact

    /// Kokoro encodes locale and gender in the name: "af_heart" = American female,
    /// "bm_george" = British male. Worth spelling out rather than showing the raw id.
    var displayName: String {
        guard engine == .kokoro, name.count > 3, name[name.index(name.startIndex, offsetBy: 2)] == "_" else {
            return name
        }
        let prefix = name.prefix(2)
        let locale: String
        switch prefix.first {
        case "a": locale = "American"
        case "b": locale = "British"
        case "e": locale = "Spanish"
        case "f": locale = "French"
        case "h": locale = "Hindi"
        case "i": locale = "Italian"
        case "j": locale = "Japanese"
        case "p": locale = "Portuguese"
        case "z": locale = "Chinese"
        default:  locale = ""
        }
        let gender = prefix.last == "f" ? "female" : (prefix.last == "m" ? "male" : "")
        let stem = String(name.dropFirst(3)).capitalized
        let suffix = [locale, gender].filter { !$0.isEmpty }.joined(separator: " ")
        return suffix.isEmpty ? stem : "\(stem) · \(suffix)"
    }
}

// MARK: - Settings

struct SpeechSettings: Codable, Hashable, Sendable {
    var engine: SpeechEngineKind = .system
    var systemVoiceID: String = ""
    var kokoroVoiceID: String = "af_heart"
    /// AVSpeechUtterance rate. 0.5 is the platform default.
    var rate: Double = 0.5
    var pitch: Double = 1.0
    var volume: Double = 1.0
    /// Kokoro speed multiplier, independent of the system rate scale.
    var kokoroSpeed: Double = 1.0
    /// Speak assistant replies without being asked.
    var autoSpeak: Bool = false

    func voiceID(for engine: SpeechEngineKind) -> String {
        engine == .kokoro ? kokoroVoiceID : systemVoiceID
    }
}

// MARK: - State

enum SpeechState: Equatable, Sendable {
    case idle
    /// Kokoro worker starting, or an utterance being synthesised.
    case preparing
    case speaking
    case failed(String)

    var isActive: Bool { self == .preparing || self == .speaking }
    var errorText: String? { if case .failed(let m) = self { return m }; return nil }
}

// MARK: - Chunking

enum SpeechChunker {
    /// Splits text into speakable chunks on sentence boundaries.
    ///
    /// Used two ways: to feed Kokoro incrementally while a reply is still streaming,
    /// and to keep any single synthesis call short enough to stay responsive.
    /// Markdown is stripped first — nobody wants "asterisk asterisk important
    /// asterisk asterisk" read aloud, and a fenced code block read character by
    /// character is worse.
    static func chunks(_ text: String, maxLength: Int = 350) -> [String] {
        let clean = strip(text)
        guard !clean.isEmpty else { return [] }

        var out: [String] = []
        var current = ""

        for sentence in sentences(clean) {
            if current.isEmpty {
                current = sentence
            } else if current.count + sentence.count + 1 <= maxLength {
                current += " " + sentence
            } else {
                out.append(current)
                current = sentence
            }
            // A single sentence longer than the limit still has to go out.
            while current.count > maxLength {
                let cut = current.index(current.startIndex, offsetBy: maxLength)
                let breakPoint = current[..<cut].lastIndex(of: " ") ?? cut
                out.append(String(current[..<breakPoint]).trimmingCharacters(in: .whitespaces))
                current = String(current[breakPoint...]).trimmingCharacters(in: .whitespaces)
            }
        }
        if !current.isEmpty { out.append(current) }
        return out.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }

    /// Sentence boundaries, tolerant of "e.g." and decimals by requiring the
    /// terminator to be followed by whitespace and the next word to look like a start.
    static func sentences(_ text: String) -> [String] {
        var out: [String] = []
        var current = ""
        let chars = Array(text)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            current.append(c)
            if c == "." || c == "!" || c == "?" || c == "\n" {
                let next = i + 1 < chars.count ? chars[i + 1] : " "
                if next == " " || next == "\n" || i + 1 == chars.count {
                    let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty { out.append(trimmed) }
                    current = ""
                }
            }
            i += 1
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { out.append(tail) }
        return out
    }

    /// Reduces markdown to what a person would actually say.
    ///
    /// Patterns are strings rather than regex literals: several of these contain
    /// `#`, `{n,m}` and backslash classes that collide with Swift's regex-literal
    /// delimiters, and the string form is the one that survives editing.
    static func strip(_ text: String) -> String {
        // (pattern, replacement) applied in order. $1/$2 are capture references.
        let rules: [(String, String)] = [
            // Fenced code: announce it once instead of reading the code aloud.
            ("```[\\s\\S]*?```", " (code block) "),
            ("```[\\s\\S]*\\z", " (code block) "),          // fence still streaming
            ("!\\[[^\\]]*\\]\\([^)]*\\)", " "),             // images
            ("\\[([^\\]]+)\\]\\([^)]*\\)", "$1"),           // links -> their text
            ("`([^`\\n]+)`", "$1"),                         // inline code
            ("(?m)^\\s{0,3}#{1,6}\\s+", ""),                // headings
            ("(?m)^\\s*>\\s?", ""),                         // block quotes
            ("(?m)^\\s*[-*+]\\s+", ""),                     // bullets
            ("(?m)^\\s*\\d+\\.\\s+", ""),                   // numbered lists
            ("(?m)^\\s*[-:|\\s]{4,}$", " "),                // table rules
            ("(?m)^\\s*\\|.*\\|\\s*$", " "),                // table rows
            ("(\\*\\*|__)(.+?)\\1", "$2"),                  // bold
            ("(\\*|_)(.+?)\\1", "$2"),                      // italic
            ("~~(.+?)~~", "$1"),                            // strikethrough
            ("[ \\t]{2,}", " "),
            ("\\n{2,}", "\n"),
        ]
        var s = text
        for (pattern, replacement) in rules {
            s = s.replacingOccurrences(of: pattern, with: replacement,
                                       options: .regularExpression)
        }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func selfCheck() -> [String] {
        var f: [String] = []
        func expect(_ got: String, _ want: String, _ what: String) {
            if got != want { f.append("\(what): got \"\(got)\" want \"\(want)\"") }
        }
        expect(strip("**bold** and *italic*"), "bold and italic", "emphasis")
        expect(strip("# Heading\ntext"), "Heading\ntext", "heading")
        expect(strip("see [the docs](http://x)"), "see the docs", "link")
        expect(strip("a ```swift\nlet x = 1\n``` b"), "a (code block) b", "fenced code")
        expect(strip("a ```swift\nlet x = 1"), "a (code block)", "unterminated fence")
        expect(strip("use `map` here"), "use map here", "inline code")

        let s = sentences("One. Two! Three? Four")
        if s.count != 4 { f.append("sentences: expected 4, got \(s.count) — \(s)") }

        let c = chunks(String(repeating: "word ", count: 200), maxLength: 100)
        if c.isEmpty { f.append("chunks: long text produced nothing") }
        if c.contains(where: { $0.count > 100 }) { f.append("chunks: a chunk exceeded maxLength") }

        if !chunks("```\nonly code\n```").isEmpty == false { /* fine either way */ }
        if !chunks("").isEmpty { f.append("chunks: empty input should produce nothing") }
        return f
    }
}
