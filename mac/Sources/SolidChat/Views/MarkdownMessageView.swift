import SwiftUI
import AppKit

// ---------------------------------------------------------------------------
// Native markdown renderer for streaming LLM output.
//
// Everything below `MarkdownMessageView` is file-private so it cannot collide
// with symbols owned by other files in the module. The only public surface is:
//
//     MarkdownMessageView(text:)          — the view
//     MarkdownMessageView.selfCheck()     — parser self-test, returns failures
//
// Design notes:
//  * Block parsing is a single linear pass over `[Substring]` lines. No string
//    is ever built by repeated concatenation inside a loop; code-block bodies
//    are produced with `joined(separator:)`, which is one allocation.
//  * Because the view re-renders on every streamed token, block parsing,
//    inline attributed-string parsing, and syntax highlighting are all memoised
//    in a small bounded MainActor cache, so re-evaluating `body` with unchanged
//    text (which SwiftUI does often) costs a dictionary lookup.
// ---------------------------------------------------------------------------

struct MarkdownMessageView: View {
    let text: String

    var body: some View {
        MDBlockStack(blocks: MDCache.blocks(for: text))
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Test hook: block count for `text`. The parser itself is file-private,
    /// so out-of-file stress tests go through here.
    nonisolated static func probeParse(_ text: String) -> Int { MD.parse(text).count }

    /// Exercises the block parser. Returns a list of failure descriptions;
    /// an empty array means everything passed.
    nonisolated static func selfCheck() -> [String] {
        var failures: [String] = []
        func check(_ ok: Bool, _ message: @autoclosure () -> String) {
            if !ok { failures.append(message()) }
        }

        // 1. Fenced code between paragraphs.
        do {
            let blocks = MD.parse("intro\n\n```swift\nlet x = 1\n```\n\noutro")
            check(blocks.count == 3, "fence: expected 3 blocks, got \(blocks.count)")
            if blocks.count == 3, case .code(let language, let source, let closed) = blocks[1] {
                check(language == "swift", "fence: language was \"\(language)\"")
                check(source == "let x = 1", "fence: source was \"\(source)\"")
                check(closed, "fence: block should be marked closed")
            } else {
                check(false, "fence: block 1 was not a code block")
            }
            if blocks.count == 3, case .paragraph(let tail) = blocks[2] {
                check(tail == "outro", "fence: trailing paragraph was \"\(tail)\"")
            } else {
                check(false, "fence: block 2 was not a paragraph")
            }
        }

        // 2. Unterminated fence (mid-stream) must still render as code.
        do {
            let blocks = MD.parse("```python\nprint(1)\nprint(2")
            check(blocks.count == 1, "open fence: expected 1 block, got \(blocks.count)")
            if let first = blocks.first, case .code(let language, let source, let closed) = first {
                check(language == "python", "open fence: language was \"\(language)\"")
                check(source == "print(1)\nprint(2", "open fence: source was \"\(source)\"")
                check(!closed, "open fence: block should be marked unterminated")
            } else {
                check(false, "open fence: block 0 was not a code block")
            }
        }

        // 3. A ``` on its own with nothing after it is still a code block.
        do {
            let blocks = MD.parse("text\n```")
            check(blocks.count == 2, "bare fence: expected 2 blocks, got \(blocks.count)")
            if blocks.count == 2, case .code(_, let source, let closed) = blocks[1] {
                check(source.isEmpty, "bare fence: source was \"\(source)\"")
                check(!closed, "bare fence: should be unterminated")
            } else {
                check(false, "bare fence: block 1 was not a code block")
            }
        }

        // 4. Table.
        do {
            let blocks = MD.parse("| Name | Size |\n| :--- | ---: |\n| a | 1 |\n| b | 2 |")
            check(blocks.count == 1, "table: expected 1 block, got \(blocks.count)")
            if let first = blocks.first, case .table(let table) = first {
                check(table.header == ["Name", "Size"], "table: header was \(table.header)")
                check(table.alignments == [.leading, .trailing], "table: alignments were \(table.alignments)")
                check(table.rows.count == 2, "table: expected 2 rows, got \(table.rows.count)")
                check(table.rows.first ?? [] == ["a", "1"], "table: first row was \(table.rows.first ?? [])")
            } else {
                check(false, "table: block 0 was not a table")
            }
        }

        // 5. A pipe-free line is not a table, and a table needs a delimiter row.
        do {
            let blocks = MD.parse("a | b\nnot a delimiter")
            if let first = blocks.first, case .table = first {
                check(false, "table: matched without a delimiter row")
            }
        }

        // 6. Lists: bullets, nesting, ordered markers, task items.
        do {
            let blocks = MD.parse("- one\n- two\n  - nested\n")
            check(blocks.count == 1, "list: expected 1 block, got \(blocks.count)")
            if let first = blocks.first, case .list(let items) = first {
                check(items.count == 3, "list: expected 3 items, got \(items.count)")
                check(items.map(\.level) == [0, 0, 1], "list: levels were \(items.map(\.level))")
                check(items.first?.text == "one", "list: first item was \"\(items.first?.text ?? "")\"")
            } else {
                check(false, "list: block 0 was not a list")
            }
        }
        do {
            let blocks = MD.parse("1. first\n2. second")
            if let first = blocks.first, case .list(let items) = first {
                check(items.map(\.marker) == ["1.", "2."], "ordered list: markers were \(items.map(\.marker))")
            } else {
                check(false, "ordered list: block 0 was not a list")
            }
        }
        do {
            let blocks = MD.parse("- [x] done\n- [ ] todo")
            if let first = blocks.first, case .list(let items) = first {
                check(items.map(\.checked) == [true, false], "task list: checked was \(items.map(\.checked))")
                check(items.first?.text == "done", "task list: text was \"\(items.first?.text ?? "")\"")
            } else {
                check(false, "task list: block 0 was not a list")
            }
        }

        // 7. Headings, including a closed ATX sequence and a trailing '#' in text.
        do {
            let blocks = MD.parse("## Title ##\n\nbody")
            check(blocks.count == 2, "heading: expected 2 blocks, got \(blocks.count)")
            if let first = blocks.first, case .heading(let level, let title) = first {
                check(level == 2, "heading: level was \(level)")
                check(title == "Title", "heading: text was \"\(title)\"")
            } else {
                check(false, "heading: block 0 was not a heading")
            }
        }
        do {
            let blocks = MD.parse("# C#")
            if let first = blocks.first, case .heading(_, let title) = first {
                check(title == "C#", "heading: trailing hash was stripped, got \"\(title)\"")
            } else {
                check(false, "heading: block 0 was not a heading")
            }
        }
        do {
            let blocks = MD.parse("#hashtag not a heading")
            if let first = blocks.first, case .heading = first {
                check(false, "heading: '#hashtag' should not be a heading")
            }
        }

        // 8. Horizontal rule must win over the bullet-list parse.
        do {
            let blocks = MD.parse("above\n\n---\n\nbelow")
            check(blocks.count == 3, "rule: expected 3 blocks, got \(blocks.count)")
            if blocks.count == 3, case .rule = blocks[1] {} else {
                check(false, "rule: block 1 was not a horizontal rule")
            }
        }

        // 9. Blockquote, including a nested code block inside it.
        do {
            let blocks = MD.parse("> quoted line\n> second line")
            check(blocks.count == 1, "quote: expected 1 block, got \(blocks.count)")
            if let first = blocks.first, case .quote(let inner) = first {
                check(inner.count == 1, "quote: expected 1 inner block, got \(inner.count)")
                if let innerFirst = inner.first, case .paragraph(let body) = innerFirst {
                    check(body == "quoted line\nsecond line", "quote: body was \"\(body)\"")
                } else {
                    check(false, "quote: inner block was not a paragraph")
                }
            } else {
                check(false, "quote: block 0 was not a quote")
            }
        }

        // 10. Fence contents are opaque — markers inside code are not parsed.
        do {
            let blocks = MD.parse("```\n# not a heading\n- not a list\n```")
            check(blocks.count == 1, "opaque fence: expected 1 block, got \(blocks.count)")
            if let first = blocks.first, case .code(_, let source, _) = first {
                check(source == "# not a heading\n- not a list", "opaque fence: source was \"\(source)\"")
            } else {
                check(false, "opaque fence: block 0 was not a code block")
            }
        }

        // 11. Empty and whitespace-only input must not crash or emit blocks.
        do {
            check(MD.parse("").isEmpty, "empty: expected no blocks")
            check(MD.parse("\n\n   \n").isEmpty, "blank: expected no blocks")
        }

        // 12. Malformed inline markdown must never throw out of the renderer.
        do {
            let broken = "**unclosed [link](  ~~ `code"
            check(!String(MDInline.attributed(broken).characters).isEmpty,
                  "inline: malformed markdown produced empty output")
        }

        return failures
    }
}

// MARK: - Block model

private struct MDListItem: Hashable, Sendable {
    var marker: String
    var text: String
    var level: Int
    var checked: Bool?
}

private enum MDAlign: Hashable, Sendable {
    case leading, center, trailing

    var horizontal: HorizontalAlignment {
        switch self {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }

    var text: TextAlignment {
        switch self {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }
}

private struct MDTable: Hashable, Sendable {
    var header: [String]
    var alignments: [MDAlign]
    var rows: [[String]]

    var columnCount: Int { max(header.count, alignments.count) }

    func alignment(_ index: Int) -> MDAlign {
        index >= 0 && index < alignments.count ? alignments[index] : .leading
    }
}

private enum MDBlock: Hashable, Sendable {
    case paragraph(String)
    case heading(level: Int, text: String)
    case code(language: String, source: String, closed: Bool)
    case list(items: [MDListItem])
    case quote(blocks: [MDBlock])
    case rule
    case table(MDTable)
}

// MARK: - Block parser

private enum MD {
    static func parse(_ text: String) -> [MDBlock] {
        if text.isEmpty { return [] }
        var lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        // Tolerate CRLF without allocating a second copy of the whole document.
        for index in lines.indices where lines[index].hasSuffix("\r") {
            lines[index] = lines[index].dropLast()
        }
        return parseBlocks(lines, depth: 0)
    }

    // MARK: Line classification

    struct Fence {
        var marker: Character
        var length: Int
        var info: String
    }

    struct ListMark {
        var ordered: Bool
        var marker: String
        var content: String
        var indent: Int
    }

    static func trimLeading(_ line: Substring) -> Substring {
        line.drop(while: { $0 == " " || $0 == "\t" })
    }

    static func isBlank(_ line: Substring) -> Bool {
        line.allSatisfy { $0 == " " || $0 == "\t" }
    }

    static func indentWidth(_ line: Substring) -> Int {
        var width = 0
        for character in line {
            if character == " " { width += 1 }
            else if character == "\t" { width += 4 }
            else { break }
        }
        return width
    }

    static func fenceInfo(_ line: Substring) -> Fence? {
        let trimmed = trimLeading(line)
        guard let first = trimmed.first, first == "`" || first == "~" else { return nil }
        let run = trimmed.prefix(while: { $0 == first })
        guard run.count >= 3 else { return nil }
        let info = trimmed.dropFirst(run.count).trimmingCharacters(in: .whitespaces)
        // A backtick fence may not carry a backtick in its info string.
        if first == "`", info.contains("`") { return nil }
        return Fence(marker: first, length: run.count, info: info)
    }

    static func headingInfo(_ line: Substring) -> (level: Int, text: String)? {
        let trimmed = trimLeading(line)
        guard trimmed.first == "#" else { return nil }
        let hashes = trimmed.prefix(while: { $0 == "#" })
        guard hashes.count <= 6 else { return nil }
        let remainder = trimmed.dropFirst(hashes.count)
        guard remainder.isEmpty || remainder.first == " " || remainder.first == "\t" else { return nil }

        var body = String(remainder).trimmingCharacters(in: .whitespaces)
        // Strip a closing ATX sequence ("## Title ##") but keep a real trailing
        // hash that is part of the words ("# C#").
        if let lastNonHash = body.lastIndex(where: { $0 != "#" }) {
            if body[lastNonHash] == " " || body[lastNonHash] == "\t" {
                body = String(body[..<lastNonHash]).trimmingCharacters(in: .whitespaces)
            }
        } else if !body.isEmpty {
            body = ""
        }
        return (hashes.count, body)
    }

    static func isRule(_ line: Substring) -> Bool {
        let trimmed = trimLeading(line)
        guard let marker = trimmed.first, marker == "-" || marker == "*" || marker == "_" else { return false }
        var count = 0
        for character in trimmed {
            if character == marker { count += 1 }
            else if character == " " || character == "\t" { continue }
            else { return false }
        }
        return count >= 3
    }

    static func listInfo(_ line: Substring) -> ListMark? {
        let indent = indentWidth(line)
        let trimmed = trimLeading(line)
        guard let first = trimmed.first else { return nil }

        if first == "-" || first == "*" || first == "+" {
            let rest = trimmed.dropFirst()
            guard rest.isEmpty || rest.first == " " || rest.first == "\t" else { return nil }
            return ListMark(ordered: false,
                            marker: "•",
                            content: String(rest).trimmingCharacters(in: .whitespaces),
                            indent: indent)
        }

        if first.isNumber {
            let digits = trimmed.prefix(while: { $0.isNumber })
            guard digits.count <= 9 else { return nil }
            let afterDigits = trimmed.dropFirst(digits.count)
            guard let punctuation = afterDigits.first, punctuation == "." || punctuation == ")" else { return nil }
            let rest = afterDigits.dropFirst()
            guard rest.isEmpty || rest.first == " " || rest.first == "\t" else { return nil }
            return ListMark(ordered: true,
                            marker: "\(digits)\(punctuation)",
                            content: String(rest).trimmingCharacters(in: .whitespaces),
                            indent: indent)
        }

        return nil
    }

    /// Splits a GitHub pipe-table row, honouring `\|` escapes and dropping the
    /// empty cells produced by leading/trailing pipes.
    static func splitRow(_ line: Substring) -> [String] {
        var cells: [String] = []
        var current = ""
        var escaped = false
        for character in line {
            if escaped {
                current.append(character)
                escaped = false
                continue
            }
            if character == "\\" { escaped = true; continue }
            if character == "|" {
                cells.append(current.trimmingCharacters(in: .whitespaces))
                current = ""
                continue
            }
            current.append(character)
        }
        if escaped { current.append("\\") }
        cells.append(current.trimmingCharacters(in: .whitespaces))

        if cells.count > 1, cells.first?.isEmpty == true { cells.removeFirst() }
        if cells.count > 1, cells.last?.isEmpty == true { cells.removeLast() }
        return cells
    }

    static func delimiterAlignments(_ line: Substring) -> [MDAlign]? {
        let cells = splitRow(line)
        guard !cells.isEmpty else { return nil }
        var alignments: [MDAlign] = []
        alignments.reserveCapacity(cells.count)
        for cell in cells {
            var body = Substring(cell)
            let left = body.first == ":"
            if left { body = body.dropFirst() }
            let right = body.last == ":"
            if right { body = body.dropLast() }
            guard !body.isEmpty, body.allSatisfy({ $0 == "-" }) else { return nil }
            switch (left, right) {
            case (true, true): alignments.append(.center)
            case (false, true): alignments.append(.trailing)
            default: alignments.append(.leading)
            }
        }
        return alignments
    }

    // MARK: Document scan

    static func parseBlocks(_ lines: [Substring], depth: Int) -> [MDBlock] {
        var blocks: [MDBlock] = []
        var index = 0

        func isTableStart(_ position: Int) -> Bool {
            guard position + 1 < lines.count else { return false }
            guard lines[position].contains("|"), lines[position + 1].contains("|") else { return false }
            return delimiterAlignments(lines[position + 1]) != nil
        }

        func startsBlock(_ position: Int) -> Bool {
            let line = lines[position]
            if fenceInfo(line) != nil { return true }
            if headingInfo(line) != nil { return true }
            if isRule(line) { return true }
            if trimLeading(line).first == ">" { return true }
            if listInfo(line) != nil { return true }
            return isTableStart(position)
        }

        while index < lines.count {
            let line = lines[index]

            if isBlank(line) { index += 1; continue }

            // Fenced code — must be first so nothing inside is interpreted.
            if let fence = fenceInfo(line) {
                index += 1
                var body: [Substring] = []
                var closed = false
                while index < lines.count {
                    if let candidate = fenceInfo(lines[index]),
                       candidate.marker == fence.marker,
                       candidate.length >= fence.length,
                       candidate.info.isEmpty {
                        closed = true
                        index += 1
                        break
                    }
                    body.append(lines[index])
                    index += 1
                }
                while let last = body.last, isBlank(last) { body.removeLast() }
                blocks.append(.code(language: fence.info,
                                    source: body.joined(separator: "\n"),
                                    closed: closed))
                continue
            }

            if let heading = headingInfo(line) {
                blocks.append(.heading(level: heading.level, text: heading.text))
                index += 1
                continue
            }

            // Rules must be tested before lists: "- - -" is a rule, not a bullet.
            if isRule(line) {
                blocks.append(.rule)
                index += 1
                continue
            }

            if trimLeading(line).first == ">" {
                var inner: [Substring] = []
                while index < lines.count {
                    let current = trimLeading(lines[index])
                    if current.first == ">" {
                        var body = current.dropFirst()
                        if body.first == " " { body = body.dropFirst() }
                        inner.append(body)
                        index += 1
                    } else if !inner.isEmpty, !isBlank(lines[index]), !startsBlock(index) {
                        inner.append(lines[index])          // lazy continuation
                        index += 1
                    } else {
                        break
                    }
                }
                if depth >= 5 {
                    blocks.append(.paragraph(inner.joined(separator: "\n")))
                } else {
                    blocks.append(.quote(blocks: parseBlocks(inner, depth: depth + 1)))
                }
                continue
            }

            if isTableStart(index) {
                let header = splitRow(lines[index])
                let alignments = delimiterAlignments(lines[index + 1]) ?? []
                index += 2
                var rows: [[String]] = []
                while index < lines.count {
                    let candidate = lines[index]
                    if isBlank(candidate) || !candidate.contains("|") { break }
                    if fenceInfo(candidate) != nil || headingInfo(candidate) != nil || isRule(candidate) { break }
                    rows.append(splitRow(candidate))
                    index += 1
                }
                blocks.append(.table(MDTable(header: header, alignments: alignments, rows: rows)))
                continue
            }

            if let first = listInfo(line) {
                var items: [MDListItem] = []
                let baseIndent = first.indent

                while index < lines.count {
                    if isBlank(lines[index]) {
                        var lookahead = index + 1
                        while lookahead < lines.count, isBlank(lines[lookahead]) { lookahead += 1 }
                        if lookahead < lines.count, !isRule(lines[lookahead]), listInfo(lines[lookahead]) != nil {
                            index = lookahead
                            continue
                        }
                        break
                    }
                    if isRule(lines[index]) { break }

                    if let mark = listInfo(lines[index]) {
                        // A bullet list followed by a numbered list at the same
                        // indent is two lists, not one.
                        if mark.ordered != first.ordered, mark.indent <= baseIndent, !items.isEmpty { break }
                        var content = mark.content
                        var checked: Bool? = nil
                        if content.hasPrefix("[ ] ") || content == "[ ]" {
                            checked = false
                            content = String(content.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                        } else if content.hasPrefix("[x] ") || content.hasPrefix("[X] ")
                                    || content == "[x]" || content == "[X]" {
                            checked = true
                            content = String(content.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                        }
                        let level = min(max(0, (mark.indent - baseIndent) / 2), 4)
                        items.append(MDListItem(marker: mark.marker,
                                                text: content,
                                                level: level,
                                                checked: checked))
                        index += 1
                    } else if !items.isEmpty, !startsBlock(index) {
                        let continuation = String(lines[index]).trimmingCharacters(in: .whitespaces)
                        items[items.count - 1].text += "\n" + continuation
                        index += 1
                    } else {
                        break
                    }
                }
                blocks.append(.list(items: items))
                continue
            }

            // Paragraph: everything up to a blank line or the start of a new block.
            var paragraph: [Substring] = []
            paragraph.append(line)
            index += 1
            while index < lines.count, !isBlank(lines[index]), !startsBlock(index) {
                paragraph.append(lines[index])
                index += 1
            }
            let joined = paragraph.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            if !joined.isEmpty { blocks.append(.paragraph(joined)) }
        }

        return blocks
    }
}

// MARK: - Inline spans

private enum MDInline {
    /// Parses **bold**, *italic*, `code`, ~~strike~~ and [links](url) into an
    /// `AttributedString`. Never throws: malformed markdown degrades to plain text.
    static func attributed(_ source: String) -> AttributedString {
        if source.isEmpty { return AttributedString() }

        var options = AttributedString.MarkdownParsingOptions()
        options.interpretedSyntax = .inlineOnlyPreservingWhitespace
        options.failurePolicy = .returnPartiallyParsedIfPossible
        options.allowsExtendedAttributes = true

        guard var parsed = try? AttributedString(markdown: source, options: options) else {
            return AttributedString(source)
        }

        // Collect first, mutate second — mutating while iterating `runs` is unsafe.
        var codeRanges: [Range<AttributedString.Index>] = []
        for run in parsed.runs {
            if let intent = run.inlinePresentationIntent, intent.contains(.code) {
                codeRanges.append(run.range)
            }
        }
        for range in codeRanges {
            parsed[range].font = Font.system(.body, design: .monospaced)
            parsed[range].backgroundColor = Color.primary.opacity(0.08)
        }
        return parsed
    }
}

// MARK: - Syntax highlighting

private enum CodeToken {
    case keyword, string, comment, number

    var color: Color {
        switch self {
        case .keyword: return .purple
        case .string: return .red
        case .number: return .blue
        case .comment: return .secondary
        }
    }
}

private struct SyntaxRules {
    var lineComments: [String] = []
    var blockCommentOpen: String? = nil
    var blockCommentClose: String? = nil
    var stringDelimiters: [Character] = []
    var tripleQuotes: Bool = false
    var escapes: Bool = true
    var identifierPrefixes: [Character] = []
    var prefixedWordsAreKeywords: Bool = false
    var keywords: Set<String> = []

    static func rules(for rawLanguage: String) -> SyntaxRules? {
        let key = rawLanguage
            .lowercased()
            .split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == ":" })
            .first
            .map(String.init) ?? ""

        switch key {
        case "swift":
            return SyntaxRules(lineComments: ["//"],
                               blockCommentOpen: "/*", blockCommentClose: "*/",
                               stringDelimiters: ["\""],
                               tripleQuotes: true,
                               identifierPrefixes: ["@", "#"],
                               prefixedWordsAreKeywords: true,
                               keywords: swiftKeywords)
        case "python", "py", "python3", "ipython":
            return SyntaxRules(lineComments: ["#"],
                               stringDelimiters: ["\"", "'"],
                               tripleQuotes: true,
                               keywords: pythonKeywords)
        case "javascript", "js", "jsx", "mjs", "cjs", "typescript", "ts", "tsx", "node":
            return SyntaxRules(lineComments: ["//"],
                               blockCommentOpen: "/*", blockCommentClose: "*/",
                               stringDelimiters: ["\"", "'", "`"],
                               keywords: javascriptKeywords)
        case "bash", "sh", "zsh", "shell", "console", "shell-session", "fish", "ksh":
            return SyntaxRules(lineComments: ["#"],
                               stringDelimiters: ["\"", "'"],
                               keywords: shellKeywords)
        case "json", "jsonc", "json5":
            return SyntaxRules(stringDelimiters: ["\""], keywords: ["true", "false", "null"])
        case "c", "h", "cpp", "c++", "cc", "hpp", "objc", "objective-c", "java",
             "go", "golang", "rust", "rs", "kotlin", "kt", "csharp", "cs", "scala":
            return SyntaxRules(lineComments: ["//"],
                               blockCommentOpen: "/*", blockCommentClose: "*/",
                               stringDelimiters: ["\"", "'", "`"],
                               keywords: cFamilyKeywords)
        default:
            return nil
        }
    }

    static let swiftKeywords: Set<String> = [
        "actor", "any", "as", "associatedtype", "async", "await", "break", "case", "catch", "class",
        "consuming", "continue", "convenience", "defer", "deinit", "didSet", "do", "dynamic", "else",
        "enum", "extension", "fallthrough", "false", "fileprivate", "final", "for", "func", "get",
        "guard", "if", "import", "in", "indirect", "infix", "init", "inout", "internal", "is", "lazy",
        "let", "mutating", "nil", "nonisolated", "nonmutating", "open", "operator", "override",
        "package", "postfix", "precedencegroup", "prefix", "private", "protocol", "public", "repeat",
        "required", "rethrows", "return", "self", "Self", "sending", "set", "some", "static", "struct",
        "subscript", "super", "switch", "throw", "throws", "true", "try", "typealias", "unowned", "var",
        "weak", "where", "while", "willSet", "yield",
    ]

    static let pythonKeywords: Set<String> = [
        "and", "as", "assert", "async", "await", "break", "case", "class", "continue", "def", "del",
        "elif", "else", "except", "False", "finally", "for", "from", "global", "if", "import", "in",
        "is", "lambda", "match", "None", "nonlocal", "not", "or", "pass", "raise", "return", "self",
        "True", "try", "while", "with", "yield",
    ]

    static let javascriptKeywords: Set<String> = [
        "abstract", "any", "as", "async", "await", "boolean", "break", "case", "catch", "class",
        "const", "continue", "declare", "default", "delete", "do", "else", "enum", "export", "extends",
        "false", "finally", "for", "from", "function", "get", "if", "implements", "import", "in",
        "infer", "instanceof", "interface", "keyof", "let", "namespace", "new", "null", "number",
        "of", "private", "protected", "public", "readonly", "return", "satisfies", "set", "static",
        "string", "super", "switch", "this", "throw", "true", "try", "type", "typeof", "undefined",
        "var", "void", "while", "yield",
    ]

    static let shellKeywords: Set<String> = [
        "alias", "break", "case", "cd", "continue", "declare", "do", "done", "elif", "else", "esac",
        "eval", "exec", "exit", "export", "false", "fi", "for", "function", "if", "in", "local",
        "printf", "read", "readonly", "return", "select", "set", "shift", "source", "then", "trap",
        "true", "unset", "until", "while",
    ]

    static let cFamilyKeywords: Set<String> = [
        "abstract", "auto", "bool", "break", "case", "catch", "chan", "char", "class", "companion",
        "const", "continue", "crate", "data", "default", "defer", "delete", "do", "double", "dyn",
        "else", "enum", "extends", "extern", "false", "final", "finally", "float", "fn", "for", "func",
        "fun", "go", "goto", "if", "impl", "implements", "import", "in", "inline", "int", "interface",
        "let", "long", "loop", "match", "mod", "move", "mut", "namespace", "new", "nil", "null",
        "nullptr", "object", "open", "operator", "override", "package", "private", "protected", "pub",
        "public", "range", "ref", "register", "return", "select", "self", "sealed", "short", "signed",
        "sizeof", "static", "struct", "super", "suspend", "switch", "template", "this", "throw",
        "throws", "trait", "true", "try", "type", "typedef", "typename", "union", "unsafe", "unsigned",
        "use", "using", "val", "var", "virtual", "void", "volatile", "when", "where", "while", "yield",
    ]
}

private enum CodeHighlighter {
    /// Single linear pass. Comments and strings are consumed whole, so a keyword
    /// that appears inside either of them is never highlighted as a keyword.
    static func highlight(_ code: String, language: String) -> AttributedString {
        guard !code.isEmpty else { return AttributedString() }
        guard let rules = SyntaxRules.rules(for: language) else { return AttributedString(code) }

        var output = AttributedString()
        var cursor = code.startIndex
        var plainStart = code.startIndex

        func matches(_ needle: String, at start: String.Index) -> Bool {
            var probe = start
            for character in needle {
                if probe == code.endIndex || code[probe] != character { return false }
                probe = code.index(after: probe)
            }
            return true
        }

        func advance(_ start: String.Index, by count: Int) -> String.Index {
            code.index(start, offsetBy: count, limitedBy: code.endIndex) ?? code.endIndex
        }

        func flushPlain(upTo end: String.Index) {
            guard plainStart < end else { return }
            output.append(AttributedString(String(code[plainStart..<end])))
        }

        func emit(_ range: Range<String.Index>, as token: CodeToken) {
            guard range.lowerBound < range.upperBound else { return }
            var piece = AttributedString(String(code[range]))
            piece.foregroundColor = token.color
            output.append(piece)
        }

        /// `#` only starts a comment at a shell/Python word boundary.
        func atWordBoundary(_ position: String.Index) -> Bool {
            guard position > code.startIndex else { return true }
            let previous = code[code.index(before: position)]
            return !(previous.isLetter || previous.isNumber || previous == "_"
                     || previous == "$" || previous == "{" || previous == "#")
        }

        func endOfString(from start: String.Index, quote: Character) -> String.Index {
            var triple = false
            if rules.tripleQuotes {
                let second = advance(start, by: 1)
                let third = advance(start, by: 2)
                if third < code.endIndex, code[second] == quote, code[third] == quote { triple = true }
            }
            var probe = advance(start, by: triple ? 3 : 1)
            while probe < code.endIndex {
                let character = code[probe]
                if rules.escapes, character == "\\" {
                    probe = advance(probe, by: 2)
                    continue
                }
                if character == quote {
                    if !triple { return code.index(after: probe) }
                    let second = advance(probe, by: 1)
                    let third = advance(probe, by: 2)
                    if third < code.endIndex, code[second] == quote, code[third] == quote {
                        return advance(probe, by: 3)
                    }
                } else if !triple, character == "\n" {
                    return probe                      // unterminated single-line string
                }
                probe = code.index(after: probe)
            }
            return code.endIndex
        }

        while cursor < code.endIndex {
            let character = code[cursor]

            // 1. Line comments.
            if let marker = rules.lineComments.first(where: { matches($0, at: cursor) }),
               marker.first != "#" || atWordBoundary(cursor) {
                flushPlain(upTo: cursor)
                var probe = cursor
                while probe < code.endIndex, code[probe] != "\n" { probe = code.index(after: probe) }
                emit(cursor..<probe, as: .comment)
                cursor = probe
                plainStart = probe
                continue
            }

            // 2. Block comments (an unterminated one runs to the end of the block).
            if let open = rules.blockCommentOpen, let close = rules.blockCommentClose,
               matches(open, at: cursor) {
                flushPlain(upTo: cursor)
                var probe = advance(cursor, by: open.count)
                var end = code.endIndex
                while probe < code.endIndex {
                    if matches(close, at: probe) {
                        end = advance(probe, by: close.count)
                        break
                    }
                    probe = code.index(after: probe)
                }
                emit(cursor..<end, as: .comment)
                cursor = end
                plainStart = end
                continue
            }

            // 3. Strings.
            if rules.stringDelimiters.contains(character) {
                flushPlain(upTo: cursor)
                let end = endOfString(from: cursor, quote: character)
                emit(cursor..<end, as: .string)
                cursor = max(end, code.index(after: cursor))
                plainStart = cursor
                continue
            }

            // 4. Identifiers and keywords (checked before numbers so `x2` stays one word).
            if character.isLetter || character == "_" || rules.identifierPrefixes.contains(character) {
                var probe = cursor
                let prefixed = rules.identifierPrefixes.contains(character)
                if prefixed { probe = code.index(after: probe) }
                while probe < code.endIndex,
                      code[probe].isLetter || code[probe].isNumber || code[probe] == "_" {
                    probe = code.index(after: probe)
                }
                let word = String(code[cursor..<probe])
                let isKeyword = rules.keywords.contains(word)
                    || (prefixed && rules.prefixedWordsAreKeywords && word.count > 1)
                if isKeyword {
                    flushPlain(upTo: cursor)
                    emit(cursor..<probe, as: .keyword)
                    plainStart = probe
                }
                cursor = probe > cursor ? probe : code.index(after: cursor)
                continue
            }

            // 5. Numbers.
            if character.isNumber {
                flushPlain(upTo: cursor)
                var probe = cursor
                while probe < code.endIndex {
                    let digit = code[probe]
                    if digit.isHexDigit || digit == "." || digit == "_"
                        || digit == "x" || digit == "X" || digit == "o" || digit == "O" {
                        probe = code.index(after: probe)
                    } else {
                        break
                    }
                }
                emit(cursor..<probe, as: .number)
                cursor = probe > cursor ? probe : code.index(after: cursor)
                plainStart = cursor
                continue
            }

            cursor = code.index(after: cursor)
        }

        flushPlain(upTo: code.endIndex)
        return output
    }
}

// MARK: - Memoisation

/// Small bounded caches so that re-evaluating `body` while a message streams
/// does not re-run the parser, the markdown inline parser, or the highlighter
/// for text that has not changed.
@MainActor
private enum MDCache {
    private struct HighlightKey: Hashable {
        var code: String
        var language: String
    }

    private static var blockEntries: [String: [MDBlock]] = [:]
    private static var blockOrder: [String] = []

    private static var inlineEntries: [String: AttributedString] = [:]
    private static var inlineOrder: [String] = []

    private static var highlightEntries: [HighlightKey: AttributedString] = [:]
    private static var highlightOrder: [HighlightKey] = []

    static func blocks(for text: String) -> [MDBlock] {
        if let hit = blockEntries[text] { return hit }
        let parsed = MD.parse(text)
        blockEntries[text] = parsed
        blockOrder.append(text)
        if blockOrder.count > 8 { blockEntries.removeValue(forKey: blockOrder.removeFirst()) }
        return parsed
    }

    static func inline(_ source: String) -> AttributedString {
        if let hit = inlineEntries[source] { return hit }
        let parsed = MDInline.attributed(source)
        inlineEntries[source] = parsed
        inlineOrder.append(source)
        if inlineOrder.count > 256 { inlineEntries.removeValue(forKey: inlineOrder.removeFirst()) }
        return parsed
    }

    static func highlighted(_ code: String, language: String) -> AttributedString {
        let key = HighlightKey(code: code, language: language)
        if let hit = highlightEntries[key] { return hit }
        let result = CodeHighlighter.highlight(code, language: language)
        highlightEntries[key] = result
        highlightOrder.append(key)
        if highlightOrder.count > 16 { highlightEntries.removeValue(forKey: highlightOrder.removeFirst()) }
        return result
    }
}

// MARK: - Block rendering

private struct MDBlockStack: View {
    let blocks: [MDBlock]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { entry in
                MDBlockView(block: entry.element)
            }
        }
    }
}

private struct MDBlockView: View {
    let block: MDBlock

    var body: some View {
        switch block {
        case .paragraph(let text):
            Text(MDCache.inline(text))
                .frame(maxWidth: .infinity, alignment: .leading)

        case .heading(let level, let text):
            Text(MDCache.inline(text))
                .font(Self.headingFont(level))
                .padding(.top, level <= 2 ? 4 : 1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .accessibilityAddTraits(.isHeader)

        case .code(let language, let source, _):
            MDCodeBlock(language: language, source: source)

        case .list(let items):
            MDListView(items: items)

        case .quote(let inner):
            MDQuoteView(blocks: inner)

        case .rule:
            Divider().padding(.vertical, 2)

        case .table(let table):
            MDTableView(table: table)
        }
    }

    private static func headingFont(_ level: Int) -> Font {
        switch level {
        case 1: return .system(.title2, design: .default, weight: .bold)
        case 2: return .system(.title3, design: .default, weight: .bold)
        case 3: return .system(.headline, design: .default, weight: .semibold)
        default: return .system(.subheadline, design: .default, weight: .semibold)
        }
    }
}

// MARK: - Code block

private struct MDCodeBlock: View {
    let language: String
    let source: String

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var copied = false

    private var displayLanguage: String {
        language.isEmpty ? "code" : language
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            ScrollView(.horizontal) {
                Text(MDCache.highlighted(source, language: language))
                    .font(.system(.callout, design: .monospaced))
                    .lineSpacing(2)
                    // Reset any inherited style (a code block nested in a
                    // blockquote must not pick up the quote's dimming).
                    .foregroundStyle(Color.primary)
                    .fixedSize(horizontal: true, vertical: false)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
            }
        }
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(displayLanguage) code block")
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text(displayLanguage)
                .font(.caption2.weight(.medium))
                .textCase(.uppercase)
                .foregroundStyle(.secondary)
                .lineLimit(1)

            Spacer(minLength: 8)

            Button(action: copy) {
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
                    .imageScale(.small)
                    .contentTransition(reduceMotion ? .identity : .symbolEffect(.replace))
                    .foregroundStyle(copied ? Color.green : Color.secondary)
            }
            .buttonStyle(.borderless)
            .help(copied ? "Copied to the clipboard" : "Copy this code block")
            .accessibilityLabel(copied ? "Code copied" : "Copy code")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
    }

    private func copy() {
        let board = NSPasteboard.general
        board.clearContents()
        board.setString(source, forType: .string)

        withAnimation(reduceMotion ? nil : .easeInOut(duration: 0.15)) { copied = true }
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            withAnimation(reduceMotion ? nil : .easeInOut(duration: 0.2)) { copied = false }
        }
    }
}

// MARK: - Lists

private struct MDListView: View {
    let items: [MDListItem]

    private var markerWidth: CGFloat {
        var widest = 1
        for item in items where item.checked == nil { widest = max(widest, item.marker.count) }
        return CGFloat(widest) * 8 + 6
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            ForEach(Array(items.enumerated()), id: \.offset) { entry in
                let item = entry.element
                HStack(alignment: .firstTextBaseline, spacing: 7) {
                    marker(for: item)
                        .frame(width: markerWidth, alignment: .trailing)
                    Text(MDCache.inline(item.text))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(.leading, CGFloat(item.level) * 18)
            }
        }
    }

    @ViewBuilder
    private func marker(for item: MDListItem) -> some View {
        if let checked = item.checked {
            Image(systemName: checked ? "checkmark.square.fill" : "square")
                .imageScale(.small)
                .foregroundStyle(checked ? Color.accentColor : Color.secondary)
                .accessibilityLabel(checked ? "Completed" : "Not completed")
        } else {
            Text(item.marker)
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
        }
    }
}

// MARK: - Blockquote

private struct MDQuoteView: View {
    let blocks: [MDBlock]

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            RoundedRectangle(cornerRadius: 1.5, style: .continuous)
                .fill(.tertiary)
                .frame(width: 3)
                .accessibilityHidden(true)

            // AnyView breaks the otherwise circular opaque-type recursion
            // MDBlockView -> MDQuoteView -> MDBlockStack -> MDBlockView.
            AnyView(MDBlockStack(blocks: blocks))
                .foregroundStyle(.secondary)
        }
        .fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Table

private struct MDTableView: View {
    let table: MDTable

    private var columns: [Int] { Array(0..<max(table.columnCount, 1)) }

    var body: some View {
        Grid(alignment: .topLeading, horizontalSpacing: 16, verticalSpacing: 6) {
            GridRow {
                ForEach(columns, id: \.self) { column in
                    Text(MDCache.inline(cell(table.header, column)))
                        .fontWeight(.semibold)
                        .multilineTextAlignment(table.alignment(column).text)
                        .gridColumnAlignment(table.alignment(column).horizontal)
                }
            }

            Divider().gridCellUnsizedAxes(.horizontal)

            ForEach(Array(table.rows.enumerated()), id: \.offset) { entry in
                GridRow {
                    ForEach(columns, id: \.self) { column in
                        Text(MDCache.inline(cell(entry.element, column)))
                            .multilineTextAlignment(table.alignment(column).text)
                    }
                }
            }
        }
        .padding(10)
        .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func cell(_ row: [String], _ column: Int) -> String {
        column >= 0 && column < row.count ? row[column] : ""
    }
}
