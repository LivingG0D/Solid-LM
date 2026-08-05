import Foundation

/// Walks an LM Studio-style models root — `<root>/<publisher>/<model>/…` — and turns it into `LocalModel`s.
///
/// Everything here is pure `Foundation`, has no shared mutable state, and is safe to call off the main actor.
enum ModelScanner {

    // MARK: - Tunables

    /// Measured effective memory bandwidth of this machine (base M5, 24 GB unified), in bytes/sec.
    /// Token rate is bandwidth-bound for local inference: est tok/s ≈ bandwidth / bytes-read-per-token.
    static let effectiveBandwidth: Double = 115e9

    /// Usable-memory thresholds in GiB for `FitClass`.
    static let fitsCeilingGiB: Double = 14
    static let tightCeilingGiB: Double = 19

    private static let bytesPerGiB: Double = 1_073_741_824

    // MARK: - Entry point

    /// Scan two levels deep: `<root>/<publisher>/<model>/`.
    /// Directories whose name starts with `.` or `_` are skipped at every level.
    /// Result is de-duplicated by `id` (first wins) and sorted by `id`.
    static func scan(root: URL) -> [LocalModel] {
        let fm = FileManager.default
        var found: [LocalModel] = []
        var seen = Set<String>()

        for publisherDir in childDirectories(of: root, fm: fm) {
            let publisher = publisherDir.lastPathComponent
            for modelDir in childDirectories(of: publisherDir, fm: fm) {
                for model in models(inModelDir: modelDir, publisher: publisher, fm: fm) {
                    if seen.insert(model.id).inserted { found.append(model) }
                }
            }
        }

        found.sort { $0.id < $1.id }
        return found
    }

    // MARK: - One model directory

    private static func models(inModelDir dir: URL, publisher: String, fm: FileManager) -> [LocalModel] {
        let entries = (try? fm.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        let dirName = dir.lastPathComponent
        var out: [LocalModel] = []

        // --- GGUF (llama.cpp) -------------------------------------------------
        let ggufCandidates = entries
            .filter { $0.pathExtension.lowercased() == "gguf" }
            .map { GGUFCandidate(fileName: $0.lastPathComponent, size: fileSize($0)) }

        // A publisher that ships `mtp-*.gguf` beside the model is handing us a
        // speculative-decoding draft head. Measured on Gemma4-12B Q4_K_M: 14.1 tok/s
        // without it, 25.5 with. Skipping it entirely — as this scanner used to —
        // left a 1.8x speedup sitting unused on disk.
        let draft = entries.first { $0.lastPathComponent.lowercased().hasPrefix("mtp-") }?.path

        for file in collapseGGUF(ggufCandidates) {
            let name = file.displayName
            let matchText = "\(publisher)/\(dirName)/\(name)"
            out.append(LocalModel(
                id: "\(publisher)/\(name)",
                name: name,
                publisher: publisher,
                engines: [.llama],
                path: dir.appendingPathComponent(file.primaryFileName).path,
                size: file.size,
                quant: quantToken(in: file.primaryFileName) ?? "unknown",
                arch: guessArch(from: matchText),
                estTokS: estimatedTokensPerSecond(size: file.size, matching: matchText),
                fit: fitClass(size: file.size),
                draftPath: draft
            ))
        }

        // --- MLX / safetensors ------------------------------------------------
        let safetensors = entries.filter { $0.pathExtension.lowercased() == "safetensors" }
        if !safetensors.isEmpty, let cfg = readMLXConfig(in: dir, fm: fm) {
            let size = safetensors.reduce(Int64(0)) { $0 + fileSize($1) }
            let matchText = "\(publisher)/\(dirName)"

            var engines: [EngineKind] = [.mlx]
            let quant: String
            if let bits = cfg.bits {
                quant = "\(bits)bit"
            } else if cfg.hasQuantization {
                quant = quantToken(inDirectoryName: dirName) ?? "quantized"
            } else {
                // No quantization block at all: full-precision weights, so vLLM is a candidate too.
                quant = quantToken(inDirectoryName: dirName) ?? "BF16"
                engines.append(.vllm)
            }

            out.append(LocalModel(
                id: "\(publisher)/\(dirName)",
                name: dirName,
                publisher: publisher,
                engines: engines,
                // MLX loads a DIRECTORY, not a single weight file.
                path: dir.path,
                size: size,
                quant: quant,
                arch: cfg.arch ?? guessArch(from: matchText),
                estTokS: estimatedTokensPerSecond(size: size, matching: matchText),
                fit: fitClass(size: size)
            ))
        }

        return out
    }

    // MARK: - GGUF collapsing (pure, no filesystem)

    /// One `.gguf` file on disk, as a name + byte count.
    struct GGUFCandidate: Hashable, Sendable {
        var fileName: String
        var size: Int64

        init(fileName: String, size: Int64) {
            self.fileName = fileName
            self.size = size
        }
    }

    /// One *logical* GGUF model: a single file, or a whole shard set collapsed into its first part.
    struct GGUFModelFile: Hashable, Sendable {
        var displayName: String       // shard suffix stripped, no ".gguf"
        var primaryFileName: String   // the file to hand to `llama-server -m`
        var size: Int64               // sum over every shard
    }

    /// Vision projectors and draft/MTP heads are not standalone models.
    static func isIgnorableGGUF(fileName: String) -> Bool {
        let lower = fileName.lowercased()
        if lower.contains("mmproj") { return true }
        if lower.hasPrefix("mtp-") { return true }
        return false
    }

    /// Drop non-models, then fold `NAME-00001-of-00003.gguf` … into a single entry named `NAME`
    /// whose size is the sum of every shard. A shard set with no `-00001-` part is incomplete and dropped.
    static func collapseGGUF(_ candidates: [GGUFCandidate]) -> [GGUFModelFile] {
        var singles: [GGUFModelFile] = []
        var groups: [String: (base: String, primary: String?, size: Int64)] = [:]
        var groupOrder: [String] = []

        for candidate in candidates {
            guard !isIgnorableGGUF(fileName: candidate.fileName) else { continue }
            let size = max(0, candidate.size)

            if let shard = shardComponents(fileName: candidate.fileName) {
                let key = "\(shard.base)|of|\(shard.total)"
                if groups[key] == nil {
                    groups[key] = (shard.base, nil, 0)
                    groupOrder.append(key)
                }
                groups[key]?.size += size
                if shard.index == 1 { groups[key]?.primary = candidate.fileName }
            } else {
                singles.append(GGUFModelFile(
                    displayName: ggufStem(candidate.fileName),
                    primaryFileName: candidate.fileName,
                    size: size
                ))
            }
        }

        var out = singles
        for key in groupOrder {
            guard let group = groups[key], let primary = group.primary else { continue }
            out.append(GGUFModelFile(displayName: group.base, primaryFileName: primary, size: group.size))
        }

        out.sort {
            $0.displayName == $1.displayName
                ? $0.primaryFileName < $1.primaryFileName
                : $0.displayName < $1.displayName
        }
        return out
    }

    /// `"Model-Q4_K_M-00001-of-00003.gguf"` → `(base: "Model-Q4_K_M", index: 1, total: 3)`.
    /// Returns nil for a plain single-file model.
    static func shardComponents(fileName: String) -> (base: String, index: Int, total: Int)? {
        let stem = ggufStem(fileName)
        guard let range = stem.range(
            of: #"-[0-9]{5}-of-[0-9]{5}$"#,
            options: [.regularExpression, .caseInsensitive]
        ) else { return nil }

        let numbers = stem[range].split(separator: "-").compactMap { Int($0) }
        guard numbers.count == 2, numbers[0] >= 1, numbers[1] >= 1 else { return nil }

        let base = String(stem[stem.startIndex..<range.lowerBound])
        guard !base.isEmpty else { return nil }
        return (base, numbers[0], numbers[1])
    }

    /// Strips a trailing `.gguf` (case-insensitively) without touching dots inside the name.
    static func ggufStem(_ fileName: String) -> String {
        fileName.lowercased().hasSuffix(".gguf") ? String(fileName.dropLast(5)) : fileName
    }

    // MARK: - Quantisation

    /// Ordered most-specific-first. Each is anchored so a token can never be matched mid-word
    /// (this is what keeps `IQ4_XS` from degrading to `Q4_...`, and `Q3_K_XL` from truncating to `Q3_K`).
    private static let quantPatterns: [String] = [
        #"(?<![A-Za-z0-9])MXFP[0-9]+(?:_MOE)?(?![A-Za-z0-9])"#,   // MXFP4, MXFP8
        #"(?<![A-Za-z0-9])IQ[0-9]+_[A-Z]{1,3}(?![A-Za-z0-9])"#,   // IQ4_XS, IQ4_NL, IQ3_XXS
        #"(?<![A-Za-z0-9])IQ[0-9]+(?![A-Za-z0-9])"#,              // IQ4
        #"(?<![A-Za-z0-9])Q[0-9]+_K_[A-Z]{1,3}(?![A-Za-z0-9])"#,  // Q4_K_M, Q3_K_XL, Q6_K_L
        #"(?<![A-Za-z0-9])Q[0-9]+_K(?![A-Za-z0-9])"#,             // Q6_K
        #"(?<![A-Za-z0-9])Q[0-9]+_[0-9]+(?![A-Za-z0-9])"#,        // Q4_0, Q2_0, Q8_0, Q5_1
        #"(?<![A-Za-z0-9])BF16(?![A-Za-z0-9])"#,
        #"(?<![A-Za-z0-9])FP16(?![A-Za-z0-9])"#,
        #"(?<![A-Za-z0-9])F16(?![A-Za-z0-9])"#,
        #"(?<![A-Za-z0-9])F32(?![A-Za-z0-9])"#,
    ]

    /// Pulls a quant label out of a file (or directory) name. Uppercased. Nil when nothing matches.
    /// Safe to hand a full filename — the patterns ignore the extension.
    static func quantToken(in text: String) -> String? {
        for pattern in quantPatterns {
            if let range = text.range(of: pattern, options: [.regularExpression, .caseInsensitive]) {
                return text[range].uppercased()
            }
        }
        return nil
    }

    /// Directory names carry looser labels than GGUF filenames do: `-4bit`, `-8Bit`, `-3.7bpw`.
    static func quantToken(inDirectoryName name: String) -> String? {
        if let token = quantToken(in: name) { return token }

        if let range = name.range(
            of: #"(?<![A-Za-z0-9])[0-9]+(?:\.[0-9]+)?[-_ ]?BPW(?![A-Za-z0-9])"#,
            options: [.regularExpression, .caseInsensitive]
        ) {
            return compactLabel(String(name[range]))
        }

        if let range = name.range(
            of: #"(?<![A-Za-z0-9])[0-9]+[-_ ]?BIT(?![A-Za-z0-9])"#,
            options: [.regularExpression, .caseInsensitive]
        ) {
            return compactLabel(String(name[range]))
        }

        return nil
    }

    private static func compactLabel(_ s: String) -> String {
        s.lowercased().filter { !" -_".contains($0) }
    }

    // MARK: - Architecture

    private static let archHints: [(pattern: String, arch: String)] = [
        (#"(?<![A-Za-z])gpt.?oss"#, "gpt-oss"),
        (#"(?<![A-Za-z])qwen"#, "qwen"),
        (#"(?<![A-Za-z])gemma"#, "gemma"),
        (#"(?<![A-Za-z])mixtral"#, "mixtral"),
        (#"(?<![A-Za-z])mistral"#, "mistral"),
        (#"(?<![A-Za-z])llama"#, "llama"),
        (#"(?<![A-Za-z])deepseek"#, "deepseek"),
        (#"(?<![A-Za-z])phi"#, "phi"),
        (#"(?<![A-Za-z])granite"#, "granite"),
        (#"(?<![A-Za-z])smollm"#, "smollm"),
        (#"(?<![A-Za-z])glm"#, "glm"),
        (#"(?<![A-Za-z])falcon"#, "falcon"),
        (#"(?<![A-Za-z])olmo"#, "olmo"),
        (#"(?<![A-Za-z])minicpm"#, "minicpm"),
    ]

    /// Best-effort family name. GGUF headers are not parsed, so this is a name heuristic only;
    /// MLX models prefer `model_type` straight out of `config.json`.
    static func guessArch(from text: String) -> String {
        for hint in archHints {
            if text.range(of: hint.pattern, options: [.regularExpression, .caseInsensitive]) != nil {
                return hint.arch
            }
        }
        return "unknown"
    }

    // MARK: - Speed & fit

    /// Fraction of the weights actually read per token. Dense models read everything;
    /// mixture-of-experts models only touch their active experts.
    static func readFraction(matching text: String) -> Double {
        if text.range(of: #"gpt.?oss"#, options: [.regularExpression, .caseInsensitive]) != nil { return 0.252 }
        if text.range(of: #"A4B"#, options: [.regularExpression, .caseInsensitive]) != nil { return 0.166 }
        if text.range(of: #"A3B"#, options: [.regularExpression, .caseInsensitive]) != nil { return 0.14 }
        return 1.0
    }

    /// Bandwidth-bound estimate, rounded to one decimal.
    static func estimatedTokensPerSecond(size: Int64, matching text: String) -> Double {
        guard size > 0 else { return 0 }
        let bytesPerToken = Double(size) * readFraction(matching: text)
        guard bytesPerToken > 0 else { return 0 }
        return ((effectiveBandwidth / bytesPerToken) * 10).rounded() / 10
    }

    static func fitClass(size: Int64) -> FitClass {
        let giB = Double(size) / bytesPerGiB
        if giB <= fitsCeilingGiB { return .fits }
        if giB <= tightCeilingGiB { return .tight }
        return .tooBig
    }

    // MARK: - config.json

    private struct MLXConfig {
        var arch: String?
        var bits: Int?
        var hasQuantization: Bool
    }

    /// Returns nil when the directory has no `config.json` (i.e. it is not an MLX/safetensors model).
    private static func readMLXConfig(in dir: URL, fm: FileManager) -> MLXConfig? {
        let url = dir.appendingPathComponent("config.json")
        guard fm.fileExists(atPath: url.path) else { return nil }

        var config = MLXConfig(arch: nil, bits: nil, hasQuantization: false)
        guard let data = try? Data(contentsOf: url),
              let raw = try? JSONSerialization.jsonObject(with: data),
              let object = raw as? [String: Any]
        else { return config }

        if let modelType = object["model_type"] as? String, !modelType.isEmpty {
            config.arch = modelType
        } else if let text = object["text_config"] as? [String: Any],
                  let modelType = text["model_type"] as? String, !modelType.isEmpty {
            // Multimodal configs nest the language model's type.
            config.arch = modelType
        }

        if let quantization = object["quantization"] as? [String: Any] {
            config.hasQuantization = true
            // Per-layer overrides also carry "bits"; the top-level value is the one that describes the model.
            if let bits = quantization["bits"] as? Int { config.bits = bits }
        }

        return config
    }

    // MARK: - Filesystem helpers

    private static func childDirectories(of url: URL, fm: FileManager) -> [URL] {
        let items = (try? fm.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        return items.filter { child in
            let name = child.lastPathComponent
            guard !name.hasPrefix("."), !name.hasPrefix("_") else { return false }
            return isDirectory(child)
        }
    }

    private static func isDirectory(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
    }

    private static func fileSize(_ url: URL) -> Int64 {
        let values = try? url.resourceValues(forKeys: [.fileSizeKey])
        return Int64(values?.fileSize ?? 0)
    }

    // MARK: - Self check

    /// Pure string-level regression tests for the two parts that are easy to get subtly wrong:
    /// the quant regex and shard collapsing. Returns human-readable failures; `[]` means all good.
    static func selfCheck() -> [String] {
        var failures: [String] = []

        // --- quant extraction ---
        let quantCases: [(input: String, expected: String?)] = [
            ("Qwen3-4B-IQ4_XS.gguf", "IQ4_XS"),
            ("OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf", "IQ4_NL"),
            ("Qwen3.6-12B-IQ-IQ4_XS.gguf", "IQ4_XS"),
            ("Qwen3-4B-Q4_K_M.gguf", "Q4_K_M"),
            ("Qwen3-4B-Q3_K_XL.gguf", "Q3_K_XL"),          // two-letter suffix must survive
            ("Qwythos-9B-Claude-Mythos-5-1M-MTP-Q6_K.gguf", "Q6_K"),
            ("Qwen3-4B-Q4_0.gguf", "Q4_0"),
            ("Ternary-Bonsai-27B-Abliterated-LowDeg-Q2_0.gguf", "Q2_0"),
            ("gpt-oss-20b-MXFP4.gguf", "MXFP4"),
            ("mmproj-Qwythos-9B-v2-BF16.gguf", "BF16"),
            ("mmproj-Gemma-4-E4B-Aggressive-f16.gguf", "F16"),
            ("mmproj-F32.gguf", "F32"),
            ("Model-Q4_K_M-00001-of-00003.gguf", "Q4_K_M"),
            ("Qwen3.6-27B-Heretic.gguf", nil),
            ("Qwen2.5-0.5B-Instruct.gguf", nil),
        ]
        for testCase in quantCases {
            let got = quantToken(in: testCase.input)
            if got != testCase.expected {
                failures.append("quantToken(\"\(testCase.input)\") = \(got.map { "\"\($0)\"" } ?? "nil"), expected \(testCase.expected.map { "\"\($0)\"" } ?? "nil")")
            }
        }

        // --- directory-name quant fallback ---
        let dirCases: [(input: String, expected: String?)] = [
            ("Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-MLX-mxfp4", "MXFP4"),
            ("Huihui-Qwythos-9B-Claude-Mythos-5-1M-abliterated-mlx-8Bit", "8bit"),
            ("gemma-4-26B-it-uncensored-abliterix-MLX-4bit-mixed_4_6", "4bit"),
            ("Qwen3.6-27B-Claude-Opus-Distill-v2-abliterated-OptiQ-3.7bpw-mlx", "3.7bpw"),
            ("Qwen3.6-27B-AEON-Plain", nil),
        ]
        for testCase in dirCases {
            let got = quantToken(inDirectoryName: testCase.input)
            if got != testCase.expected {
                failures.append("quantToken(inDirectoryName: \"\(testCase.input)\") = \(got.map { "\"\($0)\"" } ?? "nil"), expected \(testCase.expected.map { "\"\($0)\"" } ?? "nil")")
            }
        }

        // --- shard parsing ---
        if let shard = shardComponents(fileName: "Big-Model-Q4_K_M-00001-of-00003.gguf") {
            if shard.base != "Big-Model-Q4_K_M" || shard.index != 1 || shard.total != 3 {
                failures.append("shardComponents(part 1) = (\(shard.base), \(shard.index), \(shard.total)), expected (Big-Model-Q4_K_M, 1, 3)")
            }
        } else {
            failures.append("shardComponents(\"Big-Model-Q4_K_M-00001-of-00003.gguf\") = nil, expected a shard")
        }
        if shardComponents(fileName: "plain-model-Q4_K_M.gguf") != nil {
            failures.append("shardComponents(\"plain-model-Q4_K_M.gguf\") matched, expected nil")
        }
        if shardComponents(fileName: "model-0001-of-0003.gguf") != nil {
            failures.append("shardComponents(4-digit shard) matched, expected nil (llama.cpp always writes 5 digits)")
        }

        // --- shard collapsing + filtering ---
        let collapsed = collapseGGUF([
            GGUFCandidate(fileName: "A-Q4_K_M-00002-of-00003.gguf", size: 20),
            GGUFCandidate(fileName: "A-Q4_K_M-00001-of-00003.gguf", size: 10),
            GGUFCandidate(fileName: "A-Q4_K_M-00003-of-00003.gguf", size: 30),
            GGUFCandidate(fileName: "B-Q6_K.gguf", size: 5),
            GGUFCandidate(fileName: "mmproj-A-F16.gguf", size: 7),
            GGUFCandidate(fileName: "mtp-A.gguf", size: 9),
            GGUFCandidate(fileName: "C-00002-of-00002.gguf", size: 11),   // incomplete: no part 1
        ])
        let expected: [GGUFModelFile] = [
            GGUFModelFile(displayName: "A-Q4_K_M", primaryFileName: "A-Q4_K_M-00001-of-00003.gguf", size: 60),
            GGUFModelFile(displayName: "B-Q6_K", primaryFileName: "B-Q6_K.gguf", size: 5),
        ]
        if collapsed != expected {
            let describe = collapsed.map { "\($0.displayName)|\($0.primaryFileName)|\($0.size)" }.joined(separator: ", ")
            failures.append("collapseGGUF = [\(describe)], expected A-Q4_K_M (60 bytes, part 1) and B-Q6_K (5 bytes) only")
        }

        // --- fit + speed ---
        if fitClass(size: Int64(13.0 * bytesPerGiB)) != .fits { failures.append("fitClass(13 GiB) != .fits") }
        if fitClass(size: Int64(17.0 * bytesPerGiB)) != .tight { failures.append("fitClass(17 GiB) != .tight") }
        if fitClass(size: Int64(22.0 * bytesPerGiB)) != .tooBig { failures.append("fitClass(22 GiB) != .tooBig") }
        if readFraction(matching: "openai/gpt-oss-20b") != 0.252 { failures.append("readFraction(gpt-oss) != 0.252") }
        if readFraction(matching: "google/gemma-4-26B-A4B-it") != 0.166 { failures.append("readFraction(A4B) != 0.166") }
        if readFraction(matching: "Qwen/Qwen3-30B-A3B") != 0.14 { failures.append("readFraction(A3B) != 0.14") }
        if readFraction(matching: "Qwen/Qwen3-4B") != 1.0 { failures.append("readFraction(dense) != 1.0") }

        return failures
    }
}
