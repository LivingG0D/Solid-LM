import Foundation

/// Walks the models roots and turns whatever looks like a diffusion checkpoint or an
/// ESRGAN network into `ImageModel`s.
///
/// Image models are not laid out like LM Studio's `<publisher>/<model>/` tree — people drop
/// a `.safetensors` wherever it landed — so this walks a bounded slice of the tree instead of
/// assuming a shape. Everything here is pure `Foundation` with no shared mutable state, so
/// `scan` is safe to call from a detached Task.
enum ImageScanner {

    // MARK: - Tunables

    /// Directory levels visited per root, counting the root itself. Three covers
    /// `<root>/_image-models/<family>/model.safetensors` and stops a symlinked or
    /// pathological tree from turning a rescan into a full-disk crawl.
    static let maxDepth = 3

    /// Below this a file is a config, a preview, or a half-written stub — never a checkpoint.
    /// The smallest thing we ship against is an 18 MB ESRGAN net.
    static let minimumSize: Int64 = 4 * 1024 * 1024

    private static let bytesPerGiB: Int64 = 1_073_741_824

    // MARK: - Entry point

    /// Scan each root recursively, at most `maxDepth` levels deep.
    /// De-duplicated by path (first root wins), diffusion models first, then upscalers.
    static func scan(roots: [URL]) -> [ImageModel] {
        let fm = FileManager.default
        var found: [ImageModel] = []
        var seen = Set<String>()

        for root in roots {
            collect(in: root, levelsRemaining: maxDepth, fm: fm, seen: &seen, into: &found)
        }

        found.sort { a, b in
            if a.kind != b.kind { return a.kind == .diffusion }
            let order = a.name.caseInsensitiveCompare(b.name)
            return order == .orderedSame ? a.id < b.id : order == .orderedAscending
        }
        return found
    }

    // MARK: - Walking

    private static func collect(in directory: URL,
                                levelsRemaining: Int,
                                fm: FileManager,
                                seen: inout Set<String>,
                                into found: inout [ImageModel]) {
        guard levelsRemaining > 0 else { return }

        let entries = (try? fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        for entry in entries {
            let name = entry.lastPathComponent
            // Only dot-directories are skipped. Image models deliberately live in
            // "_image-models", which the LLM scanner skips and this one must not.
            guard !name.hasPrefix(".") else { continue }

            if isDirectory(entry) {
                collect(in: entry, levelsRemaining: levelsRemaining - 1, fm: fm, seen: &seen, into: &found)
            } else if let model = model(at: entry, fileName: name) {
                if seen.insert(model.id).inserted { found.append(model) }
            }
        }
    }

    private static func model(at url: URL, fileName: String) -> ImageModel? {
        guard let kind = kind(forFileName: fileName), !isComponentFile(fileName) else { return nil }

        let size = fileSize(url)
        guard size >= minimumSize else { return nil }

        // The models root is shared with the LLM side, so a plain "*.gguf" sweep also
        // catches every language model on disk. Loading one into sd-server wastes
        // minutes and then fails with "get sd version from file failed".
        //
        // Discriminator: llama.cpp requires `general.architecture` in the GGUF header
        // and every LLM declares it ("gemma4", "qwen35", …); the diffusion GGUFs
        // sd.cpp reads have no such key. Verified against both on this machine.
        if url.pathExtension.lowercased() == "gguf", declaresLLMArchitecture(url) { return nil }

        // Same problem for the MLX side: those language models ship as
        // `model-00001-of-00002.safetensors` beside a transformers `config.json`.
        // sd.cpp wants a single-file checkpoint, so anything in that layout is not ours.
        if url.pathExtension.lowercased() == "safetensors", isTransformersShard(url, fileName: fileName) {
            return nil
        }

        let path = url.standardizedFileURL.path
        return ImageModel(
            id: path,
            name: displayName(forFileName: fileName),
            kind: kind,
            // An ESRGAN net has no diffusion architecture; guessing one off its size
            // would put "SD 1.5" next to an upscaler for no reason.
            arch: kind == .upscaler ? .unknown : arch(forFileName: fileName, size: size),
            path: path,
            size: size,
            quant: quant(forFileName: fileName)
        )
    }

    // MARK: - Transformers / MLX layout

    /// True when this `.safetensors` belongs to a HuggingFace-style language model
    /// rather than being a standalone diffusion checkpoint.
    ///
    /// Two signals, either is enough:
    ///  - the `model-00001-of-00002.safetensors` shard convention, which transformers
    ///    uses and single-file image checkpoints never do
    ///  - a sibling `config.json` declaring `model_type`, which is how MLX and
    ///    transformers mark a model directory
    static func isTransformersShard(_ url: URL, fileName: String) -> Bool {
        if matchesShardConvention(fileName) { return true }
        let config = url.deletingLastPathComponent().appending(path: "config.json")
        guard let data = try? Data(contentsOf: config), data.count < 1 << 20,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        return json["model_type"] != nil || json["architectures"] != nil
    }

    /// `model-00001-of-00002.safetensors` and friends.
    static func matchesShardConvention(_ fileName: String) -> Bool {
        fileName.range(of: "^model-\\d{5}-of-\\d{5}\\.safetensors$",
                       options: [.regularExpression, .caseInsensitive]) != nil
    }

    // MARK: - GGUF header

    /// True when this GGUF carries `general.architecture`, i.e. it is a language
    /// model rather than something sd.cpp can diffuse with.
    ///
    /// Reads only the key-value header — a few KB off the front — never the tensors,
    /// so scanning a folder of 15 GB models stays instant.
    static func declaresLLMArchitecture(_ url: URL) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }
        // 1 MB is far beyond any real header; a diffusion GGUF simply runs out of keys first.
        guard let head = try? handle.read(upToCount: 1 << 20) else { return false }
        return headerDeclaresArchitecture(head)
    }

    /// Split out from the file handle so it can be tested on bytes.
    static func headerDeclaresArchitecture(_ data: Data) -> Bool {
        var cursor = 0
        func take(_ n: Int) -> Data? {
            guard n >= 0, cursor + n <= data.count else { return nil }
            defer { cursor += n }
            return data.subdata(in: cursor..<(cursor + n))
        }
        func u32() -> UInt32? { take(4).map { $0.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) } } }
        func u64() -> UInt64? { take(8).map { $0.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) } } }
        func str() -> String? {
            guard let n = u64(), n < 1 << 20, let bytes = take(Int(n)) else { return nil }
            return String(decoding: bytes, as: UTF8.self)
        }

        guard take(4) == Data("GGUF".utf8), u32() != nil,        // magic, version
              u64() != nil, let kvCount = u64(), kvCount < 100_000  // tensor count, kv count
        else { return false }

        /// Sizes of the fixed-width GGUF value types, by type id.
        let widths: [UInt32: Int] = [0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8]

        func skipValue(_ type: UInt32, depth: Int = 0) -> Bool {
            if type == 8 { return str() != nil }                  // string
            if type == 9 {                                        // array
                guard depth < 4, let elem = u32(), let n = u64(), n < 1 << 24 else { return false }
                for _ in 0..<n where !skipValue(elem, depth: depth + 1) { return false }
                return true
            }
            guard let w = widths[type] else { return false }
            return take(w) != nil
        }

        for _ in 0..<kvCount {
            guard let key = str(), let type = u32() else { return false }
            if key == "general.architecture" { return true }
            guard skipValue(type) else { return false }
        }
        return false
    }

    // MARK: - Classification (pure)

    /// Everything else on disk — json, yaml, png previews, txt — is noise.
    private static let modelExtensions: Set<String> = ["gguf", "safetensors", "ckpt", "pth"]

    /// Nil means "not a model file at all".
    static func kind(forFileName fileName: String) -> ImageModelKind? {
        let ext = (fileName as NSString).pathExtension.lowercased()
        guard modelExtensions.contains(ext) else { return nil }

        let lower = fileName.lowercased()
        // `.pth` is only ever an ESRGAN net in this world; the name test also catches
        // upscalers that ship as `.safetensors`.
        if ext == "pth" || lower.contains("esrgan") || lower.contains("upscal") { return .upscaler }
        return .diffusion
    }

    /// Split-file model *parts*. A standalone VAE or text encoder is not something the user
    /// can generate from — listing it just invites a confusing failure at load time.
    /// `.part`/`.partial` catch downloads still in flight.
    private static let componentTokens = [
        "vae", "clip_l", "clip_g", "t5xxl", "text_encoder",
        "lora", "-tensordata", ".part", ".partial",
    ]

    static func isComponentFile(_ fileName: String) -> Bool {
        // Vision projectors and draft heads are LLM components by the same argument, and a root
        // shared with the LLM side is full of them.
        if ModelScanner.isIgnorableGGUF(fileName: fileName) { return true }
        let lower = fileName.lowercased()
        return componentTokens.contains { lower.contains($0) }
    }

    /// Ordered: the first match wins, so SDXL is tested before SD 1.5. That ordering is what
    /// keeps `dreamshaper-xl` out of the SD 1.5 bucket, and the negative lookahead keeps plain
    /// `dreamshaper` in it. Getting Juggernaut / Pony / RealVis right matters — they are the
    /// popular SDXL checkpoints, and a wrong arch hands the user 512×512 defaults on a
    /// 1024-native model, which just looks broken.
    private static let archPatterns: [(pattern: String, arch: ImageArch)] = [
        (#"flux"#, .flux),
        (#"qwen.?image"#, .qwen),
        (#"sd_?3|stable.?diffusion.?3"#, .sd3),
        (#"sdxl|xl.?base|xl.?refiner|juggernaut|realvis|dreamshaper.?xl|pony"#, .sdxl),
        (#"sd.?1\.?5|v1-5|sd15|dreamshaper(?!.?xl)|realistic.?vision"#, .sd15),
    ]

    /// Filename first, size only as a tiebreak — the families barely overlap in bytes.
    static func arch(forFileName fileName: String, size: Int64) -> ImageArch {
        for hint in archPatterns
        where fileName.range(of: hint.pattern, options: [.regularExpression, .caseInsensitive]) != nil {
            return hint.arch
        }

        if size < 3 * bytesPerGiB { return .sd15 }
        if size < 10 * bytesPerGiB { return .sdxl }
        return .flux
    }

    /// Same quant tokens the LLM scanner recognises (`Q4_0`, `Q4_K_M`, `F16`, `BF16`, …).
    /// Empty when the name carries none, which is the normal case for a full-precision
    /// `.safetensors` checkpoint.
    static func quant(forFileName fileName: String) -> String {
        ModelScanner.quantToken(in: fileName) ?? ""
    }

    static func displayName(forFileName fileName: String) -> String {
        (fileName as NSString).deletingPathExtension
    }

    // MARK: - Filesystem helpers

    private static func isDirectory(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
    }

    private static func fileSize(_ url: URL) -> Int64 {
        let values = try? url.resourceValues(forKeys: [.fileSizeKey])
        return Int64(values?.fileSize ?? 0)
    }

    // MARK: - Self check

    /// Pure string-level regression tests over the classification helpers — the part that is
    /// easy to break silently, since a misclassified checkpoint only shows up as a bad default
    /// resolution or a load failure much later. Returns human-readable failures; `[]` means good.
    static func selfCheck() -> [String] {
        var failures: [String] = []

        // --- GGUF header discrimination ---
        // Hand-build the two shapes rather than needing a model on disk.
        func gguf(keys: [(String, String)]) -> Data {
            var d = Data("GGUF".utf8)
            d.append(contentsOf: withUnsafeBytes(of: UInt32(3).littleEndian, Array.init))
            d.append(contentsOf: withUnsafeBytes(of: UInt64(0).littleEndian, Array.init))          // tensors
            d.append(contentsOf: withUnsafeBytes(of: UInt64(keys.count).littleEndian, Array.init)) // kv count
            for (k, v) in keys {
                d.append(contentsOf: withUnsafeBytes(of: UInt64(k.utf8.count).littleEndian, Array.init))
                d.append(Data(k.utf8))
                d.append(contentsOf: withUnsafeBytes(of: UInt32(8).littleEndian, Array.init))      // string type
                d.append(contentsOf: withUnsafeBytes(of: UInt64(v.utf8.count).littleEndian, Array.init))
                d.append(Data(v.utf8))
            }
            return d
        }
        if !headerDeclaresArchitecture(gguf(keys: [("general.name", "x"), ("general.architecture", "gemma4")])) {
            failures.append("gguf: an LLM header with general.architecture was not detected")
        }
        if headerDeclaresArchitecture(gguf(keys: [("general.name", "sd"), ("general.quantization_version", "2")])) {
            failures.append("gguf: a diffusion header was wrongly flagged as an LLM")
        }
        if headerDeclaresArchitecture(Data("not a gguf at all".utf8)) {
            failures.append("gguf: garbage input was treated as an LLM header")
        }
        if headerDeclaresArchitecture(Data()) {
            failures.append("gguf: empty input was treated as an LLM header")
        }

        // --- transformers shard convention ---
        for name in ["model-00001-of-00002.safetensors", "model-00003-of-00012.safetensors"]
        where !matchesShardConvention(name) {
            failures.append("shard: \"\(name)\" should be recognised as a transformers shard")
        }
        for name in ["juggernautXL_v9.safetensors", "model.safetensors", "sd15.safetensors",
                     "model-0001-of-0002.safetensors"]
        where matchesShardConvention(name) {
            failures.append("shard: \"\(name)\" is not a transformers shard but was treated as one")
        }

        // --- extension routing ---
        let kindCases: [(input: String, expected: ImageModelKind?)] = [
            ("sd15-Q4_0.gguf", .diffusion),
            ("juggernautXL_v9.safetensors", .diffusion),
            ("v1-5-pruned-emaonly.ckpt", .diffusion),
            ("flux1-dev.GGUF", .diffusion),                        // extension test is case-insensitive
            ("RealESRGAN_x4plus_anime_6B.pth", .upscaler),
            ("4x-UltraSharp.pth", .upscaler),                      // .pth alone is enough
            ("ESRGAN_x4.safetensors", .upscaler),                  // name wins over extension
            ("4x-upscaler-v2.safetensors", .upscaler),
            ("model_index.json", nil),
            ("preview-esrgan.png", nil),                           // right name, wrong extension
            ("notes-about-upscaling.txt", nil),
            ("sd15", nil),                                         // no extension at all
        ]
        for testCase in kindCases {
            let got = kind(forFileName: testCase.input)
            if got != testCase.expected {
                failures.append("kind(\"\(testCase.input)\") = \(describe(got)), expected \(describe(testCase.expected))")
            }
        }

        // --- component-file exclusions ---
        let componentCases: [(input: String, expected: Bool)] = [
            ("sdxl_vae.safetensors", true),
            ("ae.vae.safetensors", true),
            ("clip_l.safetensors", true),
            ("clip_g.safetensors", true),
            ("t5xxl_fp16.safetensors", true),
            ("text_encoder.safetensors", true),
            ("add-detail-lora.safetensors", true),
            ("flux1-dev-tensordata.safetensors", true),
            ("sd15-Q4_0.gguf.part", true),
            ("sdxl.safetensors.partial", true),
            ("mmproj-Qwythos-9B-v2-BF16.gguf", true),
            ("mtp-gemma-4-12B-it.gguf", true),
            ("sd15-Q4_0.gguf", false),
            ("juggernautXL_v9.safetensors", false),
            ("RealESRGAN_x4plus_anime_6B.pth", false),
        ]
        for testCase in componentCases {
            let got = isComponentFile(testCase.input)
            if got != testCase.expected {
                failures.append("isComponentFile(\"\(testCase.input)\") = \(got), expected \(testCase.expected)")
            }
        }

        // --- architecture, by name (size deliberately absurd so a name miss is obvious) ---
        let nameArchCases: [(input: String, expected: ImageArch)] = [
            ("flux1-dev-Q4_0.gguf", .flux),
            ("FLUX.1-schnell-fp8.safetensors", .flux),
            ("qwen-image-edit-2509-Q4_K_M.gguf", .qwen),
            ("Qwen_Image_Q8_0.gguf", .qwen),
            ("sd3.5_large_turbo.safetensors", .sd3),
            ("sd_3_medium_incl_clips.safetensors", .sd3),
            ("stable-diffusion-3.5-medium.gguf", .sd3),
            ("sdxl-turbo.safetensors", .sdxl),
            ("sd_xl_base_1.0.safetensors", .sdxl),
            ("sd_xl_refiner_1.0.safetensors", .sdxl),
            ("juggernautXL_v9Rundiffusion.safetensors", .sdxl),
            ("Juggernaut-X-RunDiffusion-NSFW.safetensors", .sdxl),   // no "xl" in the name at all
            ("ponyDiffusionV6XL.safetensors", .sdxl),
            ("realvisxlV50_v50Bakedvae.safetensors", .sdxl),
            ("DreamShaperXL_Turbo_v2.safetensors", .sdxl),
            ("dreamshaper-xl-lightning.safetensors", .sdxl),
            ("dreamshaper_8.safetensors", .sd15),                    // plain DreamShaper is SD 1.5
            ("v1-5-pruned-emaonly.safetensors", .sd15),
            ("sd15-Q4_0.gguf", .sd15),
            ("sd-1.5-inpainting.safetensors", .sd15),
            ("Realistic_Vision_V6.0_NV_B1.safetensors", .sd15),
        ]
        for testCase in nameArchCases {
            let got = arch(forFileName: testCase.input, size: 99 * bytesPerGiB)
            if got != testCase.expected {
                failures.append("arch(\"\(testCase.input)\") = .\(got.rawValue), expected .\(testCase.expected.rawValue)")
            }
        }

        // --- architecture, by size fallback (name says nothing) ---
        let sizeArchCases: [(size: Int64, expected: ImageArch)] = [
            (2 * bytesPerGiB, .sd15),
            (7 * bytesPerGiB, .sdxl),
            (20 * bytesPerGiB, .flux),
        ]
        for testCase in sizeArchCases {
            let got = arch(forFileName: "mystery-checkpoint.safetensors", size: testCase.size)
            if got != testCase.expected {
                failures.append("arch(size: \(testCase.size / bytesPerGiB) GiB) = .\(got.rawValue), expected .\(testCase.expected.rawValue)")
            }
        }

        // --- quant + display name ---
        let quantCases: [(input: String, expected: String)] = [
            ("sd15-Q4_0.gguf", "Q4_0"),
            ("flux1-dev-Q4_K_M.gguf", "Q4_K_M"),
            ("flux1-schnell-f16.gguf", "F16"),
            ("juggernautXL_v9.safetensors", ""),
            ("RealESRGAN_x4plus_anime_6B.pth", ""),
        ]
        for testCase in quantCases {
            let got = quant(forFileName: testCase.input)
            if got != testCase.expected {
                failures.append("quant(\"\(testCase.input)\") = \"\(got)\", expected \"\(testCase.expected)\"")
            }
        }
        if displayName(forFileName: "sd3.5_large.safetensors") != "sd3.5_large" {
            failures.append("displayName dropped more than the extension from \"sd3.5_large.safetensors\"")
        }

        return failures
    }

    private static func describe(_ kind: ImageModelKind?) -> String {
        kind.map { ".\($0.rawValue)" } ?? "nil"
    }
}
