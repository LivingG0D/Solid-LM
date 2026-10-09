import Foundation

// MARK: - Roles

/// A file a multi-file diffusion model needs beside its transformer.
///
/// SD 1.5 and SDXL bundle the transformer, both text encoders and the VAE into one
/// checkpoint, which is why `-m <file>` alone was enough until now. Every family
/// newer than SDXL ships those pieces separately, so sd-server has to be told where
/// each one is. Flags verified against `sd-server --help` on this build.
enum ImageComponentRole: String, CaseIterable, Sendable {
    case vae, clipL, clipG, t5xxl, llm

    var flag: String {
        switch self {
        case .vae: return "--vae"
        case .clipL: return "--clip_l"
        case .clipG: return "--clip_g"
        case .t5xxl: return "--t5xxl"
        case .llm: return "--llm"
        }
    }

    /// Used in the "you are missing X" error, so it has to read like a file someone
    /// can go and download.
    var label: String {
        switch self {
        case .vae: return "VAE"
        case .clipL: return "CLIP-L text encoder"
        case .clipG: return "CLIP-G text encoder"
        case .t5xxl: return "T5-XXL text encoder"
        case .llm: return "LLM text encoder (Qwen)"
        }
    }

    /// Matched against lowercased file names, **most specific first** — the first
    /// pattern that matches anything wins.
    ///
    /// The ordering matters for `.vae`: FLUX and Z-Image both want the FLUX autoencoder,
    /// which ships as a bare `ae.safetensors`. A folder that also holds an SD 1.5
    /// `…-vae-ft-mse.safetensors` would otherwise hand the SD 1.5 VAE to FLUX and
    /// produce garbage latents.
    var patterns: [String] {
        switch self {
        case .vae: return [#"^ae\.(safetensors|sft|gguf)$"#, #"vae"#]
        case .clipL: return [#"clip[_\-]?l"#]
        case .clipG: return [#"clip[_\-]?g"#]
        case .t5xxl: return [#"t5[_\-]?xxl"#]
        // Z-Image wants Qwen3-4B, Qwen-Image wants Qwen2.5-VL. Both are "the Qwen
        // one", and a folder realistically holds whichever its model needs.
        case .llm: return [#"qwen[_\-]?3"#, #"qwen[_.\-]?2[._\-]?5[_\-]?vl"#]
        }
    }
}

// MARK: - Per-architecture requirements

extension ImageArch {
    /// Files sd-server needs beside the main one. Empty means a self-contained
    /// checkpoint, which is what `-m` alone has always handled.
    var componentRoles: [ImageComponentRole] {
        switch self {
        case .sd15, .sdxl, .unknown: return []
        case .sd3: return [.clipL, .clipG, .t5xxl]
        case .flux: return [.vae, .clipL, .t5xxl]
        case .zimage, .qwen: return [.vae, .llm]
        }
    }

    /// SD 3.5 distributes a full checkpoint that still needs external text encoders, so
    /// its main file stays on `-m`. FLUX, Z-Image and Qwen-Image distribute a bare
    /// transformer, which sd.cpp loads through `--diffusion-model` instead.
    var usesStandaloneDiffusionFlag: Bool {
        switch self {
        case .flux, .zimage, .qwen: return true
        case .sd15, .sdxl, .sd3, .unknown: return false
        }
    }
}

// MARK: - Resolution

/// Finds the companion files a multi-file checkpoint needs, by convention, from the
/// folders around it.
///
/// This mirrors how `LocalModel.draftPath` finds a sibling `mtp-*.gguf`: nobody wants
/// to hand-wire four paths in a settings screen for every model they download.
enum ImageComponents {

    /// Extensions a component can ship as. `.sft` is what sd.cpp's own docs use for
    /// the FLUX autoencoder.
    private static let componentExtensions: Set<String> = ["safetensors", "sft", "gguf"]

    // MARK: Pure matching

    /// Nil when nothing in `fileNames` can serve `role`.
    static func match(role: ImageComponentRole, in fileNames: [String]) -> String? {
        for pattern in role.patterns {
            let hit = fileNames.first {
                $0.lowercased().range(of: pattern, options: [.regularExpression]) != nil
            }
            if let hit { return hit }
        }
        return nil
    }

    // MARK: Filesystem

    /// A folder that holds components rather than models — `text_encoders/`, `vae/`.
    ///
    /// This is what bounds the search. Reaching into *any* neighbouring folder finds
    /// files that belong to a different model: a FLUX checkpoint with no T5 beside it
    /// would silently borrow the one from an unrelated model's folder and launch
    /// looking perfectly healthy.
    static func isComponentDirectory(_ name: String) -> Bool {
        let lower = name.lowercased()
        if lower.contains("encoder") { return true }
        return ["vae", "clip", "llm", "t5", "tokenizer"].contains(lower)
    }

    /// Where to look, nearest first: the model's own folder, then component folders
    /// beside it, then component folders beside its parent.
    ///
    /// Two layouts are common and this covers both — everything dropped flat into
    /// `_image-models/`, and the ComfyUI split where `diffusion_models/`,
    /// `text_encoders/` and `vae/` are siblings.
    ///
    /// The parent folder itself is deliberately never scanned. In the default layout
    /// the model's own folder *is* `_image-models/`, whose parent is the shared LM
    /// Studio models root — scanning that would offer every local Qwen3 language model
    /// as a candidate `--llm` text encoder, including a 30 GB one.
    static func searchDirectories(forModelAt modelPath: String, fm: FileManager = .default) -> [URL] {
        let own = URL(filePath: modelPath).deletingLastPathComponent()
        let parent = own.deletingLastPathComponent()

        var dirs: [URL] = []
        var seen = Set<String>()
        func add(_ url: URL) {
            if seen.insert(url.standardizedFileURL.path).inserted { dirs.append(url) }
        }

        add(own)
        for base in [own, parent] {
            let children = (try? fm.contentsOfDirectory(at: base,
                                                        includingPropertiesForKeys: [.isDirectoryKey],
                                                        options: [.skipsHiddenFiles])) ?? []
            for child in children.sorted(by: { $0.path < $1.path })
            where isComponentDirectory(child.lastPathComponent)
                && (try? child.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
                add(child)
            }
        }
        return dirs
    }

    /// Every file near `modelPath` that could be a component.
    /// Sorted by path so resolution is deterministic — `contentsOfDirectory` is not.
    static func candidates(forModelAt modelPath: String,
                           fm: FileManager = .default) -> [(name: String, path: String)] {
        let selfPath = URL(filePath: modelPath).standardizedFileURL.path
        var out: [(name: String, path: String)] = []
        for dir in searchDirectories(forModelAt: modelPath, fm: fm) {
            let entries = (try? fm.contentsOfDirectory(at: dir,
                                                       includingPropertiesForKeys: nil,
                                                       options: [.skipsHiddenFiles])) ?? []
            for entry in entries.sorted(by: { $0.path < $1.path })
            where componentExtensions.contains(entry.pathExtension.lowercased()) {
                let path = entry.standardizedFileURL.path
                guard path != selfPath else { continue }
                out.append((entry.lastPathComponent, path))
            }
        }
        return out
    }

    /// Resolved path per role. A role with nothing to match is simply absent.
    static func resolve(roles: [ImageComponentRole],
                        modelPath: String,
                        fm: FileManager = .default) -> [ImageComponentRole: String] {
        guard !roles.isEmpty else { return [:] }
        let files = candidates(forModelAt: modelPath, fm: fm)
        let names = files.map(\.name)

        var found: [ImageComponentRole: String] = [:]
        for role in roles {
            guard let name = match(role: role, in: names),
                  let file = files.first(where: { $0.name == name })
            else { continue }
            found[role] = file.path
        }
        return found
    }

    // MARK: Self check

    /// Pure name-level regression tests. The filesystem walk is deliberately not covered
    /// here — the part that breaks silently is the matching, where a wrong VAE still
    /// launches and only shows up as unusable images.
    static func selfCheck() -> [String] {
        var f: [String] = []

        func expect(_ role: ImageComponentRole, _ names: [String], _ want: String?, _ note: String) {
            let got = match(role: role, in: names)
            if got != want {
                f.append("match(.\(role.rawValue)) = \(got ?? "nil"), expected \(want ?? "nil") — \(note)")
            }
        }

        // The ordering rule: a bare `ae.safetensors` beats any other VAE in the folder.
        expect(.vae, ["sd15-vae-ft-mse.safetensors", "ae.safetensors"], "ae.safetensors",
               "FLUX autoencoder must win over an SD 1.5 VAE")
        expect(.vae, ["ae.sft"], "ae.sft", "sd.cpp docs ship the autoencoder as .sft")
        expect(.vae, ["sd15-vae-ft-mse.safetensors"], "sd15-vae-ft-mse.safetensors",
               "a named VAE is still a VAE when there is no bare ae")
        expect(.vae, ["clip_l.safetensors"], nil, "a text encoder is not a VAE")

        // CLIP-L and CLIP-G differ by one character and must not cross-match.
        let clips = ["clip_g.safetensors", "clip_l.safetensors", "t5xxl_fp16.safetensors"]
        expect(.clipL, clips, "clip_l.safetensors", "")
        expect(.clipG, clips, "clip_g.safetensors", "")
        expect(.t5xxl, clips, "t5xxl_fp16.safetensors", "")
        expect(.clipL, ["t5xxl_fp16.safetensors"], nil, "t5xxl contains an l but is not clip-l")
        expect(.clipG, ["clip_l.safetensors"], nil, "clip-l must not satisfy clip-g")

        // Both Qwen encoders, spelled the several ways the repos spell them.
        expect(.llm, ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"], "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
               "Z-Image's encoder")
        expect(.llm, ["qwen_3_4b.safetensors"], "qwen_3_4b.safetensors", "underscore spelling")
        expect(.llm, ["qwen2.5-vl-7b.safetensors"], "qwen2.5-vl-7b.safetensors", "Qwen-Image's encoder")
        expect(.llm, ["t5xxl_fp16.safetensors"], nil, "t5xxl is not an LLM encoder")

        // --- per-architecture wiring ---
        for arch in [ImageArch.sd15, .sdxl, .unknown] where !arch.componentRoles.isEmpty {
            f.append("\(arch.rawValue) is single-file and must need no components")
        }
        for arch in [ImageArch.sd15, .sdxl, .sd3, .unknown] where arch.usesStandaloneDiffusionFlag {
            f.append("\(arch.rawValue) must load through -m, not --diffusion-model")
        }
        for arch in [ImageArch.flux, .zimage, .qwen] where !arch.usesStandaloneDiffusionFlag {
            f.append("\(arch.rawValue) ships a bare transformer and needs --diffusion-model")
        }
        if ImageArch.sd3.componentRoles != [.clipL, .clipG, .t5xxl] {
            f.append("sd3 needs clip_l + clip_g + t5xxl")
        }
        if ImageArch.flux.componentRoles != [.vae, .clipL, .t5xxl] {
            f.append("flux needs vae + clip_l + t5xxl")
        }
        if ImageArch.zimage.componentRoles != [.vae, .llm] { f.append("zimage needs vae + llm") }
        if ImageArch.qwen.componentRoles != [.vae, .llm] { f.append("qwen needs vae + llm") }

        // --- search scope ---
        // The bound that stops one model borrowing another model's encoder.
        for name in ["text_encoders", "text_encoder", "encoders", "vae", "clip", "llm", "VAE"]
        where !isComponentDirectory(name) {
            f.append("\"\(name)\" should be searched for components")
        }
        for name in ["diffusion_models", "flux", "z-image", "sd35", "unet", "loras", "checkpoints"]
        where isComponentDirectory(name) {
            f.append("\"\(name)\" holds models, not components — searching it lets one model steal another's encoder")
        }

        // Flags are what actually reaches the process; a typo is a silent launch failure.
        let flags = Dictionary(uniqueKeysWithValues: ImageComponentRole.allCases.map { ($0, $0.flag) })
        let want: [ImageComponentRole: String] = [.vae: "--vae", .clipL: "--clip_l", .clipG: "--clip_g",
                                                  .t5xxl: "--t5xxl", .llm: "--llm"]
        if flags != want { f.append("component flags drifted from sd-server's: \(flags)") }

        return f
    }
}
