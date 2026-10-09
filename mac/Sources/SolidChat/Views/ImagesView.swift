import AppKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

// ---------------------------------------------------------------------------
// Surface consumed from the rest of the module.
//
//   AppStore     var  engineState: EngineState        // the LLM engine
//                var  activeModel: LocalModel?        // the loaded LLM, for the memory warning
//                let  stats: SystemStats              // .totalBytes
//                let  images: ImageStore
//
//   ImagePaths   static var imageModelRoots: [URL]    (Core/ImageEngine.swift)
//
//   ImageStore   @Observable @MainActor final class   (Core/ImageStore.swift)
//     models     var  diffusionModels / upscalerModels: [ImageModel]
//                func rescan() async
//     engine     var  state: EngineState
//                var  loadedModel: ImageModel?
//                func load(_:) / unload() / applyDefaults(for:)
//     input      var  mode: ImageMode
//                var  params: ImageParams
//                var  inputImage / maskImage: URL?
//                var  upscaler: ImageModel?
//                var  upscaleRepeats: Int             // 1...4
//                var  upscaleTileSize: Int            // 0 = auto (ImageClient picks by input size)
//     running    var  isBusy: Bool, canRun: Bool
//                var  progress: ImageProgress
//                var  lastError: String?
//                func run() / cancel() / clearError()
//     results    var  gallery: [GeneratedImage]       // any order; sorted here
//                var  galleryDirectory: URL           // for GeneratedImage.url(in:)
//                func delete(_ image: GeneratedImage)
// ---------------------------------------------------------------------------

// MARK: - Cross-view signals

enum ImagesViewSignals {
    /// Posted by the empty state's "Get Models" button when no explicit handler was
    /// injected. `RootView` owns the section picker, so for the button to reach the
    /// Downloads section it has to observe this:
    ///
    ///     .onReceive(NotificationCenter.default.publisher(for: ImagesViewSignals.showDownloads)) { _ in
    ///         tab = .downloads
    ///     }
    static let showDownloads = Notification.Name("solidchat.showDownloads")
}

// MARK: - Constants

/// The VAE downsamples by 8 and the UNet halves it three more times, so a side that
/// is not a multiple of 64 gets quietly rounded by the backend anyway.
private let sizeStep = 64
private let sizeBounds = 256...2048

/// One click each. The two rectangles are the SDXL aspect buckets.
private let sizePresets: [(label: String, width: Int, height: Int)] = [
    ("512", 512, 512),
    ("768", 768, 768),
    ("1024", 1024, 1024),
    ("832×1216", 832, 1216),
    ("1216×832", 1216, 832),
]

/// 0 means one pass over the whole image.
private let tileSizes = [0, 128, 256, 512]

/// Both engines draw on the same unified memory. Leave this much for macOS, the
/// window server and the app itself — on this 24 GB machine that is a ~20 GB budget.
private let memoryHeadroom: Int64 = 4 * 1_073_741_824

/// Past this on a side the result is measured in gigapixels.
private let hugeResultThreshold = 8192

/// Where checkpoints are meant to go. The scanner walks each root recursively, so
/// this is a convention rather than a rule — but it is the convention that keeps
/// image models out of the LLM list, since that scanner skips names starting "_".
private var imageModelsHint: String {
    let root = ImagePaths.imageModelRoots.first
        ?? FileManager.default.homeDirectoryForCurrentUser
    return root
        .appending(path: "_image-models", directoryHint: .isDirectory)
        .path(percentEncoded: false)
}

// MARK: - Shared derivations

/// Newest first, whatever order the store happens to keep them in.
@MainActor
private func sortedGallery(_ images: ImageStore) -> [GeneratedImage] {
    images.gallery.sorted { $0.created > $1.created }
}

/// What the canvas shows: the explicit selection, else the newest result.
@MainActor
private func displayedImage(_ images: ImageStore, _ selection: UUID?) -> GeneratedImage? {
    let list = sortedGallery(images)
    if let selection, let found = list.first(where: { $0.id == selection }) { return found }
    return list.first
}

/// `model · 512 × 512 · 20 steps · cfg 7 · euler_a · seed 12345 · 4.2s`.
///
/// Fields that mean nothing for the mode that produced the image — steps on an
/// upscale — are dropped rather than printed as a misleading zero. Deliberately
/// nonisolated: it reads a `Sendable` value and nothing else.
private func imageSummary(_ image: GeneratedImage) -> String {
    var parts: [String] = []
    parts.append(image.modelName.isEmpty ? image.mode.label.lowercased() : image.modelName)
    parts.append("\(image.width) × \(image.height)")
    if image.steps > 0 { parts.append("\(image.steps) steps") }
    if image.cfgScale > 0 { parts.append("cfg \(ImgFmt.number(image.cfgScale))") }
    if !image.sampler.isEmpty { parts.append(image.sampler) }
    if image.seed >= 0 { parts.append("seed \(image.seed)") }
    if image.elapsed > 0 { parts.append(String(format: "%.1fs", image.elapsed)) }
    return parts.joined(separator: " · ")
}

// MARK: - ImagesView

/// The whole image surface: mode picker on top, controls on the left, canvas and
/// gallery on the right.
@MainActor
struct ImagesView: View {
    /// Optional injection point for "Get Models". `RootView` calls `ImagesView()`,
    /// in which case the button falls back to `ImagesViewSignals.showDownloads`.
    var onShowDownloads: (() -> Void)?

    @Environment(AppStore.self) private var store

    /// nil means "follow the newest result".
    @State private var selectedImageID: UUID?

    private var images: ImageStore { store.images }

    private var hasAnyModel: Bool {
        !images.diffusionModels.isEmpty || !images.upscalerModels.isEmpty
    }

    var body: some View {
        VStack(spacing: 0) {
            modeBar
            Divider()

            if hasAnyModel {
                HSplitView {
                    ImageControlsPane(selectedImageID: $selectedImageID)
                        .frame(minWidth: 300, idealWidth: 344, maxWidth: 520)
                    ImageCanvasPane(selectedImageID: $selectedImageID)
                        .frame(minWidth: 340)
                }
            } else {
                ImagesEmptyState(onShowDownloads: showDownloads)
            }
        }
        // A finished run should be what you are looking at — unless you deliberately
        // went back to an older one, which is exactly what a non-nil selection means.
        .onChange(of: images.gallery.count) { previous, current in
            if current > previous { selectedImageID = nil }
        }
    }

    // MARK: Mode bar

    private var modeBar: some View {
        @Bindable var bound = store.images

        return HStack(spacing: 12) {
            Picker("Mode", selection: $bound.mode) {
                ForEach(ImageMode.allCases, id: \.self) { mode in
                    Label(mode.label, systemImage: mode.symbol)
                        .labelStyle(.titleAndIcon)
                        .tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(minWidth: 280, idealWidth: 340)
            // No mode is ever disabled: Upscale needs no diffusion model, and for the
            // other two the run button says what is missing rather than going grey
            // with no explanation.
            .help("Generate makes an image from a prompt, Edit reworks one you supply, Upscale enlarges one with ESRGAN")
            .accessibilityLabel("Image mode")

            Spacer(minLength: 0)

            Text(modeExplanation)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .background(.bar)
    }

    private var modeExplanation: String {
        switch images.mode {
        case .generate: return "Text to image"
        case .edit: return "Image to image, with an optional inpainting mask"
        case .upscale: return "ESRGAN ×4 — no diffusion model needed"
        }
    }

    private func showDownloads() {
        if let onShowDownloads {
            onShowDownloads()
        } else {
            NotificationCenter.default.post(name: ImagesViewSignals.showDownloads, object: nil)
        }
    }
}

// MARK: - Controls pane

@MainActor
private struct ImageControlsPane: View {
    @Binding var selectedImageID: UUID?

    @Environment(AppStore.self) private var store

    private var images: ImageStore { store.images }

    var body: some View {
        ScrollView(.vertical) {
            VStack(alignment: .leading, spacing: 18) {
                ImageModelBar()
                Divider()
                modeControls
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            VStack(spacing: 0) {
                Divider()
                RunBar()
            }
            .background(.bar)
        }
    }

    @ViewBuilder
    private var modeControls: some View {
        switch images.mode {
        case .generate:
            PromptSection()
            SizeSection()
            SamplingSection(selectedImageID: $selectedImageID)

        case .edit:
            EditInputSection()
            PromptSection()
            SizeSection()
            SamplingSection(selectedImageID: $selectedImageID)

        case .upscale:
            UpscaleSection()
        }
    }
}

// MARK: - Model bar

/// Always visible, in every mode. Choosing a model loads it: there is one image
/// engine, so a selection *is* a load.
@MainActor
private struct ImageModelBar: View {
    @Environment(AppStore.self) private var store

    private var images: ImageStore { store.images }
    private var loaded: ImageModel? { images.loadedModel }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ImageSectionHeader("Diffusion model",
                               help: "The checkpoint used for Generate and Edit. Upscale does not need one.")

            HStack(spacing: 8) {
                Menu {
                    modelMenuItems
                } label: {
                    menuLabel
                }
                .menuStyle(.borderlessButton)
                .disabled(images.state.isLoading)
                .help(loaded.map { "\($0.path)\n\($0.arch.label) · \(Fmt.bytes($0.size))" }
                      ?? "Choose a checkpoint to load")
                .accessibilityLabel("Diffusion model")

                if images.state.isLoading {
                    ProgressView()
                        .progressViewStyle(.circular)
                        .controlSize(.small)
                        .accessibilityLabel("Loading the diffusion model")
                }

                if loaded != nil {
                    Button {
                        images.unload()
                    } label: {
                        Image(systemName: "eject")
                    }
                    .buttonStyle(.borderless)
                    .help("Unload the diffusion model and give its memory back")
                    .accessibilityLabel("Unload the diffusion model")
                }
            }

            if images.diffusionModels.isEmpty {
                ImageNotice(text: "No diffusion checkpoints found. Upscale still works — it only needs an ESRGAN network.",
                            systemImage: "info.circle",
                            tint: .secondary)
            }

            if let memoryWarning {
                ImageNotice(text: memoryWarning, systemImage: "exclamationmark.triangle.fill", tint: .orange)
            }

            // Measured, not theoretical — see ImageModel.quantWarning.
            if let quantWarning = images.loadedModel?.quantWarning {
                ImageNotice(text: quantWarning, systemImage: "paintpalette", tint: .orange)
            }

            if let failure = images.state.errorText {
                ImageNotice(text: failure, systemImage: "exclamationmark.triangle.fill", tint: .red)
            }
        }
    }

    // MARK: Menu

    @ViewBuilder
    private var menuLabel: some View {
        if let loaded {
            HStack(spacing: 6) {
                Circle()
                    .fill(images.state.isReady ? Color.green : Color.orange)
                    .frame(width: 7, height: 7)
                    .accessibilityHidden(true)
                Text(loaded.name)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(loaded.arch.label)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        } else if images.state.isLoading {
            Text("Loading…")
        } else {
            Text("Choose a model…")
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var modelMenuItems: some View {
        if images.diffusionModels.isEmpty {
            Text("No diffusion models found")
        } else {
            ForEach(images.diffusionModels) { model in
                // Menu rows on macOS render as one line of text, so everything worth
                // knowing goes in that line rather than into a stack that is dropped.
                Button(rowTitle(for: model)) {
                    select(model)
                }
                .help(rowHelp(for: model))
            }
        }

        Divider()

        Button("Rescan Image Models") {
            Task { await store.images.rescan() }
        }
    }

    private func rowTitle(for model: ImageModel) -> String {
        var text = model.name
        if model.arch != .unknown { text += "  ·  \(model.arch.label)" }
        if !model.quant.isEmpty { text += "  ·  \(model.quant)" }
        text += "  ·  \(Fmt.bytes(model.size))"
        return text
    }

    private func rowHelp(for model: ImageModel) -> String {
        var text = "\(model.path)\nDefaults to \(model.arch.nativeSize)px, \(model.arch.defaultSteps) steps, CFG \(ImgFmt.number(model.arch.defaultCFG))."
        // Multi-file families load nothing without their encoders, and finding out at
        // load time is worse than seeing it here.
        let roles = model.arch.componentRoles
        if !roles.isEmpty {
            let found = ImageComponents.resolve(roles: roles, modelPath: model.path)
            if found.isEmpty {
                text += "\nLoads as a single file. If that fails, this family also ships split "
                    + "across \(roles.map(\.label).joined(separator: ", "))."
            } else {
                let lines = roles.map { role in
                    "  \(found[role] != nil ? "✓" : "✗") \(role.label)"
                }
                text += "\nCompanion files:\n" + lines.joined(separator: "\n")
            }
        }
        if let over = overBudget(imageBytes: model.size) { text += "\n\(over)" }
        return text
    }

    /// `load` applies the architecture defaults itself, but doing it here too means
    /// the controls snap to the new model's sizes and step count the moment you pick
    /// it, rather than a minute later when sd-server finishes coming up.
    private func select(_ model: ImageModel) {
        images.applyDefaults(for: model)
        images.load(model)
    }

    // MARK: Memory

    private var memoryWarning: String? {
        guard let loaded else { return nil }
        return overBudget(imageBytes: loaded.size)
    }

    /// The two engines are separate processes sharing one pool of unified memory, so
    /// the number that matters is the sum, not either one on its own.
    private func overBudget(imageBytes: Int64) -> String? {
        let budget = max(store.stats.totalBytes - memoryHeadroom, 0)
        guard budget > 0 else { return nil }

        let llm = store.engineState.isReady ? store.activeModel : nil
        let llmBytes = llm?.size ?? 0
        let total = imageBytes + llmBytes
        guard total > budget else { return nil }

        if let llm {
            return "This checkpoint (\(Fmt.bytes(imageBytes))) plus the loaded language model \(llm.name) (\(Fmt.bytes(llmBytes))) comes to \(Fmt.bytes(total)) — past the \(Fmt.bytes(budget)) this machine can spare. Unload one of them before generating."
        }
        return "This checkpoint alone wants \(Fmt.bytes(imageBytes)), past the \(Fmt.bytes(budget)) this machine can spare. Expect swapping."
    }
}

// MARK: - Prompt

@MainActor
private struct PromptSection: View {
    @Environment(AppStore.self) private var store

    private var images: ImageStore { store.images }

    /// FLUX, Z-Image-Turbo and Qwen-Image are guidance-distilled: they run at CFG 1
    /// with no unconditional branch, so a negative prompt has nothing to steer with.
    private var supportsNegativePrompt: Bool {
        switch images.loadedModel?.arch ?? .unknown {
        case .flux, .zimage, .qwen: return false
        default: return true
        }
    }

    var body: some View {
        @Bindable var bound = store.images

        return VStack(alignment: .leading, spacing: 10) {
            ImageSectionHeader("Prompt", help: "What you want to see")

            PromptEditor(text: $bound.params.prompt,
                         placeholder: "a still life of pears on a windowsill, morning light",
                         minHeight: 92,
                         label: "Prompt")

            if supportsNegativePrompt {
                ImageSectionHeader("Negative prompt",
                                   help: "What to steer away from. Leave it empty if you have nothing specific in mind.")
                PromptEditor(text: $bound.params.negativePrompt,
                             placeholder: "blurry, extra fingers, watermark",
                             minHeight: 56,
                             label: "Negative prompt")
            } else {
                ImageNotice(text: "\(images.loadedModel?.arch.label ?? "This model") is guidance-distilled and ignores a negative prompt, so the field is hidden rather than left sitting there doing nothing.",
                            systemImage: "info.circle",
                            tint: .secondary)
            }
        }
    }
}

/// A `TextEditor` with a placeholder and the same border as the rest of the app.
@MainActor
private struct PromptEditor: View {
    @Binding var text: String
    let placeholder: String
    let minHeight: CGFloat
    let label: String

    var body: some View {
        ZStack(alignment: .topLeading) {
            if text.isEmpty {
                Text(placeholder)
                    .foregroundStyle(.tertiary)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 8)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
            }
            TextEditor(text: $text)
                .font(.body)
                .scrollContentBackground(.hidden)
                .frame(minHeight: minHeight)
                .padding(4)
                .accessibilityLabel(label)
        }
        .background(Color(nsColor: .textBackgroundColor),
                    in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(Color(nsColor: .separatorColor))
        }
    }
}

// MARK: - Size

@MainActor
private struct SizeSection: View {
    @Environment(AppStore.self) private var store

    private var images: ImageStore { store.images }

    private var nativeSize: Int { (images.loadedModel?.arch ?? .unknown).nativeSize }

    /// Far from the training resolution these models repeat limbs and horizons.
    private var offNative: Bool {
        guard images.loadedModel != nil else { return false }
        return max(images.params.width, images.params.height) > nativeSize * 2
            || min(images.params.width, images.params.height) < nativeSize / 2
    }

    var body: some View {
        @Bindable var bound = store.images

        return VStack(alignment: .leading, spacing: 10) {
            ImageSectionHeader("Size",
                               help: "Output pixels. Multiples of \(sizeStep) only — anything else is rounded by the backend.")

            HStack(spacing: 14) {
                dimensionField("Width", value: $bound.params.width)
                dimensionField("Height", value: $bound.params.height)
            }

            HStack(spacing: 6) {
                ForEach(sizePresets, id: \.label) { preset in
                    Button(preset.label) {
                        store.images.params.width = preset.width
                        store.images.params.height = preset.height
                    }
                    .controlSize(.small)
                    .help("\(preset.width) × \(preset.height)")
                    .accessibilityLabel("Set size to \(preset.width) by \(preset.height)")
                }

                Spacer(minLength: 0)

                Button {
                    store.images.params.width = nativeSize
                    store.images.params.height = nativeSize
                } label: {
                    Image(systemName: "arrow.counterclockwise")
                }
                .buttonStyle(.borderless)
                .controlSize(.small)
                .help("Back to this model's native \(nativeSize) × \(nativeSize)")
                .accessibilityLabel("Reset to the model's native size")
            }

            if offNative {
                ImageNotice(text: "This model was trained at \(nativeSize)px. Far from that it starts repeating limbs and horizons — generate near native, then use Upscale to get bigger.",
                            systemImage: "info.circle",
                            tint: .secondary)
            }
        }
    }

    /// `Stepper(value:in:step:)` does the clamping and the 64-pixel stride itself, so
    /// the binding is handed through untouched — a hand-rolled one would only be a
    /// second place for the bounds to drift out of agreement.
    private func dimensionField(_ title: String, value: Binding<Int>) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack(spacing: 6) {
                Text("\(value.wrappedValue)")
                    .font(.body.monospacedDigit())
                    .frame(width: 46, alignment: .trailing)
                Stepper(title, value: value, in: sizeBounds, step: sizeStep)
                    .labelsHidden()
                    .accessibilityLabel("\(title) in pixels")
                    .accessibilityValue("\(value.wrappedValue)")
            }
        }
        .help("\(title): \(sizeBounds.lowerBound) to \(sizeBounds.upperBound) pixels, in steps of \(sizeStep)")
    }
}

// MARK: - Sampling

@MainActor
private struct SamplingSection: View {
    @Binding var selectedImageID: UUID?

    @Environment(AppStore.self) private var store

    private var images: ImageStore { store.images }

    /// The seed the lock button copies back in: whatever is on the canvas.
    private var canvasSeed: Int64? {
        displayedImage(images, selectedImageID)?.seed
    }

    private var isDistilled: Bool {
        switch images.loadedModel?.arch ?? .unknown {
        case .flux, .zimage, .qwen: return true
        default: return false
        }
    }

    var body: some View {
        @Bindable var bound = store.images

        return VStack(alignment: .leading, spacing: 12) {
            ImageSectionHeader("Sampling", help: "How the image is denoised")

            sliderRow(title: "Steps",
                      value: $bound.params.steps.asDouble,
                      display: "\(images.params.steps)",
                      range: 1...100,
                      step: 1,
                      help: "Denoising passes. More is slower and, past this model's sweet spot, no better.",
                      label: "Steps")

            sliderRow(title: "CFG",
                      value: $bound.params.cfgScale,
                      display: ImgFmt.number(images.params.cfgScale),
                      range: 0...15,
                      step: 0.1,
                      help: "How hard the model is pushed toward the prompt. Too high burns contrast and detail.",
                      label: "Classifier-free guidance")

            if isDistilled {
                ImageNotice(text: "Keep CFG at 1 for \(images.loadedModel?.arch.label ?? "this model") — it was distilled on guidance, and anything higher scorches the output.",
                            systemImage: "info.circle",
                            tint: .secondary)
            }

            Picker("Sampler", selection: $bound.params.sampler) {
                ForEach(ImageParams.samplers, id: \.self) { name in
                    Text(name).tag(name)
                }
            }
            .help("The solver that walks the noise down. euler_a is a safe default; dpm++2m converges in fewer steps.")
            .accessibilityLabel("Sampler")

            Picker("Scheduler", selection: $bound.params.scheduler) {
                ForEach(ImageParams.schedulers, id: \.self) { name in
                    Text(name).tag(name)
                }
            }
            .help("How noise levels are spaced across the steps. karras helps most samplers at low step counts.")
            .accessibilityLabel("Scheduler")

            Stepper(value: $bound.params.batchCount, in: 1...8) {
                Text("Batch  \(images.params.batchCount)")
                    .monospacedDigit()
            }
            .help("Images per run. Each one costs a full generation, so eight is eight times the wait.")
            .accessibilityLabel("Batch count")
            .accessibilityValue("\(images.params.batchCount)")

            seedRow
        }
    }

    // MARK: Seed

    private var seedRow: some View {
        @Bindable var bound = store.images

        return VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 6) {
                Text("Seed")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                TextField("Seed", value: $bound.params.seed, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .labelsHidden()
                    .monospacedDigit()
                    .frame(width: 148)
                    .accessibilityLabel("Seed")

                Button {
                    store.images.params.seed = -1
                } label: {
                    Image(systemName: "dice")
                }
                .buttonStyle(.borderless)
                .disabled(images.params.seed == -1)
                .help("Draw a fresh seed on every run")
                .accessibilityLabel("Use a random seed")

                Button {
                    if let canvasSeed { store.images.params.seed = canvasSeed }
                } label: {
                    Image(systemName: "lock")
                }
                .buttonStyle(.borderless)
                .disabled(canvasSeed == nil)
                .help(canvasSeed.map { "Reuse seed \($0) — the image on the canvas" }
                      ?? "Nothing on the canvas to take a seed from")
                .accessibilityLabel("Reuse the seed of the image on the canvas")

                Spacer(minLength: 0)
            }

            Text(images.params.seed == -1
                 ? "Random: every run starts from different noise."
                 : "Fixed: the same prompt and settings reproduce this image exactly.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: Slider row

    private func sliderRow(title: String,
                           value: Binding<Double>,
                           display: String,
                           range: ClosedRange<Double>,
                           step: Double,
                           help: String,
                           label: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 0)
                Text(display)
                    .font(.caption.monospacedDigit())
            }
            Slider(value: value, in: range, step: step)
                .accessibilityLabel(label)
                .accessibilityValue(display)
        }
        .help(help)
    }
}

// MARK: - Edit input

@MainActor
private struct EditInputSection: View {
    @Environment(AppStore.self) private var store

    private var images: ImageStore { store.images }

    var body: some View {
        @Bindable var bound = store.images

        return VStack(alignment: .leading, spacing: 12) {
            ImageSectionHeader("Input image", help: "The picture the model reworks")

            ImageWell(url: $bound.inputImage,
                      emptyTitle: "Drop an image here",
                      emptySymbol: "photo.on.rectangle.angled",
                      label: "Input image")

            strengthRow

            ImageSectionHeader("Mask", help: "Optional — inpainting only")

            ImageWell(url: $bound.maskImage,
                      emptyTitle: "Optional inpainting mask",
                      emptySymbol: "theatermasks",
                      label: "Inpainting mask")

            Text("White marks the region to change; black is left alone. Leave it empty to rework the whole image.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var strengthRow: some View {
        @Bindable var bound = store.images

        return VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text("Strength")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 0)
                Text(ImgFmt.number(images.params.strength))
                    .font(.caption.monospacedDigit())
            }
            Slider(value: $bound.params.strength, in: 0...1, step: 0.05)
                .accessibilityLabel("Denoising strength")
                .accessibilityValue(ImgFmt.number(images.params.strength))
            Text("Low keeps the original and only nudges it; high throws most of it away and follows the prompt instead.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - Upscale

@MainActor
private struct UpscaleSection: View {
    @Environment(AppStore.self) private var store

    @State private var inputPixels: PixelSize?

    private var images: ImageStore { store.images }

    /// Each pass of an ×4 network multiplies both sides by four.
    private var factor: Int {
        1 << (2 * max(1, min(images.upscaleRepeats, 4)))
    }

    private var result: PixelSize? {
        guard let inputPixels else { return nil }
        return PixelSize(width: inputPixels.width * factor, height: inputPixels.height * factor)
    }

    var body: some View {
        @Bindable var bound = store.images

        return VStack(alignment: .leading, spacing: 12) {
            ImageSectionHeader("Input image", help: "The picture to enlarge")

            ImageWell(url: $bound.inputImage,
                      emptyTitle: "Drop an image here",
                      emptySymbol: "photo.on.rectangle.angled",
                      label: "Image to upscale")

            ImageNotice(text: "Upscaling runs the ESRGAN network on its own — no diffusion model has to be loaded.",
                        systemImage: "info.circle",
                        tint: .secondary)

            ImageSectionHeader("Upscaler", help: "An ESRGAN .pth network")

            upscalerPicker

            Stepper(value: $bound.upscaleRepeats, in: 1...4) {
                Text("Repeats  \(images.upscaleRepeats)")
                    .monospacedDigit()
            }
            .help("How many times to run the network. Each pass is another ×4, so two passes is ×16.")
            .accessibilityLabel("Upscale repeats")
            .accessibilityValue("\(images.upscaleRepeats)")

            Picker("Tile size", selection: $bound.upscaleTileSize) {
                ForEach(tileSizes, id: \.self) { size in
                    Text(size == 0 ? "Auto" : "\(size) px").tag(size)
                }
            }
            .help("Process the image in pieces instead of all at once. Auto picks 512 for images up to 512 px (measured 1.5x faster) and 128 above that.")
            .accessibilityLabel("Tile size")

            Text("Tiling trades speed for memory: smaller tiles fit in less RAM but can leave faint seams. Bigger is not always faster — an image that fits in a single tile skips tiling entirely, which is why Auto uses 512 for small inputs and 128 for large ones.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            resultPreview
        }
        .task(id: images.inputImage) {
            inputPixels = await Self.probe(store.images.inputImage)
        }
    }

    @ViewBuilder
    private var upscalerPicker: some View {
        @Bindable var bound = store.images

        if images.upscalerModels.isEmpty {
            ImageNotice(text: "No ESRGAN network found. Put a .pth file in \(imageModelsHint).",
                        systemImage: "exclamationmark.triangle.fill",
                        tint: .orange)
        } else {
            Picker("Upscaler", selection: $bound.upscaler) {
                Text("None").tag(ImageModel?.none)
                ForEach(images.upscalerModels) { model in
                    Text("\(model.name)  ·  \(Fmt.bytes(model.size))").tag(ImageModel?.some(model))
                }
            }
            .labelsHidden()
            .help("The network that invents the extra pixels. Anime networks smooth line art; general ones keep photographic grain.")
            .accessibilityLabel("Upscaler network")
        }
    }

    // MARK: Result preview

    @ViewBuilder
    private var resultPreview: some View {
        if let inputPixels, let result {
            VStack(alignment: .leading, spacing: 5) {
                Text("\(inputPixels.text)  →  \(result.text)")
                    .font(.callout.monospaced())
                    .textSelection(.enabled)
                    .accessibilityLabel("The result will be \(result.width) by \(result.height) pixels")

                if max(result.width, result.height) > hugeResultThreshold {
                    ImageNotice(text: "That is \(ImgFmt.megapixels(result.width, result.height)) — it will take minutes and may exhaust memory. Drop a repeat, or turn on tiling.",
                                systemImage: "exclamationmark.triangle.fill",
                                tint: .orange)
                }
            }
        } else if images.inputImage != nil {
            Text("Reading the image size…")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    /// Header-only read, off the main actor: the dimensions come out of the file's
    /// metadata with no decode, which is what makes probing an 8192² PNG free.
    private static func probe(_ url: URL?) async -> PixelSize? {
        guard let url else { return nil }
        return await Task.detached(priority: .utility) {
            ImageFiles.pixelSize(of: url)
        }.value
    }
}

private struct PixelSize: Equatable, Sendable {
    var width: Int
    var height: Int

    var text: String { "\(width) × \(height)" }
}

// MARK: - Run bar

@MainActor
private struct RunBar: View {
    @Environment(AppStore.self) private var store

    private var images: ImageStore { store.images }

    private var trimmedPrompt: String {
        images.params.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Why the run button is off, phrased so it can be read straight out.
    ///
    /// `ImageStore.canRun` decides *whether* the button is enabled — duplicating that
    /// rule here would eventually let the two disagree. This only has to explain it,
    /// so every branch mirrors a branch of `canRun` and nothing more.
    private var blockReason: String? {
        guard !images.canRun, !images.isBusy else { return nil }

        switch images.mode {
        case .generate:
            if !images.state.isReady { return notReadyReason }
            if trimmedPrompt.isEmpty { return "Type a prompt first." }

        case .edit:
            if images.inputImage == nil {
                return "Edit needs an input image — drop one on the well above, or choose a file."
            }
            if !images.state.isReady { return notReadyReason }

        case .upscale:
            if images.inputImage == nil {
                return "Upscale needs an input image — drop one on the well above, or choose a file."
            }
            if images.upscaler == nil {
                return images.upscalerModels.isEmpty
                    ? "No ESRGAN network is installed. Put a .pth file in \(imageModelsHint)."
                    : "Choose an upscaler network."
            }
        }
        return nil
    }

    private var notReadyReason: String {
        if images.state.isLoading { return "The diffusion model is still loading." }
        if let failure = images.state.errorText { return failure }
        return "Load a diffusion model first — pick one at the top."
    }

    private var runTitle: String {
        switch images.mode {
        case .generate: return images.params.batchCount > 1 ? "Generate \(images.params.batchCount)" : "Generate"
        case .edit: return "Edit"
        case .upscale: return "Upscale"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if images.isBusy {
                Button {
                    images.cancel()
                } label: {
                    Label("Cancel", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .tint(.red)
                .help("Stop this run. Anything already finished is kept.")
                .accessibilityLabel("Cancel the running job")
            } else {
                Button {
                    images.run()
                } label: {
                    Label(runTitle, systemImage: images.mode.symbol)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(!images.canRun)
                .help(blockReason ?? "\(runTitle) — ⌘Return")
                .accessibilityLabel(runTitle)
                .accessibilityHint(blockReason ?? "Starts the run")
            }

            // A disabled control's tooltip is easy to miss, and "nothing happens when
            // I press it" is the worst answer available — so say it in the open too.
            if let blockReason {
                Text(blockReason)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }
}

// MARK: - Canvas pane

@MainActor
private struct ImageCanvasPane: View {
    @Binding var selectedImageID: UUID?

    @Environment(AppStore.self) private var store
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var busySince: Date?

    private var images: ImageStore { store.images }
    private var shown: GeneratedImage? { displayedImage(images, selectedImageID) }

    var body: some View {
        VStack(spacing: 0) {
            if let failure = images.lastError, !failure.isEmpty {
                ImageErrorBanner(text: failure) { store.images.clearError() }
                Divider()
            }

            canvas
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            if let shown {
                Divider()
                Text(imageSummary(shown))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .lineLimit(2)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 7)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            Divider()
            GalleryStrip(selectedImageID: $selectedImageID)
        }
        .onChange(of: images.isBusy, initial: true) { _, busy in
            busySince = busy ? (busySince ?? .now) : nil
        }
    }

    // MARK: Canvas

    private var canvas: some View {
        ZStack {
            Checkerboard()

            if let shown {
                let url = shown.url(in: images.galleryDirectory)
                ImageFileView(url: url, label: imageSummary(shown))
                    .padding(16)
                    .onTapGesture { ImageActions.open(url) }
                    .help("Open this image in Preview")
                    .accessibilityAddTraits(.isButton)
                    .contextMenu {
                        GalleryMenu(image: shown, selectedImageID: $selectedImageID)
                    }
            } else if !images.isBusy {
                placeholder
            }

            if images.isBusy {
                busyOverlay
            }
        }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: images.isBusy)
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: shown?.id)
    }

    private var placeholder: some View {
        VStack(spacing: 10) {
            Image(systemName: images.mode.symbol)
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(.tertiary)
                .accessibilityHidden(true)
            Text("Nothing here yet")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .padding(30)
    }

    // MARK: Busy

    private var busyOverlay: some View {
        VStack(spacing: 10) {
            // sd.cpp only reports per-step progress when it feels like it. When it has
            // not, an indeterminate spinner is the honest answer — a fabricated step
            // counter crawling at the wrong rate is worse than no number at all.
            if images.progress.totalSteps > 0 {
                ProgressView(value: images.progress.fraction)
                    .progressViewStyle(.linear)
                    .frame(width: 220)
                Text("step \(images.progress.step) of \(images.progress.totalSteps)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            } else {
                ProgressView()
                    .progressViewStyle(.circular)
                    .controlSize(.large)
            }

            if !images.progress.stage.isEmpty {
                Text(images.progress.stage)
                    .font(.callout)
                    .multilineTextAlignment(.center)
            }

            if let busySince {
                TimelineView(.periodic(from: busySince, by: 1)) { context in
                    Text(Self.elapsed(since: busySince, now: context.date))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(24)
        .frame(maxWidth: 320)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(Color(nsColor: .separatorColor))
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(images.progress.stage.isEmpty ? "Working" : images.progress.stage)
    }

    /// `Fmt.duration` is empty below a second, and a timer showing nothing looks broken.
    private static func elapsed(since start: Date, now: Date) -> String {
        let text = Fmt.duration(max(0, now.timeIntervalSince(start)))
        return text.isEmpty ? "0s" : text
    }
}

// MARK: - Error banner

/// Inline, dismissible and selectable. An alert would take the message away the
/// moment it was dismissed, and these are exactly the messages worth pasting.
@MainActor
private struct ImageErrorBanner: View {
    let text: String
    let onDismiss: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 9) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
                .accessibilityHidden(true)

            ScrollView(.vertical) {
                Text(text)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 90)

            Button {
                let board = NSPasteboard.general
                board.clearContents()
                board.setString(text, forType: .string)
            } label: {
                Image(systemName: "doc.on.doc")
            }
            .buttonStyle(.borderless)
            .help("Copy this error to the clipboard")
            .accessibilityLabel("Copy the error text")

            Button {
                onDismiss()
            } label: {
                Image(systemName: "xmark")
            }
            .buttonStyle(.borderless)
            .help("Dismiss this error")
            .accessibilityLabel("Dismiss the error")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(Color.red.opacity(0.10))
        .accessibilityElement(children: .contain)
    }
}

// MARK: - Gallery

@MainActor
private struct GalleryStrip: View {
    @Binding var selectedImageID: UUID?

    @Environment(AppStore.self) private var store

    private var items: [GeneratedImage] { sortedGallery(store.images) }

    var body: some View {
        Group {
            if items.isEmpty {
                Text("Results collect here.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 12)
            } else {
                ScrollView(.horizontal) {
                    LazyHStack(spacing: 8) {
                        ForEach(items) { item in
                            GalleryThumb(image: item,
                                         isSelected: item.id == (selectedImageID ?? items.first?.id),
                                         selectedImageID: $selectedImageID)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                }
            }
        }
        .frame(height: 108)
        .background(Color(nsColor: .underPageBackgroundColor))
    }
}

@MainActor
private struct GalleryThumb: View {
    let image: GeneratedImage
    let isSelected: Bool
    @Binding var selectedImageID: UUID?

    @Environment(AppStore.self) private var store

    private var url: URL { image.url(in: store.images.galleryDirectory) }

    var body: some View {
        let fileURL = url

        return Button {
            selectedImageID = image.id
        } label: {
            ZStack {
                Checkerboard(square: 6)
                ImageFileView(url: fileURL, label: "Result")
            }
            .frame(width: 84, height: 84)
            .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .strokeBorder(isSelected ? Color.accentColor : Color(nsColor: .separatorColor),
                                  lineWidth: isSelected ? 2 : 1)
            }
        }
        .buttonStyle(.plain)
        .help(shortHelp)
        .accessibilityLabel(shortHelp)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        // Drag straight out to Finder, Mail, or anything else that takes a file. The
        // URL is captured up front so the closure never has to touch actor state.
        .onDrag { NSItemProvider(contentsOf: fileURL) ?? NSItemProvider() }
        .contextMenu {
            GalleryMenu(image: image, selectedImageID: $selectedImageID)
        }
    }

    private var shortHelp: String {
        let prompt = image.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let head = prompt.isEmpty ? image.mode.label : String(prompt.prefix(120))
        return "\(head)\n\(image.width) × \(image.height) · seed \(image.seed)"
    }
}

/// The context menu shared by the canvas and the gallery strip.
@MainActor
private struct GalleryMenu: View {
    let image: GeneratedImage
    @Binding var selectedImageID: UUID?

    @Environment(AppStore.self) private var store

    private var images: ImageStore { store.images }
    private var url: URL { image.url(in: images.galleryDirectory) }

    var body: some View {
        Group {
            Button("Open in Preview") { ImageActions.open(url) }
            Button("Copy Image") { ImageActions.copy(url) }
            Button("Save As…") { ImageActions.saveAs(url, suggested: image.fileName) }
            Button("Reveal in Finder") { ImageActions.reveal(url) }
        }

        Divider()

        Group {
            Button("Use as Input") {
                images.inputImage = url
                images.mode = .edit
            }
            .help("Load this image into Edit as the starting point")

            Button("Use Seed") {
                images.params.seed = image.seed
            }
            .disabled(image.seed < 0)
            .help(image.seed < 0 ? "This image has no recorded seed" : "Set the seed to \(image.seed)")

            Button("Copy Prompt") {
                let board = NSPasteboard.general
                board.clearContents()
                board.setString(image.prompt, forType: .string)
            }
            .disabled(image.prompt.isEmpty)
        }

        Divider()

        Button("Delete", role: .destructive) {
            if selectedImageID == image.id { selectedImageID = nil }
            images.delete(image)
        }
    }
}

// MARK: - Image well

/// Drop target, file picker and paste target for one image, with a thumbnail and a
/// clear button once it holds something.
@MainActor
private struct ImageWell: View {
    @Binding var url: URL?
    let emptyTitle: String
    let emptySymbol: String
    let label: String

    @State private var isTargeted = false
    @State private var problem: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            well
                .frame(height: 118)
                .frame(maxWidth: .infinity)
                .background(Color(nsColor: .textBackgroundColor),
                            in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 9, style: .continuous)
                        .strokeBorder(isTargeted ? Color.accentColor : Color(nsColor: .separatorColor),
                                      style: StrokeStyle(lineWidth: isTargeted ? 2 : 1,
                                                         dash: url == nil ? [5, 4] : []))
                }
                .dropDestination(for: URL.self) { dropped, _ in
                    accept(dropped)
                } isTargeted: { targeted in
                    isTargeted = targeted
                }
                .contextMenu {
                    Button("Choose Image…") { choose() }
                    Button("Paste Image") { paste() }
                    if url != nil {
                        Divider()
                        Button("Clear", role: .destructive) { clear() }
                    }
                }
                .accessibilityElement(children: .contain)
                .accessibilityLabel(label)

            controls

            if let problem {
                ImageNotice(text: problem, systemImage: "exclamationmark.triangle.fill", tint: .orange)
            } else if let url {
                Text(url.lastPathComponent)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                    .help(url.path(percentEncoded: false))
            }
        }
    }

    private var controls: some View {
        HStack(spacing: 8) {
            Button("Choose Image…") { choose() }
                .controlSize(.small)
                .help("Pick an image file")

            // Deliberately a button and not a ⌘V shortcut: a view-wide paste shortcut
            // would steal ⌘V from the prompt editor sitting right next to it.
            Button {
                paste()
            } label: {
                Label("Paste", systemImage: "doc.on.clipboard")
            }
            .controlSize(.small)
            .help("Use the image on the clipboard")
            .accessibilityLabel("Paste an image from the clipboard")

            Spacer(minLength: 0)

            if url != nil {
                Button {
                    clear()
                } label: {
                    Label("Clear", systemImage: "xmark.circle")
                }
                .controlSize(.small)
                .help("Forget this image")
                .accessibilityLabel("Clear the \(label)")
            }
        }
    }

    @ViewBuilder
    private var well: some View {
        if let url {
            ZStack {
                Checkerboard(square: 7)
                ImageFileView(url: url, label: label)
                    .padding(6)
            }
            .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
        } else {
            VStack(spacing: 6) {
                Image(systemName: emptySymbol)
                    .font(.system(size: 22, weight: .light))
                    .foregroundStyle(.tertiary)
                    .accessibilityHidden(true)
                Text(emptyTitle)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Text("or drop, paste, or choose a file")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: Sources

    private func accept(_ dropped: [URL]) -> Bool {
        guard let found = dropped.first(where: ImageFiles.isImage) else {
            problem = dropped.isEmpty
                ? "Nothing usable in that drop."
                : "\(dropped[0].lastPathComponent) is not an image file."
            return false
        }
        url = found
        problem = nil
        return true
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.message = "Choose an image"
        panel.prompt = "Use Image"
        guard panel.runModal() == .OK, let picked = panel.url else { return }
        url = picked
        problem = nil
    }

    private func paste() {
        let board = NSPasteboard.general

        if let urls = board.readObjects(forClasses: [NSURL.self]) as? [URL],
           let found = urls.first(where: ImageFiles.isImage) {
            url = found
            problem = nil
            return
        }

        // A copy out of a browser or Preview arrives as raw bitmap data with no file
        // behind it, and both sd-cli and sd-server want a path — so stage it.
        guard let pasted = NSImage(pasteboard: board), let png = ImageFiles.pngData(from: pasted) else {
            problem = "The clipboard does not hold an image."
            return
        }
        guard let staged = ImageFiles.stage(png) else {
            problem = "Could not write the pasted image to a temporary file."
            return
        }
        url = staged
        problem = nil
    }

    private func clear() {
        url = nil
        problem = nil
    }
}

// MARK: - File-backed image

/// Loads a PNG off the main actor and draws it aspect-fit.
@MainActor
private struct ImageFileView: View {
    let url: URL
    let label: String

    @State private var image: NSImage?
    @State private var missing = false

    var body: some View {
        Group {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                    .accessibilityLabel(label)
            } else if missing {
                VStack(spacing: 5) {
                    Image(systemName: "questionmark.square.dashed")
                        .font(.title3)
                        .foregroundStyle(.tertiary)
                    Text("File is gone")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .accessibilityLabel("The image file is missing")
            } else {
                ProgressView()
                    .progressViewStyle(.circular)
                    .controlSize(.small)
                    .accessibilityHidden(true)
            }
        }
        .task(id: url) { await load() }
    }

    /// `Data` crosses actors, `NSImage` does not. Reading a multi-megabyte PNG on the
    /// main actor stalls the whole window, so only the decode happens here.
    private func load() async {
        image = nil
        missing = false
        let target = url
        let data = await Task.detached(priority: .userInitiated) {
            try? Data(contentsOf: target)
        }.value
        guard !Task.isCancelled else { return }
        guard let data, let decoded = NSImage(data: data) else {
            missing = true
            return
        }
        image = decoded
    }
}

// MARK: - Transparency checkerboard

/// The usual grid behind a transparent PNG. Both colours are semantic, so it stays
/// legible in light and dark.
private struct Checkerboard: View {
    var square: CGFloat = 9

    private static let base = Color(nsColor: .textBackgroundColor)
    private static let tile = Color.primary.opacity(0.06)

    var body: some View {
        Canvas(opaque: false) { [square] context, size in
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Self.base))
            var y: CGFloat = 0
            var row = 0
            while y < size.height {
                var x: CGFloat = row.isMultiple(of: 2) ? 0 : square
                while x < size.width {
                    context.fill(Path(CGRect(x: x, y: y, width: square, height: square)),
                                 with: .color(Self.tile))
                    x += square * 2
                }
                y += square
                row += 1
            }
        }
        .accessibilityHidden(true)
    }
}

// MARK: - Small pieces

private struct ImageSectionHeader: View {
    let title: String
    let help: String

    init(_ title: String, help: String) {
        self.title = title
        self.help = help
    }

    var body: some View {
        Text(title)
            .font(.subheadline.weight(.semibold))
            .help(help)
            .accessibilityAddTraits(.isHeader)
    }
}

private struct ImageNotice: View {
    let text: String
    let systemImage: String
    let tint: Color

    var body: some View {
        Label {
            Text(text)
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: systemImage)
                .foregroundStyle(tint)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Empty state

@MainActor
private struct ImagesEmptyState: View {
    let onShowDownloads: () -> Void

    @Environment(AppStore.self) private var store

    @State private var isRescanning = false

    /// The scanner walks the whole root, so this folder is a convention rather than a
    /// requirement — but it is the convention that keeps checkpoints out of the LLM
    /// list, because that scanner skips names beginning with an underscore.
    private var folder: URL {
        (ImagePaths.imageModelRoots.first ?? FileManager.default.homeDirectoryForCurrentUser)
            .appending(path: "_image-models", directoryHint: .isDirectory)
    }

    private var folderPath: String { folder.path(percentEncoded: false) }

    var body: some View {
        ContentUnavailableView {
            Label("No image models found", systemImage: "photo.on.rectangle.angled")
        } description: {
            VStack(spacing: 8) {
                Text("Nothing found. Checkpoints belong here:")
                Text(folderPath)
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .lineLimit(3)
                    .truncationMode(.middle)
                Text("A .gguf or .safetensors checkpoint gives you Generate and Edit; an ESRGAN .pth gives you Upscale. The leading underscore matters: the language-model scanner skips folders that start with one, which is what keeps these out of the Models list.")
                    .fixedSize(horizontal: false, vertical: true)
                Text("The Downloads section can fetch either one from HuggingFace.")
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: 460)
        } actions: {
            HStack(spacing: 10) {
                Button("Reveal in Finder") {
                    ImageActions.revealFolder(folder)
                }
                .help("Open \(folderPath) in Finder")

                Button("Get Models") {
                    onShowDownloads()
                }
                .help("Open the Downloads section")

                Button("Rescan") {
                    rescan()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(isRescanning)
                .help("Look through the image-models folder again")
            }
        }
    }

    private func rescan() {
        guard !isRescanning else { return }
        isRescanning = true
        Task {
            await store.images.rescan()
            isRescanning = false
        }
    }
}

// MARK: - File actions

@MainActor
private enum ImageActions {
    /// Preview rather than a Quick Look panel: `QLPreviewPanel` drives itself off the
    /// responder chain, and a plain SwiftUI hierarchy has no member in it to accept
    /// control — the panel would open blank and close again.
    static func open(_ url: URL) {
        NSWorkspace.shared.open(url)
    }

    static func reveal(_ url: URL) {
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    /// Selects the folder when it exists and its parent when it does not, so the
    /// button still lands somewhere useful before the first model is installed.
    static func revealFolder(_ url: URL) {
        if FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        } else {
            NSWorkspace.shared.activateFileViewerSelecting([url.deletingLastPathComponent()])
        }
    }

    static func copy(_ url: URL) {
        guard let image = NSImage(contentsOf: url) else { return }
        let board = NSPasteboard.general
        board.clearContents()
        board.writeObjects([image])
    }

    static func saveAs(_ url: URL, suggested: String) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = suggested.isEmpty ? "image.png" : suggested
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        let fm = FileManager.default
        // NSSavePanel asks about replacing but does not do the removal itself.
        try? fm.removeItem(at: destination)
        try? fm.copyItem(at: url, to: destination)
    }
}

// MARK: - Image file helpers

private enum ImageFiles {
    /// Extension match rather than a UTType query: a dropped URL often points at a
    /// file nothing has stat'd yet, and a wrong "no" here is worse than letting the
    /// backend reject an odd format later.
    static let extensions: Set<String> = [
        "png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "webp", "heic", "heif",
    ]

    static func isImage(_ url: URL) -> Bool {
        extensions.contains(url.pathExtension.lowercased())
    }

    static func pngData(from image: NSImage) -> Data? {
        guard let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff) else { return nil }
        return rep.representation(using: .png, properties: [:])
    }

    /// Pasted bitmaps have no file behind them and the backend wants a path. Reaping
    /// the temp directory is the OS's job.
    static func stage(_ data: Data) -> URL? {
        let fm = FileManager.default
        let directory = fm.temporaryDirectory.appending(path: "SolidChat-input", directoryHint: .isDirectory)
        try? fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appending(path: "pasted-\(UUID().uuidString).png", directoryHint: .notDirectory)
        do {
            try data.write(to: file, options: .atomic)
            return file
        } catch {
            return nil
        }
    }

    /// Dimensions straight out of the file header — no pixels are decoded, which is
    /// the whole point when the input is already an 8192² upscale.
    ///
    /// The keys are read as plain strings so the `kCGImageProperty…` globals never
    /// have to cross an isolation boundary: `kCGImagePropertyPixelWidth` *is*
    /// the string "PixelWidth".
    static func pixelSize(of url: URL) -> PixelSize? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any],
              let width = properties["PixelWidth"] as? Int,
              let height = properties["PixelHeight"] as? Int,
              width > 0, height > 0
        else { return nil }
        return PixelSize(width: width, height: height)
    }
}

// MARK: - Binding bridge

private extension Binding where Value == Int {
    /// `Slider` only speaks `Double`, and `ImageParams.steps` is an `Int`.
    ///
    /// `Binding`'s accessors are `@Sendable`, so a hand-written pair cannot touch
    /// main-actor state on its own account. SwiftUI never calls them from anywhere
    /// else, which is exactly what `assumeIsolated` is for — the alternative is a
    /// `Double` mirror of the field that has to be kept in sync by hand.
    var asDouble: Binding<Double> {
        Binding<Double>(
            get: { MainActor.assumeIsolated { Double(wrappedValue) } },
            set: { newValue in
                MainActor.assumeIsolated { wrappedValue = Int(newValue.rounded()) }
            }
        )
    }
}

// MARK: - Formatting

/// Deliberately separate from `Fmt` in Types.swift: this file must not add names to
/// a type other files also extend.
private enum ImgFmt {
    /// 7.0 prints as "7", 7.5 as "7.5" — a slider readout should not grow a
    /// meaningless ".0".
    static func number(_ value: Double) -> String {
        value == value.rounded() ? String(Int(value)) : String(format: "%.1f", value)
    }

    static func megapixels(_ width: Int, _ height: Int) -> String {
        let mp = Double(width) * Double(height) / 1_000_000
        return mp >= 100 ? String(format: "%.0f megapixels", mp) : String(format: "%.1f megapixels", mp)
    }
}
