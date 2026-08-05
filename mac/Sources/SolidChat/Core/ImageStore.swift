import AppKit
import Foundation
import Observation

// =============================================================================
// ImageStore — the image side of the app's state, shaped like AppStore.
//
// Written against the other files in this module:
//   ImagePaths.galleryDir / .imageModelRoots           (ImageEngine.swift)
//   ImageScanner.scan(roots:) -> [ImageModel]          (sync, off-actor safe)
//   ImageEngine { state: EngineState, model: ImageModel?,
//                 load(_:) async, unload() async }
//   ImageClient.txt2img(params:) async throws -> [Data]
//   ImageClient.img2img(params:initImage:mask:) async throws -> [Data]
//   ImageClient.upscale(input:model:repeats:tileSize:) async throws -> Data
//
// Every ImageClient reference is funnelled through the two call sites in
// generate() and the one in upscale(), so a signature that lands differently
// is a three-line fix rather than a rewrite.
// =============================================================================

@Observable
@MainActor
final class ImageStore {

    // MARK: - Models

    var models: [ImageModel] = []

    var diffusionModels: [ImageModel] { models.filter { $0.kind == .diffusion } }
    var upscalerModels: [ImageModel] { models.filter { $0.kind == .upscaler } }

    // MARK: - Composition

    /// Generation parameters. Assigning also records *which* of the arch-derived
    /// fields the user touched, so `applyDefaults(for:)` can leave them alone.
    var params: ImageParams {
        get {
            access(keyPath: \.params)
            return rawParams
        }
        set {
            let old = rawParams
            withMutation(keyPath: \.params) { rawParams = newValue }
            if !applyingDefaults { noteEdits(from: old, to: newValue) }
        }
    }

    var mode: ImageMode = .generate

    /// Source image for edit and upscale, and the optional inpaint mask.
    ///
    /// Held as file URLs rather than bytes: a 4K PNG is ~30 MB, and parking that
    /// in observable state means every unrelated redraw drags it along. The bytes
    /// are read once per job instead, off the main actor.
    var inputImage: URL?
    var maskImage: URL?

    // MARK: - Upscaling

    var upscaler: ImageModel?
    /// Each repeat is another 4× ESRGAN pass — 2 turns 512 into 8192.
    var upscaleRepeats: Int = 1
    /// 0 lets sd.cpp choose. Tiling trades speed for peak memory on large inputs.
    var upscaleTileSize: Int = 0

    // MARK: - Results

    /// Newest first.
    var gallery: [GeneratedImage] = []

    var isBusy = false
    var progress = ImageProgress()
    var lastError: String?

    // MARK: - Sub-systems

    let engine = ImageEngine()

    // MARK: - Constants

    /// sd-server reports `limits.max_batch_count: 8` in /sdcpp/v1/capabilities.
    nonisolated static let maxBatchCount = 8

    // MARK: - Private state

    @ObservationIgnored private var rawParams = ImageParams()

    /// Fields the user changed by hand since the current model was selected.
    @ObservationIgnored private var tuned: Set<TunedField> = []
    @ObservationIgnored private var tunedFor: String?
    @ObservationIgnored private var applyingDefaults = false

    @ObservationIgnored private var job: Task<Void, Never>?
    /// Bumped by every generate()/upscale()/cancel(). A task whose token is stale
    /// finishes silently instead of clobbering newer state.
    @ObservationIgnored private var jobToken = 0
    @ObservationIgnored private var ticker: Task<Void, Never>?
    @ObservationIgnored private var didBootstrap = false

    private enum TunedField: Hashable { case width, height, steps, cfg }

    init() {}

    // MARK: - Lifecycle

    /// Load the gallery index, then rescan the models folder.
    func bootstrap() {
        guard !didBootstrap else { return }
        didBootstrap = true

        let directory = galleryDirectory
        GalleryDisk.prepare(directory)

        Task { [weak self] in
            let loaded = await Task.detached(priority: .utility) {
                GalleryDisk.load(from: directory)
            }.value
            guard let self else { return }
            // Anything generated while the read was in flight wins.
            let live = Set(self.gallery.map(\.id))
            self.gallery = (self.gallery + loaded.entries.filter { !live.contains($0.id) })
                .sorted { $0.created > $1.created }
            // Only rewrite the index if entries were actually dropped.
            if loaded.dropped { self.persistGallery() }
        }

        Task { [weak self] in await self?.rescan() }
    }

    /// Rescan for image models. Disk I/O, so it runs off the main actor and only
    /// the result comes back.
    func rescan() async {
        let roots = ImagePaths.imageModelRoots
        models = await Task.detached(priority: .userInitiated) {
            ImageScanner.scan(roots: roots)
        }.value

        // Keep the upscaler picker pointed at something that still exists.
        if let chosen = upscaler, !models.contains(where: { $0.id == chosen.id }) {
            upscaler = nil
        }
        if upscaler == nil { upscaler = upscalerModels.first }
    }

    // MARK: - Engine bridge

    var state: EngineState { engine.state }
    /// Spelled out for callers that also hold an `AppStore`, where a bare
    /// `engineState` means the LLM engine.
    var engineState: EngineState { engine.state }

    /// The model sd-server is holding, if any.
    var loadedModel: ImageModel? { engine.model }

    /// Where the PNGs live. Views need this to resolve `GeneratedImage.url(in:)`.
    var galleryDirectory: URL { ImagePaths.galleryDir }

    var canRun: Bool {
        guard !isBusy else { return false }
        switch mode {
        case .generate:
            return engine.state.isReady && !rawParams.prompt.trimmed.isEmpty
        case .edit:
            return engine.state.isReady && inputImage != nil
        case .upscale:
            // An ESRGAN pass is a separate binary; no diffusion model need be loaded.
            return inputImage != nil && upscaler != nil
        }
    }

    func load(_ model: ImageModel) {
        cancel()
        applyDefaults(for: model)
        Task { await engine.load(model) }
    }

    func unload() {
        cancel()
        Task { await engine.unload() }
    }

    // MARK: - Defaults

    /// Re-point width/height/steps/CFG at what this architecture actually wants.
    ///
    /// Called when the user picks a model, because the alternative is a FLUX
    /// checkpoint quietly running at CFG 7 and producing sludge. Fields the user
    /// has edited *since selecting this same model* are left alone; picking a
    /// different model clears that record and applies the full set.
    func applyDefaults(for model: ImageModel) {
        // An ESRGAN network has no sampler, no CFG and no native resolution.
        guard model.kind == .diffusion else { return }

        if model.id != tunedFor {
            tunedFor = model.id
            tuned.removeAll()
        }

        var next = rawParams
        if !tuned.contains(.width) { next.width = model.arch.nativeSize }
        if !tuned.contains(.height) { next.height = model.arch.nativeSize }
        if !tuned.contains(.steps) { next.steps = model.arch.defaultSteps }
        if !tuned.contains(.cfg) { next.cfgScale = model.arch.defaultCFG }
        guard next != rawParams else { return }

        applyingDefaults = true
        params = next
        applyingDefaults = false
    }

    private func noteEdits(from old: ImageParams, to new: ImageParams) {
        if new.width != old.width { tuned.insert(.width) }
        if new.height != old.height { tuned.insert(.height) }
        if new.steps != old.steps { tuned.insert(.steps) }
        if new.cfgScale != old.cfgScale { tuned.insert(.cfg) }
    }

    // MARK: - Running

    /// The primary action. Dispatches on `mode` so the view needs one button.
    func run() {
        switch mode {
        case .generate, .edit:
            generate()
        case .upscale:
            guard let source = inputImage else {
                lastError = "Choose an image to upscale first."
                return
            }
            guard let model = upscaler else {
                lastError = upscalerModels.isEmpty
                    ? "No upscaler found. Put an ESRGAN .pth in your image models folder."
                    : "Choose an upscaler first."
                return
            }
            upscale(input: source, using: model)
        }
    }

    /// txt2img, or img2img when `mode == .edit` and an input image is set.
    func generate() {
        guard !isBusy else { return }
        guard engine.state.isReady else {
            lastError = engine.state.errorText
                ?? "No image model is loaded. Choose one and load it first."
            return
        }

        let base = rawParams
        switch mode {
        case .generate:
            guard !base.prompt.trimmed.isEmpty else {
                lastError = "Write a prompt first."
                return
            }
        case .edit:
            guard inputImage != nil else {
                lastError = "Choose an image to edit first."
                return
            }
        case .upscale:
            // Upscaling needs an ESRGAN network, not the diffusion model this path drives.
            lastError = "Pick an image and an upscaler, then run Upscale."
            return
        }

        let token = beginJob(stage: "Preparing")

        let count = max(1, min(base.batchCount, Self.maxBatchCount))
        let modelName = engine.model?.name ?? ""
        let resultMode: ImageMode = (mode == .edit) ? .edit : .generate
        let inputURL = inputImage
        let maskURL = maskImage
        let directory = galleryDirectory

        job = Task { [weak self] in
            guard let self else { return }

            // Read the edit inputs once per job rather than once per image.
            var input: Data?
            var mask: Data?
            if resultMode == .edit {
                guard let loaded = await Self.readBytes(inputURL) else {
                    guard self.jobToken == token else { return }
                    self.finish(error: "Could not read \(inputURL?.lastPathComponent ?? "the input image").")
                    return
                }
                input = loaded
                if let maskURL {
                    guard let loadedMask = await Self.readBytes(maskURL) else {
                        guard self.jobToken == token else { return }
                        self.finish(error: "Could not read \(maskURL.lastPathComponent).")
                        return
                    }
                    mask = loadedMask
                }
            }

            let clock = ContinuousClock()
            var failure: String?

            for index in 0..<count {
                if Task.isCancelled { break }
                self.progress.stage = count > 1 ? "Generating \(index + 1) of \(count)" : "Generating"

                // One request per image with an explicit seed: a batch left to the
                // server comes back with no way to tell which seed made which image,
                // and a seed you cannot reproduce is not worth recording.
                var request = base
                request.batchCount = 1
                request.seed = Self.seed(from: base.seed, index: index)

                let started = clock.now
                do {
                    let pngs: [Data]
                    if let input {
                        pngs = try await ImageClient.img2img(params: request, initImage: input, mask: mask)
                    } else {
                        pngs = try await ImageClient.txt2img(params: request)
                    }
                    let elapsed = Self.seconds(started.duration(to: clock.now))
                    await self.record(pngs,
                                      request: request,
                                      mode: resultMode,
                                      modelName: modelName,
                                      elapsed: elapsed,
                                      into: directory)
                } catch is CancellationError {
                    break
                } catch {
                    failure = Self.describe(error)
                    break
                }
            }

            guard self.jobToken == token else { return }   // cancelled, or a newer job owns this
            self.finish(error: failure)
        }
    }

    /// ESRGAN pass over an image already on disk. No diffusion model involved, so
    /// this works even when nothing is loaded for generation.
    func upscale(input: URL, using model: ImageModel) {
        guard !isBusy else { return }
        guard model.kind == .upscaler else {
            lastError = "\(model.name) is not an upscaler."
            return
        }
        guard FileManager.default.fileExists(atPath: input.path(percentEncoded: false)) else {
            lastError = "\(input.lastPathComponent) is no longer on disk."
            return
        }

        let token = beginJob(stage: "Upscaling")
        let directory = galleryDirectory
        let repeats = max(1, min(upscaleRepeats, 4))
        let tileSize = max(0, upscaleTileSize)

        job = Task { [weak self] in
            guard let self else { return }
            let clock = ContinuousClock()
            let started = clock.now
            var failure: String?

            do {
                let png = try await ImageClient.upscale(input: input,
                                                        model: model,
                                                        repeats: repeats,
                                                        tileSize: tileSize)
                let elapsed = Self.seconds(started.duration(to: clock.now))

                // Nothing here is sampled: no prompt, no sampler, and the result is
                // deterministic, so seed 0 reads as "not applicable" rather than
                // "random" — which is what -1 means everywhere else in this file.
                var request = ImageParams()
                request.prompt = ""
                request.negativePrompt = ""
                request.steps = 0
                request.cfgScale = 0
                request.seed = 0
                request.sampler = ""
                await self.record([png],
                                  request: request,
                                  mode: .upscale,
                                  modelName: model.name,
                                  elapsed: elapsed,
                                  into: directory)
            } catch is CancellationError {
                // Nothing to keep.
            } catch {
                failure = Self.describe(error)
            }

            guard self.jobToken == token else { return }
            self.finish(error: failure)
        }
    }

    /// Abandon the running job.
    ///
    /// sd-server reports `features.cancel_generating: false`, so an image already
    /// on the GPU finishes there regardless — we stop waiting for it and stop
    /// issuing the rest of the batch. Anything already written to disk is kept,
    /// because the index and the folder have to agree.
    func cancel() {
        guard isBusy else { return }
        job?.cancel()
        job = nil
        jobToken &+= 1          // the abandoned task must not run finish()
        stopTicker()
        isBusy = false
        progress = ImageProgress()
    }

    func clearError() { lastError = nil }

    // MARK: - Gallery

    func deleteImage(_ id: UUID) {
        guard let index = gallery.firstIndex(where: { $0.id == id }) else { return }
        let removed = gallery.remove(at: index)
        let url = removed.url(in: galleryDirectory)
        Task.detached(priority: .utility) { GalleryDisk.remove(url) }
        persistGallery()

        // A deleted file must not stay wired up as the next job's input.
        if inputImage == url { inputImage = nil }
        if maskImage == url { maskImage = nil }
    }

    func delete(_ image: GeneratedImage) { deleteImage(image.id) }

    func revealInFinder(_ image: GeneratedImage) {
        NSWorkspace.shared.activateFileViewerSelecting([image.url(in: galleryDirectory)])
    }

    /// Blocking save for `applicationWillTerminate`. The detached write that
    /// `persistGallery()` schedules would not outlive the process, which would
    /// strand the last image of the session as a PNG with no index entry.
    func flushToDisk() {
        GalleryDisk.save(gallery, in: galleryDirectory)
    }

    // MARK: - Job plumbing

    private func beginJob(stage: String) -> Int {
        job?.cancel()
        jobToken &+= 1
        isBusy = true
        lastError = nil
        // totalSteps stays 0, which is the contract for "no real step counter":
        // sd-server exposes none. Verified against GET /sdcpp/v1/jobs/<id>, which
        // returns only status (queued / generating / completed / failed),
        // queue_position and timestamps — polled once a second across a 30-step
        // 512x512 run, it never once reported a step. A non-zero totalSteps here
        // makes the view draw "step 0 of 20" and a frozen bar for the whole run,
        // which is a worse lie than an honest spinner. If a future client learns
        // real per-step progress, set both fields together and the view's
        // determinate branch starts working with no change there.
        progress = ImageProgress(step: 0, totalSteps: 0, stage: stage, elapsed: 0)
        startTicker()
        return jobToken
    }

    private func finish(error: String?) {
        stopTicker()
        job = nil
        isBusy = false
        progress = ImageProgress()
        if let error { lastError = error }
    }

    /// The only honest progress signal we have: wall-clock time since the job started.
    private func startTicker() {
        ticker?.cancel()
        let clock = ContinuousClock()
        let started = clock.now
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(200))
                guard !Task.isCancelled, let self else { return }
                self.progress.elapsed = Self.seconds(started.duration(to: clock.now))
            }
        }
    }

    private func stopTicker() {
        ticker?.cancel()
        ticker = nil
    }

    /// Write the PNGs, then add their metadata. Disk first: a row with no file
    /// behind it is a broken thumbnail that survives every relaunch.
    private func record(_ pngs: [Data],
                        request: ImageParams,
                        mode: ImageMode,
                        modelName: String,
                        elapsed: TimeInterval,
                        into directory: URL) async {
        var added = false

        for png in pngs {
            if Task.isCancelled { break }

            let id = UUID()
            let fileName = "\(id.uuidString).png"
            let url = directory.appending(path: fileName, directoryHint: .notDirectory)

            let wrote = await Task.detached(priority: .utility) {
                GalleryDisk.writePNG(png, to: url)
            }.value
            guard wrote else {
                lastError = "Could not write to \(directory.path(percentEncoded: false))."
                continue
            }

            // What the server produced, not what we asked for: sd.cpp rounds
            // dimensions to a multiple of 64, and an upscale has no requested size.
            let measured = Self.pngSize(png)
            gallery.insert(GeneratedImage(id: id,
                                          fileName: fileName,
                                          mode: mode,
                                          prompt: request.prompt,
                                          negativePrompt: request.negativePrompt,
                                          modelName: modelName,
                                          width: measured?.width ?? request.width,
                                          height: measured?.height ?? request.height,
                                          steps: request.steps,
                                          cfgScale: request.cfgScale,
                                          seed: request.seed,
                                          sampler: request.sampler,
                                          elapsed: elapsed),
                           at: 0)
            added = true
        }

        if added { persistGallery() }
    }

    private func persistGallery() {
        let snapshot = gallery
        let directory = galleryDirectory
        Task.detached(priority: .utility) { GalleryDisk.save(snapshot, in: directory) }
    }

    // MARK: - Helpers

    nonisolated private static func readBytes(_ url: URL?) async -> Data? {
        guard let url else { return nil }
        return await Task.detached(priority: .utility) { try? Data(contentsOf: url) }.value
    }

    /// `-1` asks sd.cpp for a random seed but it never tells us which one it used,
    /// so we choose it here and send it explicitly. Bounded to 32 bits because
    /// several samplers seed a 32-bit RNG from it and a wider value would not
    /// round-trip. A fixed seed walks per image — otherwise a batch of 4 is the
    /// same picture four times, which is what sd.cpp does for a batch anyway.
    nonisolated private static func seed(from requested: Int64, index: Int) -> Int64 {
        requested >= 0 ? requested &+ Int64(index) : Int64.random(in: 1...0xFFFF_FFFF)
    }

    /// Width and height straight out of the PNG's IHDR chunk — exact, and without
    /// decoding several megabytes of pixels just to read two integers.
    nonisolated private static func pngSize(_ data: Data) -> (width: Int, height: Int)? {
        guard data.count >= 24 else { return nil }
        let head = [UInt8](data.prefix(24))
        guard head[0] == 0x89, head[1] == 0x50, head[2] == 0x4E, head[3] == 0x47,
              head[4] == 0x0D, head[5] == 0x0A, head[6] == 0x1A, head[7] == 0x0A,
              head[12] == 0x49, head[13] == 0x48, head[14] == 0x44, head[15] == 0x52   // "IHDR"
        else { return nil }

        func be32(_ offset: Int) -> Int {
            (Int(head[offset]) << 24) | (Int(head[offset + 1]) << 16)
                | (Int(head[offset + 2]) << 8) | Int(head[offset + 3])
        }
        let width = be32(16)
        let height = be32(20)
        guard width > 0, height > 0 else { return nil }
        return (width, height)
    }

    nonisolated private static func seconds(_ d: Duration) -> Double {
        let c = d.components
        return Double(c.seconds) + Double(c.attoseconds) / 1e18
    }

    nonisolated private static func describe(_ error: Error) -> String {
        if let localized = error as? LocalizedError,
           let text = localized.errorDescription, !text.isEmpty {
            return text
        }
        return (error as NSError).localizedDescription
    }
}

// MARK: - String convenience

private extension String {
    var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
}

// MARK: - Disk

/// `~/Library/Application Support/SolidChat/gallery/` — `<uuid>.png` files beside
/// an `index.json` of `GeneratedImage`.
///
/// File-private, synchronous, nonisolated and failure-tolerant: a corrupt index is
/// treated as an empty one rather than taken as fatal, and every write is atomic so
/// a crash mid-save cannot leave half a gallery behind. Directories arrive as
/// arguments so nothing here has to reach for `ImagePaths` off the main actor.
private enum GalleryDisk {

    static func indexFile(in directory: URL) -> URL {
        directory.appending(path: "index.json", directoryHint: .notDirectory)
    }

    static func prepare(_ directory: URL) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    private static func encoder() -> JSONEncoder {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        return e
    }

    private static func decoder() -> JSONDecoder {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }

    /// Returns the surviving entries, newest first, and whether any were dropped.
    /// An entry whose PNG has gone is a blank tile forever, so it does not survive.
    static func load(from directory: URL) -> (entries: [GeneratedImage], dropped: Bool) {
        guard let data = try? Data(contentsOf: indexFile(in: directory)),
              let saved = try? decoder().decode([GeneratedImage].self, from: data)
        else { return ([], false) }

        let fm = FileManager.default
        let kept = saved.filter {
            fm.fileExists(atPath: $0.url(in: directory).path(percentEncoded: false))
        }
        return (kept.sorted { $0.created > $1.created }, kept.count != saved.count)
    }

    static func save(_ entries: [GeneratedImage], in directory: URL) {
        prepare(directory)
        guard let data = try? encoder().encode(entries) else { return }
        try? data.write(to: indexFile(in: directory), options: .atomic)
    }

    /// True only when the bytes are on disk. `.atomic` writes a sibling temp file
    /// first, so the directory has to exist before we start.
    static func writePNG(_ data: Data, to url: URL) -> Bool {
        prepare(url.deletingLastPathComponent())
        do {
            try data.write(to: url, options: .atomic)
            return true
        } catch {
            return false
        }
    }

    static func remove(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }
}
