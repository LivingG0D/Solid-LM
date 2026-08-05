import Foundation
import Observation

// =============================================================================
// AppStore — the app's root state object.
//
// Written against the other files in this module:
//   Paths.appSupport / .modelsRoot
//   ModelScanner.scan(root:) -> [LocalModel]                (sync, off-actor safe)
//   ChatClient.stream(messages:system:samplers:engine:modelPath:modelID:)
//       -> AsyncThrowingStream<ChatDelta, Error>
//   ChatDelta { content, reasoning, completionTokens }
//   ThinkSplitter.feed(_:) -> (content:thinking:), .flush()
//   EngineManager { state, model, engine, load(_:engine:config:), unload() }
//   Downloader() / SystemStats().start()
// =============================================================================

@Observable
@MainActor
final class AppStore {

    // MARK: - Models

    var models: [LocalModel] = []

    /// LM Studio's own folder — shared, not copied. Repointed through
    /// `UserDefaults` (`Paths.Key.modelsRoot`), so it is read-only here.
    var modelsRoot: URL { Paths.modelsRoot }

    // MARK: - Conversations

    var conversations: [Conversation] = []
    var current = Conversation()

    // MARK: - Settings (persisted, debounced)

    /// Sampling parameters. Assigning schedules a debounced save, so views can
    /// bind straight to this during a slider drag.
    var samplers: Samplers {
        get {
            access(keyPath: \.samplers)
            return rawSamplers
        }
        set {
            withMutation(keyPath: \.samplers) { rawSamplers = newValue }
            scheduleSettingsSave()
        }
    }

    /// HuggingFace token, stored in plain text. Use a read-only token.
    var hfToken: String {
        get {
            access(keyPath: \.hfToken)
            return rawHFToken
        }
        set {
            withMutation(keyPath: \.hfToken) { rawHFToken = newValue }
            scheduleSettingsSave()
        }
    }

    /// Per-model load settings, keyed by `LocalModel.id`.
    var modelConfigs: [String: LoadConfig] {
        get {
            access(keyPath: \.modelConfigs)
            return rawModelConfigs
        }
        set {
            withMutation(keyPath: \.modelConfigs) { rawModelConfigs = newValue }
            scheduleSettingsSave()
        }
    }

    @ObservationIgnored private var rawSamplers = Samplers()
    @ObservationIgnored private var rawHFToken = ""
    @ObservationIgnored private var rawModelConfigs: [String: LoadConfig] = [:]

    // MARK: - Streaming

    var isStreaming = false
    /// Which conversation the in-flight generation belongs to. Views must ask
    /// `isStreaming(in:)` — a bare `isStreaming` makes a *different* chat's finished
    /// message render with a spinner while this one generates.
    private(set) var streamingConversationID: UUID?

    func isStreaming(in conversationID: UUID) -> Bool {
        isStreaming && streamingConversationID == conversationID
    }

    // MARK: - Sub-systems

    let engine = EngineManager()
    let images = ImageStore()
    let downloader = Downloader()
    let stats = SystemStats()
    let speech = SpeechEngine()

    // MARK: - Private state

    @ObservationIgnored private var generation: Task<Void, Never>?
    @ObservationIgnored private var generationToken = 0
    @ObservationIgnored private var settingsSave: Task<Void, Never>?
    @ObservationIgnored private var didBootstrap = false

    /// UI coalescing budget: ~20 redraws/sec. Fast models emit 200+ tok/s here
    /// and re-rendering per token stutters.
    @ObservationIgnored private let flushInterval: Duration = .milliseconds(50)

    init() {}

    // MARK: - Lifecycle

    /// Load persisted state, scan models, start the memory sampler.
    func bootstrap() {
        guard !didBootstrap else { return }
        didBootstrap = true

        StoreDisk.prepare()
        if let saved = StoreDisk.loadSettings() { applySettings(saved) }

        Task { [weak self] in
            let loaded = await Task.detached(priority: .utility) {
                StoreDisk.loadConversations()
            }.value
            guard let self else { return }
            // Anything created while the read was in flight wins.
            let live = Set(self.conversations.map(\.id))
            self.conversations = (self.conversations + loaded.filter { !live.contains($0.id) })
                .sorted { $0.updated > $1.updated }
        }

        Task { [weak self] in await self?.rescanModels() }

        // A finished download must surface the model regardless of which tab is open.
        downloader.onModelInstalled = { [weak self] in
            Task { await self?.rescanModels() }
        }

        images.bootstrap()
        // After applySettings, so the engine warms up on the voice the user chose.
        speech.bootstrap()
        // Armed last: bootstrap may settle on a default voice, and there is no
        // point scheduling a save for a value we just read off disk.
        watchSpeechSettings()
        stats.start()
    }

    /// Rescan the models folder. The scan is disk I/O, so it runs off the main
    /// actor and only the result comes back.
    func rescanModels() async {
        let root = Paths.modelsRoot
        models = await Task.detached(priority: .userInitiated) {
            ModelScanner.scan(root: root)
        }.value
    }

    // MARK: - Engine bridge

    var engineState: EngineState { engine.state }

    /// The model the engine is serving, if any.
    var activeModel: LocalModel? { engine.model }

    /// The engine serving `activeModel`.
    var activeEngine: EngineKind? { engine.engine }

    /// Load `model` on `kind`, using this model's saved load settings.
    func load(_ model: LocalModel, on kind: EngineKind) {
        stop()
        if current.messages.isEmpty { current.modelID = model.id }
        let settings = config(for: model.id)
        Task { await engine.load(model, engine: kind, config: settings) }
    }

    func unload() {
        stop()
        Task { await engine.unload() }
    }

    // MARK: - Conversations

    /// Sidebar selection. Reading gives the open chat; writing opens one.
    var selectedConversationID: UUID? {
        get { current.id }
        set { if let newValue { open(newValue) } }
    }

    func newConversation() {
        persistCurrent()
        current = Conversation(modelID: activeModel?.id ?? current.modelID)
    }

    func open(_ id: UUID) {
        guard id != current.id else { return }
        persistCurrent()
        if let found = conversations.first(where: { $0.id == id }) { current = found }
    }

    func delete(_ id: UUID) {
        conversations.removeAll { $0.id == id }
        Task.detached(priority: .utility) { StoreDisk.deleteConversation(id) }
        if current.id == id {
            current = Conversation(modelID: activeModel?.id ?? "")
        }
    }

    /// Sidebar-facing name for `delete(_:)`.
    func deleteConversation(_ id: UUID) { delete(id) }

    /// Save the open conversation if it has anything in it, and keep the
    /// sidebar ordered newest-first.
    func persistCurrent() {
        guard !current.messages.isEmpty else { return }

        if current.title.isEmpty { current.title = Self.title(for: current.messages) }
        current.updated = .now

        if let i = conversations.firstIndex(where: { $0.id == current.id }) {
            conversations[i] = current
        } else {
            conversations.append(current)
        }
        conversations.sort { $0.updated > $1.updated }

        let snapshot = current
        Task.detached(priority: .utility) { StoreDisk.save(snapshot) }
    }

    /// First ~48 characters of the first user message.
    static func title(for messages: [Msg]) -> String {
        guard let first = messages.first(where: { $0.role == .user })?.content else { return "New Chat" }
        let flat = first.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        if flat.isEmpty { return "New Chat" }
        if flat.count <= 48 { return flat }
        return String(flat.prefix(48)).trimmingCharacters(in: .whitespaces) + "…"
    }

    // MARK: - Per-model load config

    func config(for modelID: String) -> LoadConfig {
        modelConfigs[modelID] ?? LoadConfig()
    }

    func setConfig(_ c: LoadConfig, for modelID: String) {
        guard !modelID.isEmpty else { return }
        modelConfigs[modelID] = c
    }

    // MARK: - Chat

    func send(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isStreaming else { return }

        if current.modelID.isEmpty { current.modelID = activeModel?.id ?? "" }
        current.messages.append(Msg(role: .user, content: trimmed))
        if current.title.isEmpty { current.title = Self.title(for: current.messages) }
        generate()
    }

    /// Drop a trailing assistant turn and answer the same user turn again.
    func regenerate() {
        guard !isStreaming else { return }
        if current.messages.last?.role == .assistant { current.messages.removeLast() }
        guard current.messages.last?.role == .user else { return }
        generate()
    }

    /// Truncate the history at `index`, replace that turn with the edited text,
    /// and answer again.
    func editAndResend(at index: Int, text: String) {
        guard !isStreaming, current.messages.indices.contains(index) else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        current.messages.removeSubrange(index..<current.messages.count)
        current.messages.append(Msg(role: .user, content: trimmed))
        if index == 0 { current.title = Self.title(for: current.messages) }
        generate()
    }

    /// Cancel the in-flight response. What already streamed is kept.
    func stop() {
        generation?.cancel()
    }

    // MARK: - Generation

    private func generate() {
        generation?.cancel()
        generationToken &+= 1
        let token = generationToken

        // Drop assistant turns that produced no text (a failed or stopped generation).
        // Sending an empty assistant message back as history confuses every engine.
        let history = current.messages.filter {
            $0.role == .user || !$0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        let conversationID = current.id
        let sam = samplers

        let placeholder = Msg(role: .assistant, content: "")
        let messageID = placeholder.id
        current.messages.append(placeholder)

        guard engine.state.isReady else {
            let why = engine.state.errorText
                ?? "No model is loaded. Choose one in Models and load it."
            update(messageID, in: conversationID) { $0.error = why }
            persist(conversationID)
            return
        }

        let model = requestModel
        let kind = requestEngine(for: model)
        isStreaming = true
        streamingConversationID = conversationID

        // Inherits the main actor. Token bookkeeping happens here, but
        // observable state is only touched on the flush cadence below.
        generation = Task { [weak self] in
            guard let self else { return }

            var splitter = ThinkSplitter()
            var content = ""
            var thinking = ""
            var chunkCount = 0
            var reportedTokens: Int?
            var failure: String?

            let clock = ContinuousClock()
            let started = clock.now
            var firstToken: ContinuousClock.Instant?
            var lastFlush = started

            do {
                let stream = ChatClient.stream(messages: history,
                                               system: sam.system,
                                               samplers: sam,
                                               engine: kind,
                                               modelPath: model?.path ?? "",
                                               modelID: model?.id ?? conversationModelID(conversationID))

                for try await delta in stream {
                    if Task.isCancelled { break }

                    if let n = delta.completionTokens { reportedTokens = n }

                    let hasText = !delta.content.isEmpty || !delta.reasoning.isEmpty
                    guard hasText else { continue }

                    let now = clock.now
                    if firstToken == nil { firstToken = now }
                    chunkCount += 1

                    // Engines that report reasoning out-of-band feed .reasoning;
                    // ones that inline <think> feed it through the splitter.
                    if !delta.reasoning.isEmpty { thinking += delta.reasoning }
                    if !delta.content.isEmpty {
                        let piece = splitter.feed(delta.content)
                        content += piece.content
                        thinking += piece.thinking
                    }

                    if lastFlush.duration(to: now) >= self.flushInterval {
                        lastFlush = now
                        let visible = content
                        let thought = thinking
                        self.update(messageID, in: conversationID) {
                            $0.content = visible
                            $0.thinking = thought
                        }
                    }
                }
            } catch is CancellationError {
                // Stopped by the user: keep what arrived.
            } catch {
                failure = error.localizedDescription
            }

            let end = clock.now

            // Release anything the splitter was holding back as a possible tag.
            let tail = splitter.flush()
            content += tail.content
            thinking += tail.thinking

            let tokens = reportedTokens ?? chunkCount
            var measured: MsgStats?
            if let first = firstToken {
                let ttft = Self.seconds(started.duration(to: first))
                let span = Self.seconds(first.duration(to: end))
                measured = MsgStats(tokPerSec: span > 0 ? Double(tokens) / span : 0,
                                    ttft: ttft,
                                    tokens: tokens)
            }

            let visible = content
            let thought = thinking
            self.update(messageID, in: conversationID) { msg in
                msg.content = visible
                msg.thinking = thought
                msg.error = failure
                msg.stats = measured
            }

            // A newer generation may already own these.
            if self.generationToken == token {
                self.isStreaming = false
                self.streamingConversationID = nil
                self.generation = nil

                // Auto-speak the FINAL text, never the streaming partials: a later
                // chunk routinely revises a sentence that was already read out, and
                // the tail of a partial is usually half a word. Errors stay silent —
                // `msg.error` is not in `content`, so there is nothing to say. So does
                // a cancelled run: the user pressed Stop, reading the fragment aloud
                // is the opposite of what they asked for.
                if self.speech.settings.autoSpeak, failure == nil, !Task.isCancelled,
                   !visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    self.speech.toggle(visible, messageID: messageID)
                }
            }
            self.persist(conversationID)
        }
    }

    /// The model a request should be built from. `activeModel` is authoritative;
    /// the conversation's own id is the fallback after a relaunch.
    private var requestModel: LocalModel? {
        if let active = activeModel { return active }
        return models.first { $0.id == current.modelID }
    }

    /// ChatClient sends the *path* for MLX and vLLM — mlx_lm.server reads the
    /// `model` field as a HuggingFace repo id and re-downloads otherwise — so
    /// the engine kind has to be right even when nobody recorded it.
    private func requestEngine(for model: LocalModel?) -> EngineKind {
        if let active = activeEngine { return active }
        guard let model else { return .llama }
        if model.path.lowercased().hasSuffix(".gguf") { return .llama }
        return model.engines.first { $0 != .llama } ?? .mlx
    }

    private func conversationModelID(_ id: UUID) -> String {
        if current.id == id { return current.modelID }
        return conversations.first { $0.id == id }?.modelID ?? ""
    }

    /// Edit a message whether or not its conversation is still on screen.
    private func update(_ messageID: UUID, in conversationID: UUID, _ edit: (inout Msg) -> Void) {
        if current.id == conversationID,
           let i = current.messages.firstIndex(where: { $0.id == messageID }) {
            edit(&current.messages[i])
            return
        }
        if let c = conversations.firstIndex(where: { $0.id == conversationID }),
           let m = conversations[c].messages.firstIndex(where: { $0.id == messageID }) {
            edit(&conversations[c].messages[m])
        }
    }

    private func persist(_ conversationID: UUID) {
        if current.id == conversationID {
            persistCurrent()
            return
        }
        guard let i = conversations.firstIndex(where: { $0.id == conversationID }),
              !conversations[i].messages.isEmpty else { return }
        if conversations[i].title.isEmpty {
            conversations[i].title = Self.title(for: conversations[i].messages)
        }
        conversations[i].updated = .now
        let snapshot = conversations[i]
        conversations.sort { $0.updated > $1.updated }
        Task.detached(priority: .utility) { StoreDisk.save(snapshot) }
    }

    private static func seconds(_ d: Duration) -> Double {
        let c = d.components
        return Double(c.seconds) + Double(c.attoseconds) / 1e18
    }

    // MARK: - Settings persistence

    private func applySettings(_ s: StoreSettings) {
        withMutation(keyPath: \.samplers) { rawSamplers = s.samplers }
        withMutation(keyPath: \.hfToken) { rawHFToken = s.hfToken }
        withMutation(keyPath: \.modelConfigs) { rawModelConfigs = s.modelConfigs }
        speech.settings = s.speech
    }

    /// Speech settings live on `SpeechEngine` so views can bind straight to them,
    /// which means the store has to *notice* changes instead of intercepting them
    /// through a setter the way `samplers` does.
    ///
    /// `withObservationTracking` fires once and then stops, so the handler re-arms
    /// itself. It also fires *before* the write lands; hopping through a `Task`
    /// puts us after the mutation, so the payload snapshot sees the new value.
    private func watchSpeechSettings() {
        withObservationTracking {
            _ = speech.settings
        } onChange: { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                self.scheduleSettingsSave()
                self.watchSpeechSettings()
            }
        }
    }

    /// Coalesce writes so a slider drag does not hammer the disk.
    private func scheduleSettingsSave() {
        settingsSave?.cancel()
        let payload = StoreSettings(samplers: rawSamplers,
                                    hfToken: rawHFToken,
                                    modelConfigs: rawModelConfigs,
                                    speech: speech.settings)
        settingsSave = Task {
            try? await Task.sleep(for: .milliseconds(500))
            guard !Task.isCancelled else { return }
            await Task.detached(priority: .utility) { StoreDisk.save(payload) }.value
        }
    }

    /// Blocking save for `applicationWillTerminate`: a detached write would not
    /// survive the process exiting.
    func flushToDisk() {
        settingsSave?.cancel()
        StoreDisk.save(StoreSettings(samplers: rawSamplers,
                                     hfToken: rawHFToken,
                                     modelConfigs: rawModelConfigs,
                                     speech: speech.settings))
        // Above the early return: the gallery index has to land even when the
        // open chat is empty.
        images.flushToDisk()
        guard !current.messages.isEmpty else { return }
        if current.title.isEmpty { current.title = Self.title(for: current.messages) }
        current.updated = .now
        StoreDisk.save(current)
    }
}

// MARK: - settings.json payload

/// File-private so it cannot collide with anything another file declares.
private struct StoreSettings: Codable, Sendable {
    var samplers = Samplers()
    var hfToken = ""
    var modelConfigs: [String: LoadConfig] = [:]
    var speech = SpeechSettings()

    init(samplers: Samplers = Samplers(),
         hfToken: String = "",
         modelConfigs: [String: LoadConfig] = [:],
         speech: SpeechSettings = SpeechSettings()) {
        self.samplers = samplers
        self.hfToken = hfToken
        self.modelConfigs = modelConfigs
        self.speech = speech
    }

    /// Hand-written and entirely optional, because Swift's synthesised decoder
    /// throws `keyNotFound` for a missing key rather than falling back to the
    /// property's default. Verified, not assumed: a `settings.json` written before
    /// `speech` existed would fail to decode as a whole, `loadSettings()` would
    /// return nil, and the user would silently lose their samplers and HF token
    /// the first time they launched a build that knows about speech.
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        samplers = try c.decodeIfPresent(Samplers.self, forKey: .samplers) ?? Samplers()
        hfToken = try c.decodeIfPresent(String.self, forKey: .hfToken) ?? ""
        modelConfigs = try c.decodeIfPresent([String: LoadConfig].self, forKey: .modelConfigs) ?? [:]
        speech = try c.decodeIfPresent(SpeechSettings.self, forKey: .speech) ?? SpeechSettings()
    }
}

// MARK: - Disk

/// `~/Library/Application Support/SolidChat/`.
///
/// Everything here is synchronous, nonisolated and failure-tolerant: an
/// unreadable or corrupt file is skipped, never fatal, and every write is
/// atomic so a crash mid-save cannot leave a half-written chat behind.
private enum StoreDisk {

    /// Same folder the engine log lives in, so everything the app owns is
    /// together and moves together.
    static var root: URL { Paths.appSupport }

    static var conversationsDir: URL {
        root.appending(path: "conversations", directoryHint: .isDirectory)
    }

    static var settingsFile: URL {
        root.appending(path: "settings.json", directoryHint: .notDirectory)
    }

    static func prepare() {
        try? FileManager.default.createDirectory(at: conversationsDir,
                                                 withIntermediateDirectories: true)
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

    // Settings

    static func loadSettings() -> StoreSettings? {
        guard let data = try? Data(contentsOf: settingsFile) else { return nil }
        return try? decoder().decode(StoreSettings.self, from: data)
    }

    static func save(_ settings: StoreSettings) {
        prepare()
        guard let data = try? encoder().encode(settings) else { return }
        try? data.write(to: settingsFile, options: .atomic)
    }

    // Conversations

    static func loadConversations() -> [Conversation] {
        let dir = conversationsDir
        guard let names = try? FileManager.default
            .contentsOfDirectory(atPath: dir.path(percentEncoded: false)) else { return [] }

        let dec = decoder()
        var out: [Conversation] = []
        out.reserveCapacity(names.count)
        for name in names where name.hasSuffix(".json") {
            let url = dir.appending(path: name, directoryHint: .notDirectory)
            guard let data = try? Data(contentsOf: url),
                  let convo = try? dec.decode(Conversation.self, from: data) else { continue }
            out.append(convo)
        }
        return out.sorted { $0.updated > $1.updated }
    }

    static func save(_ conversation: Conversation) {
        prepare()
        guard let data = try? encoder().encode(conversation) else { return }
        let url = conversationsDir.appending(path: "\(conversation.id.uuidString).json",
                                             directoryHint: .notDirectory)
        try? data.write(to: url, options: .atomic)
    }

    static func deleteConversation(_ id: UUID) {
        let url = conversationsDir.appending(path: "\(id.uuidString).json",
                                             directoryHint: .notDirectory)
        try? FileManager.default.removeItem(at: url)
    }
}
