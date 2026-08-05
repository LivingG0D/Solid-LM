import AppKit
import SwiftUI

// ---------------------------------------------------------------------------
// AppStore surface consumed by this file (AppStore lives in Core/Store.swift):
//
//   var  models: [LocalModel]
//   var  modelsRoot: URL
//   func rescanModels() async
//   var  engineState: EngineState
//   var  activeModel: LocalModel?
//   var  activeEngine: EngineKind?
//   func load(_ model: LocalModel, on kind: EngineKind)
//   func unload()
//   func config(for modelID: String) -> LoadConfig
//   func setConfig(_ c: LoadConfig, for modelID: String)
//   let  engine: EngineManager               // .lastLoadError, for a load that was
//                                            // rejected while another model kept serving
// ---------------------------------------------------------------------------

// MARK: - Cross-view signals

enum ModelsViewSignals {
    /// Posted by the load-failure box's "Open Logs" button when no explicit
    /// handler was injected. `RootView` owns the section picker, so for the
    /// button to reach the Logs section it has to observe this:
    ///
    ///     .onReceive(NotificationCenter.default.publisher(for: ModelsViewSignals.showLogs)) { _ in
    ///         tab = .logs
    ///     }
    static let showLogs = Notification.Name("solidchat.showLogs")
}

// MARK: - Sort order

private enum ModelSortOrder: String, CaseIterable, Identifiable {
    case speed, size, name

    var id: String { rawValue }

    var title: String {
        switch self {
        case .speed: return "Estimated speed"
        case .size: return "Size"
        case .name: return "Name"
        }
    }
}

// MARK: - ModelsView

/// Everything found under the models folder: what fits, what it costs, and which
/// engines will serve it. Exactly one model is loaded at a time.
@MainActor
struct ModelsView: View {
    /// Optional injection point for "Open Logs". `RootView` calls `ModelsView()`,
    /// in which case the button falls back to `ModelsViewSignals.showLogs`.
    var onShowLogs: (() -> Void)?

    @Environment(AppStore.self) private var store

    @State private var query = ""
    @State private var sort: ModelSortOrder = .speed
    @State private var isRescanning = false

    /// The model the user last pressed Load on. `EngineManager` blanks its own
    /// `model` for the first moments of a swap and a launch that is rejected
    /// before anything is killed never sets it at all — without this the spinner
    /// and the error box would have no row to attach to.
    @State private var attemptedID: String?

    // MARK: Derived state

    private var visibleModels: [LocalModel] {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        var list = store.models

        if !needle.isEmpty {
            list = list.filter { model in
                model.id.lowercased().contains(needle)
                    || model.quant.lowercased().contains(needle)
                    || model.arch.lowercased().contains(needle)
            }
        }

        switch sort {
        case .speed:
            list.sort { lhs, rhs in
                lhs.estTokS == rhs.estTokS
                    ? lhs.id.localizedCaseInsensitiveCompare(rhs.id) == .orderedAscending
                    : lhs.estTokS > rhs.estTokS
            }
        case .size:
            list.sort { lhs, rhs in
                lhs.size == rhs.size
                    ? lhs.id.localizedCaseInsensitiveCompare(rhs.id) == .orderedAscending
                    : lhs.size > rhs.size
            }
        case .name:
            list.sort { $0.id.localizedCaseInsensitiveCompare($1.id) == .orderedAscending }
        }
        return list
    }

    private var modelsPath: String {
        store.modelsRoot.path(percentEncoded: false)
    }

    private var countText: String {
        let total = store.models.count
        let shown = visibleModels.count
        if shown == total { return total == 1 ? "1 model" : "\(total) models" }
        return "\(shown) of \(total) models"
    }

    /// A failed load, or a load that was refused before anything was killed —
    /// in the second case a model may still be happily serving.
    private var engineError: String? {
        if let text = store.engineState.errorText { return text }
        return store.engine.lastLoadError
    }

    private var errorOwnerID: String? {
        guard engineError != nil else { return nil }
        return attemptedID ?? store.activeModel?.id
    }

    /// True when the failure belongs to a row nobody can currently see.
    private var errorIsOrphaned: Bool {
        guard engineError != nil else { return false }
        guard let owner = errorOwnerID else { return true }
        return !visibleModels.contains { $0.id == owner }
    }

    private func errorText(for model: LocalModel) -> String? {
        guard let engineError, errorOwnerID == model.id else { return nil }
        return engineError
    }

    private func isLoadingRow(_ model: LocalModel) -> Bool {
        guard store.engineState.isLoading else { return false }
        if let attemptedID { return attemptedID == model.id }
        return store.activeModel?.id == model.id
    }

    private func loadedEngine(for model: LocalModel) -> EngineKind? {
        guard store.engineState.isReady, store.activeModel?.id == model.id else { return nil }
        return store.activeEngine
    }

    // MARK: Actions

    private func attemptLoad(_ model: LocalModel, on kind: EngineKind) {
        attemptedID = model.id
        store.load(model, on: kind)
    }

    private func rescan() {
        guard !isRescanning else { return }
        isRescanning = true
        Task {
            await store.rescanModels()
            isRescanning = false
        }
    }

    private func showLogs() {
        if let onShowLogs {
            onShowLogs()
        } else {
            NotificationCenter.default.post(name: ModelsViewSignals.showLogs, object: nil)
        }
    }

    private func revealModelsFolder() {
        let folder = store.modelsRoot
        let fm = FileManager.default
        if fm.fileExists(atPath: folder.path(percentEncoded: false)) {
            NSWorkspace.shared.activateFileViewerSelecting([folder])
        } else {
            NSWorkspace.shared.activateFileViewerSelecting([folder.deletingLastPathComponent()])
        }
    }

    // MARK: Body

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
        }
        .onChange(of: store.engineState) { _, state in
            // Unloading clears the attribution; a failure keeps it so the error
            // stays pinned to the row that caused it.
            if state == .idle { attemptedID = nil }
        }
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: 10) {
            searchField
                .frame(maxWidth: 260)

            Picker("Sort by", selection: $sort) {
                ForEach(ModelSortOrder.allCases) { order in
                    Text(order.title).tag(order)
                }
            }
            .pickerStyle(.menu)
            .fixedSize()
            .help("Order the list by estimated speed, size on disk, or name")

            Spacer(minLength: 8)

            if isRescanning {
                ProgressView()
                    .progressViewStyle(.circular)
                    .controlSize(.small)
                    .accessibilityLabel("Scanning the models folder")
            }

            Text(countText)
                .font(.callout)
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .accessibilityLabel(countText)

            Button {
                rescan()
            } label: {
                Label("Rescan", systemImage: "arrow.clockwise")
            }
            .disabled(isRescanning)
            .help("Look through \(modelsPath) again")
            .accessibilityLabel("Rescan the models folder")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .background(.bar)
    }

    private var searchField: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)

            TextField("Filter models", text: $query)
                .textFieldStyle(.plain)
                .accessibilityLabel("Filter models by name")

            if !query.isEmpty {
                Button {
                    query = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.borderless)
                .help("Clear the filter")
                .accessibilityLabel("Clear the filter")
            }
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 4)
        .background(Color(nsColor: .textBackgroundColor),
                    in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .strokeBorder(Color(nsColor: .separatorColor))
        }
    }

    // MARK: Content

    @ViewBuilder
    private var content: some View {
        if store.models.isEmpty {
            emptyState
        } else if visibleModels.isEmpty {
            ContentUnavailableView.search(text: query)
        } else {
            modelList
        }
    }

    private var modelList: some View {
        ScrollView {
            LazyVStack(spacing: 8) {
                if errorIsOrphaned, let engineError {
                    EngineErrorBox(title: "Load failed",
                                   text: engineError,
                                   onShowLogs: showLogs)
                        .padding(12)
                        .background(Color(nsColor: .controlBackgroundColor),
                                    in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .strokeBorder(Color.red.opacity(0.55))
                        }
                }

                ForEach(visibleModels) { model in
                    ModelRow(model: model,
                             isLoadingThisRow: isLoadingRow(model),
                             engineIsBusy: store.engineState.isLoading,
                             loadedEngine: loadedEngine(for: model),
                             errorText: errorText(for: model),
                             onLoad: { kind in attemptLoad(model, on: kind) },
                             onUnload: { store.unload() },
                             onShowLogs: showLogs)
                }
            }
            .padding(14)
        }
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label("No models found", systemImage: "shippingbox")
        } description: {
            VStack(spacing: 8) {
                Text("Nothing under this folder:")
                Text(modelsPath)
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .lineLimit(3)
                    .truncationMode(.middle)
                Text("Put a .gguf file or an MLX model directory there, laid out as publisher/model — or fetch one from HuggingFace in the Downloads section.")
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: 420)
        } actions: {
            HStack(spacing: 10) {
                Button("Reveal in Finder") {
                    revealModelsFolder()
                }
                .help("Open \(modelsPath) in Finder")

                Button("Rescan") {
                    rescan()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(isRescanning)
                .help("Look through the models folder again")
            }
        }
    }
}

// MARK: - Row

private struct ModelRow: View {
    let model: LocalModel
    let isLoadingThisRow: Bool
    let engineIsBusy: Bool
    let loadedEngine: EngineKind?
    let errorText: String?
    let onLoad: (EngineKind) -> Void
    let onUnload: () -> Void
    let onShowLogs: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var showingSettings = false

    private var isLoaded: Bool { loadedEngine != nil }

    // MARK: Body

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                FitDot(fit: model.fit)
                    .padding(.top, 2)

                VStack(alignment: .leading, spacing: 3) {
                    Text(model.name)
                        .font(.system(.body, design: .monospaced).weight(.medium))
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(detailHelp)

                    caption
                }

                Spacer(minLength: 12)

                controls
            }

            if let errorText {
                EngineErrorBox(title: "Load failed", text: errorText, onShowLogs: onShowLogs)
            }
        }
        .padding(12)
        .background(rowFill, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(isLoaded ? Color.accentColor : Color(nsColor: .separatorColor),
                              lineWidth: isLoaded ? 1.5 : 1)
        }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: isLoaded)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(accessibilitySummary)
    }

    private var rowFill: AnyShapeStyle {
        isLoaded
            ? AnyShapeStyle(Color.accentColor.opacity(0.12))
            : AnyShapeStyle(Color(nsColor: .controlBackgroundColor))
    }

    // MARK: Caption

    private var caption: some View {
        HStack(spacing: 6) {
            Text(model.publisher)
                .lineLimit(1)
                .truncationMode(.middle)

            separator
            QuantBadge(text: model.quant.isEmpty ? "unknown quant" : model.quant)
            separator

            Text(Fmt.bytes(model.size))
                .monospacedDigit()
                .help("Size on disk, all shards included")

            if model.estTokS > 0 {
                separator
                Text(estimateText)
                    .monospacedDigit()
                    .help("Estimate, not a measurement: derived from this model's size and the machine's memory bandwidth. Real throughput depends on context length, the engine and whatever else is using the GPU.")
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private var separator: some View {
        Text("·")
            .foregroundStyle(.tertiary)
            .accessibilityHidden(true)
    }

    private var estimateText: String {
        model.estTokS >= 10
            ? String(format: "~%.0f tok/s", model.estTokS)
            : String(format: "~%.1f tok/s", model.estTokS)
    }

    private var detailHelp: String {
        var parts = [model.id]
        if !model.arch.isEmpty { parts.append(model.arch) }
        parts.append(model.path)
        return parts.joined(separator: "\n")
    }

    private var accessibilitySummary: String {
        var text = "\(model.name), \(model.publisher), \(model.quant), \(Fmt.bytes(model.size)), \(model.fit.note)"
        if let loadedEngine { text += ", loaded on \(loadedEngine.label)" }
        return text
    }

    // MARK: Trailing controls

    @ViewBuilder
    private var controls: some View {
        HStack(spacing: 8) {
            if isLoadingThisRow {
                ProgressView()
                    .progressViewStyle(.circular)
                    .controlSize(.small)
                Text("Loading…")
                    .font(.caption)
                    .foregroundStyle(.secondary)

            } else if let loadedEngine {
                Text("Loaded · \(loadedEngine.label)")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(Color.accentColor)
                    .help("This model is serving on port 8181")

                Button("Unload") {
                    onUnload()
                }
                .controlSize(.small)
                .help("Stop the engine and free its memory")
                .accessibilityLabel("Unload \(model.name)")

            } else if model.engines.isEmpty {
                Text("No engine")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .help("Nothing on this machine can serve this file")

            } else {
                ForEach(model.engines, id: \.self) { kind in
                    Button("Load · \(kind.label)") {
                        onLoad(kind)
                    }
                    .controlSize(.small)
                    .disabled(engineIsBusy)
                    .help(loadHelp(for: kind))
                    .accessibilityLabel("Load \(model.name) on \(kind.label)")
                }
            }

            settingsButton
        }
        .buttonStyle(.bordered)
        .fixedSize()
    }

    private func loadHelp(for kind: EngineKind) -> String {
        var text = "Serve \(model.name) with \(kind.label) on port 8181. Whatever is loaded now is unloaded first."
        switch model.fit {
        case .fits: break
        case .tight: text += " Memory is tight for this one — close other apps first."
        case .tooBig: text += " This model is larger than usable memory and will most likely fail or crawl."
        }
        if kind == .vllm {
            text += " vLLM has no Metal backend on Apple Silicon — it would run on the CPU."
        }
        return text
    }

    private var settingsButton: some View {
        Button {
            showingSettings = true
        } label: {
            Image(systemName: "gearshape")
        }
        .buttonStyle(.borderless)
        .controlSize(.small)
        .help("Load settings for \(model.name): context length, GPU layers, KV cache")
        .accessibilityLabel("Load settings for \(model.name)")
        .popover(isPresented: $showingSettings) {
            ModelLoadSettings(model: model, isPresented: $showingSettings)
        }
    }
}

// MARK: - Fit indicator

private struct FitDot: View {
    let fit: FitClass

    private var color: Color {
        switch fit {
        case .fits: return .green
        case .tight: return .orange
        case .tooBig: return .red
        }
    }

    /// Shape as well as colour, so the state does not depend on colour alone.
    private var symbol: String {
        switch fit {
        case .fits: return "circle.fill"
        case .tight: return "exclamationmark.circle.fill"
        case .tooBig: return "xmark.circle.fill"
        }
    }

    private var explanation: String {
        switch fit {
        case .fits:
            return "Fits in memory: the weights plus a working KV cache leave room for the rest of the system."
        case .tight:
            return "Tight: it fits, but only just. Close other apps before loading, or lower the context length in this model's load settings."
        case .tooBig:
            return "Larger than usable memory. Loading it will either fail or swap so hard the model is unusable."
        }
    }

    var body: some View {
        Image(systemName: symbol)
            .font(.system(size: 11))
            .foregroundStyle(color)
            .frame(width: 13, height: 13)
            .help(explanation)
            .accessibilityLabel("Memory fit: \(fit.note)")
    }
}

// MARK: - Quantisation badge

private struct QuantBadge: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.caption2)
            .monospacedDigit()
            .foregroundStyle(.secondary)
            .padding(.horizontal, 5)
            .padding(.vertical, 1)
            .background(.quaternary, in: Capsule())
            .help("Quantisation: how heavily the weights were compressed")
            .accessibilityLabel("Quantisation \(text)")
    }
}

// MARK: - Failure box

private struct EngineErrorBox: View {
    let title: String
    let text: String
    let onShowLogs: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label {
                Text(title)
            } icon: {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
            }
            .font(.caption.weight(.semibold))

            ScrollView {
                Text(text)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
            }
            .frame(maxHeight: 130)
            .background(Color(nsColor: .textBackgroundColor),
                        in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(Color(nsColor: .separatorColor))
            }

            HStack(spacing: 8) {
                Button("Open Logs") {
                    onShowLogs()
                }
                .help("Show the engine's full output in the Logs section")
                .accessibilityLabel("Open the Logs section")

                Button {
                    let board = NSPasteboard.general
                    board.clearContents()
                    board.setString(text, forType: .string)
                } label: {
                    Label("Copy", systemImage: "doc.on.doc")
                }
                .help("Copy this error to the clipboard")
                .accessibilityLabel("Copy the error text")

                Spacer(minLength: 0)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .accessibilityElement(children: .contain)
    }
}

// MARK: - Per-model load settings

/// The gear popover. Edits a draft and only writes it back on Save, so a
/// half-typed context length never reaches the store.
private struct ModelLoadSettings: View {
    let model: LocalModel
    @Binding var isPresented: Bool

    @Environment(AppStore.self) private var store

    @State private var draft = LoadConfig()
    @State private var didAdopt = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(model.name)
                .font(.headline)
                .lineLimit(1)
                .truncationMode(.middle)
                .padding(.horizontal, 16)
                .padding(.top, 14)
                .padding(.bottom, 2)

            Text("Applies the next time this model is loaded.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 16)

            Form {
                Section {
                    LabeledContent("Context length") {
                        HStack(spacing: 6) {
                            TextField("Context length", value: $draft.ctx, format: .number)
                                .labelsHidden()
                                .multilineTextAlignment(.trailing)
                                .monospacedDigit()
                                .frame(width: 84)
                                .accessibilityLabel("Context length in tokens")

                            Stepper("Context length", value: $draft.ctx, in: 512...1_048_576, step: 2048)
                                .labelsHidden()
                                .accessibilityLabel("Adjust context length")
                        }
                    }
                    .help("Tokens the model can see at once. Every extra token costs KV-cache memory.")

                    LabeledContent("GPU layers") {
                        HStack(spacing: 6) {
                            TextField("GPU layers", value: $draft.gpuLayers, format: .number)
                                .labelsHidden()
                                .multilineTextAlignment(.trailing)
                                .monospacedDigit()
                                .frame(width: 84)
                                .accessibilityLabel("Number of layers offloaded to the GPU")

                            Stepper("GPU layers", value: $draft.gpuLayers, in: 0...999, step: 1)
                                .labelsHidden()
                                .accessibilityLabel("Adjust GPU layers")
                        }
                    }
                    .help("How many transformer layers are handed to the GPU. 999 means all of them.")
                } header: {
                    Text("Memory")
                } footer: {
                    Text("999 offloads every layer to the GPU. Full offload measured 37.9 tok/s on this machine against 3.99 tok/s with a partial auto ratio — nearly ten times faster, which is why it is the default. Lower it only for a model that will not otherwise fit.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Section {
                    Toggle("Flash attention", isOn: $draft.flashAttention)
                        .help("Fused attention kernels: less memory traffic, same output.")

                    Toggle("Quantise KV cache to Q8", isOn: $draft.quantizeKVCache)
                        .help("Stores keys and values at 8 bits, roughly halving KV-cache memory at a small quality cost.")
                } header: {
                    Text("Attention")
                } footer: {
                    Text("Both apply to llama.cpp. MLX and vLLM ignore them.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Section {
                    Toggle("Speculative decoding", isOn: $draft.speculativeDecoding)
                        .disabled(model.draftPath == nil)
                        .help(model.draftPath == nil
                              ? "This model has no mtp-*.gguf draft head beside it."
                              : "A small draft head proposes tokens that the full model verifies in one pass.")

                    if draft.speculativeDecoding, model.draftPath != nil {
                        Stepper("Draft tokens: \(draft.draftTokens)",
                                value: $draft.draftTokens, in: 1...8)
                            .help("Tokens drafted per verify pass. 3 measured fastest; 5 and above are slower than no drafting at all.")
                    }
                } header: {
                    Text("Speculative decoding")
                } footer: {
                    Text(model.draftPath == nil
                         ? "Only available when the publisher ships an mtp-*.gguf next to the model. Nothing else moves generation speed much — it is bound by memory bandwidth."
                         : "Measured on Gemma4-12B Q4_K_M: 14.1 tok/s off, 25.5 on — a 1.8× speedup for identical output. Costs about 0.25 GB for the draft head.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .formStyle(.grouped)

            Divider()

            HStack(spacing: 10) {
                Button("Reset") {
                    draft = LoadConfig()
                }
                .help("Back to the built-in defaults")

                Spacer(minLength: 0)

                Button("Cancel") {
                    isPresented = false
                }
                .keyboardShortcut(.cancelAction)

                Button("Save") {
                    store.setConfig(sanitised(draft), for: model.id)
                    isPresented = false
                }
                .keyboardShortcut(.defaultAction)
                .help("Store these settings for \(model.name)")
            }
            .padding(12)
        }
        .frame(width: 380)
        .frame(maxHeight: 520)
        .onAppear {
            guard !didAdopt else { return }
            didAdopt = true
            draft = store.config(for: model.id)
        }
    }

    /// A typed-in field can hold anything; the engine refuses a context of zero.
    private func sanitised(_ config: LoadConfig) -> LoadConfig {
        var out = config
        out.ctx = min(max(config.ctx, 512), 1_048_576)
        out.gpuLayers = min(max(config.gpuLayers, 0), 999)
        return out
    }
}
