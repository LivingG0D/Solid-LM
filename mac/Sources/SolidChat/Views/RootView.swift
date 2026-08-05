import Combine
import SwiftUI

// ---------------------------------------------------------------------------
// AppStore surface consumed by this file (AppStore lives in Core/Store.swift):
//
//   var  conversations: [Conversation]
//   var  selectedConversationID: UUID?        // settable — bound to the sidebar List
//   func newConversation()
//   func deleteConversation(_ id: UUID)
//   func unload()
//   var  engineState: EngineState
//   var  activeModel: LocalModel?
//   var  activeEngine: EngineKind?
//   let  engine: EngineManager                // .model / .engine, for the load phase
//   let  stats: SystemStats                   // .usedBytes / .totalBytes
//
// Detail views assumed to exist with no-argument initialisers, each reading
// AppStore out of the environment: ChatView, ImagesView, ModelsView,
// DownloadsView, LogsView.
// ChatView additionally takes an optional `onShowModels` closure.
// ---------------------------------------------------------------------------

// MARK: - Detail sections

private enum RootTab: String, CaseIterable, Identifiable, Hashable {
    case chat, images, models, downloads, logs

    var id: String { rawValue }

    var title: String {
        switch self {
        case .chat: return "Chat"
        case .images: return "Images"
        case .models: return "Models"
        case .downloads: return "Downloads"
        case .logs: return "Logs"
        }
    }

    /// The section's icon. macOS draws segmented controls title-only, so it does
    /// not show in the toolbar — it travels with the tab for anything else.
    var symbol: String {
        switch self {
        case .chat: return "bubble.left.and.bubble.right"
        case .images: return "photo.on.rectangle.angled"
        case .models: return "cube"
        case .downloads: return "arrow.down.circle"
        case .logs: return "text.alignleft"
        }
    }
}

// MARK: - Root

struct RootView: View {
    @Environment(AppStore.self) private var store

    @State private var tab: RootTab = .chat
    @State private var loadStart: Date?

    var body: some View {
        NavigationSplitView {
            ConversationSidebar()
                .navigationSplitViewColumnWidth(min: 210, ideal: 250, max: 340)
        } detail: {
            detail
                .frame(minWidth: 560, minHeight: 420)
                .navigationTitle("SolidChat")
                .navigationSubtitle(tab.title)
        }
        .toolbar {
            ToolbarItem(placement: .navigation) {
                modelStatus
            }
            ToolbarItem(placement: .principal) {
                tabPicker
            }
        }
        .onChange(of: store.engineState) { previous, current in
            if current.isLoading {
                if !previous.isLoading { loadStart = .now }
            } else {
                loadStart = nil
            }
        }
        .onAppear {
            if store.engineState.isLoading, loadStart == nil { loadStart = .now }
        }
        // Fallback path for anything that asks for the Models section without a
        // direct closure (see ChatViewSignals).
        .onReceive(NotificationCenter.default.publisher(for: ChatViewSignals.showModels)) { _ in
            tab = .models
        }
    }

    // MARK: Detail

    @ViewBuilder
    private var detail: some View {
        switch tab {
        case .chat: ChatView(onShowModels: { tab = .models })
        case .images: ImagesView()
        case .models: ModelsView()
        case .downloads: DownloadsView()
        case .logs: LogsView()
        }
    }

    // MARK: Toolbar — section picker

    private var tabPicker: some View {
        Picker("Section", selection: $tab) {
            ForEach(RootTab.allCases) { item in
                Label(item.title, systemImage: item.symbol).tag(item)
            }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .frame(minWidth: 360, idealWidth: 420)
        .help("Switch between chat, images, models, downloads and logs")
        .accessibilityLabel("Section")
    }

    // MARK: Toolbar — engine / model status

    /// `activeModel` is what the user asked for; `engine.model` is what the child
    /// is actually serving. Either one is enough to name the row.
    private var displayModel: LocalModel? {
        store.activeModel ?? store.engine.model
    }

    private var displayEngine: EngineKind? {
        store.activeEngine ?? store.engine.engine
    }

    @ViewBuilder
    private var modelStatus: some View {
        switch store.engineState {
        case .idle:
            Button {
                tab = .models
            } label: {
                Label("Select a Model", systemImage: "cube")
            }
            .help("No model is loaded — open the Models list")
            .accessibilityLabel("Select a model")

        case .loading:
            HStack(spacing: 7) {
                ProgressView()
                    .progressViewStyle(.circular)
                    .controlSize(.small)

                Text(displayModel?.name ?? "Loading model")
                    .lineLimit(1)
                    .truncationMode(.middle)

                if let loadStart {
                    TimelineView(.periodic(from: loadStart, by: 1)) { context in
                        Text(Self.elapsed(since: loadStart, now: context.date))
                            .font(.caption)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Loading \(displayModel?.name ?? "model")")

        case .ready:
            HStack(spacing: 7) {
                Circle()
                    .fill(Color.green)
                    .frame(width: 8, height: 8)
                    .accessibilityHidden(true)

                Text(displayModel?.name ?? "Ready")
                    .font(.system(.body, design: .monospaced))
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .accessibilityLabel("Loaded model \(displayModel?.name ?? "unknown")")

                if let engine = displayEngine {
                    StatusBadge(text: engine.label)
                }
                if let quant = displayModel?.quant, !quant.isEmpty {
                    StatusBadge(text: quant)
                }

                Button {
                    store.unload()
                } label: {
                    Image(systemName: "eject")
                }
                .buttonStyle(.borderless)
                .help("Unload model")
                .accessibilityLabel("Unload model")
            }

        case .failed(let message):
            Button {
                tab = .logs
            } label: {
                Label {
                    Text("Load failed")
                } icon: {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                }
            }
            .help(String(message.prefix(240)))
            .accessibilityLabel("Model load failed. Open the logs.")
        }
    }

    /// `Fmt.duration` is empty below one second; the toolbar wants something there.
    private static func elapsed(since start: Date, now: Date) -> String {
        let seconds = max(0, now.timeIntervalSince(start))
        let text = Fmt.duration(seconds)
        return text.isEmpty ? "0s" : text
    }
}

// MARK: - Small capsule badge

private struct StatusBadge: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.caption2)
            .monospacedDigit()
            .foregroundStyle(.secondary)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(.quaternary, in: Capsule())
            .accessibilityLabel(text)
    }
}

// MARK: - Sidebar

private struct ConversationSidebar: View {
    @Environment(AppStore.self) private var store

    var body: some View {
        @Bindable var store = store

        List(selection: $store.selectedConversationID) {
            Section("Chats") {
                if store.conversations.isEmpty {
                    Text("No chats yet")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .padding(.vertical, 4)
                } else {
                    ForEach(store.conversations) { conversation in
                        ConversationRow(conversation: conversation)
                            .tag(conversation.id)
                            .contextMenu {
                                Button("Delete Chat", role: .destructive) {
                                    store.deleteConversation(conversation.id)
                                }
                            }
                            .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                                Button(role: .destructive) {
                                    store.deleteConversation(conversation.id)
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                                .help("Delete this chat")
                            }
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .safeAreaInset(edge: .top, spacing: 0) {
            VStack(spacing: 0) {
                Button {
                    store.newConversation()
                } label: {
                    Label("New Chat", systemImage: "square.and.pencil")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .help("Start a new chat (⌘N)")
                .accessibilityLabel("New chat")
                .padding(.horizontal, 10)
                .padding(.vertical, 8)

                Divider()
            }
            .background(.bar)
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            VStack(spacing: 0) {
                Divider()
                MemoryGauge()
            }
            .background(.bar)
        }
    }
}

private struct ConversationRow: View {
    let conversation: Conversation

    private var displayTitle: String {
        conversation.title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "New Chat"
            : conversation.title
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(displayTitle)
                .lineLimit(1)
                .truncationMode(.tail)

            HStack(spacing: 5) {
                Text(conversation.updated, format: .relative(presentation: .named))
                if !conversation.modelID.isEmpty {
                    Text("·")
                    Text(conversation.modelID)
                        .lineLimit(1)
                        .truncationMode(.head)
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(displayTitle)
    }
}

// MARK: - Memory gauge

/// Unified memory pressure, sampled by `SystemStats` (active + wired + compressed,
/// which is what Activity Monitor calls "Memory Used").
private struct MemoryGauge: View {
    @Environment(AppStore.self) private var store
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var used: Int64 { store.stats.usedBytes }
    private var total: Int64 { store.stats.totalBytes }

    private var fraction: Double {
        guard total > 0 else { return 0 }
        return min(max(Double(used) / Double(total), 0), 1)
    }

    /// Accent under 70 %, amber under 85 %, red above — all semantic, so both
    /// appearances stay legible.
    private var tint: Color {
        switch fraction {
        case ..<0.70: return .accentColor
        case ..<0.85: return .orange
        default: return .red
        }
    }

    private var readout: String {
        "\(Fmt.bytes(used)) / \(Fmt.bytes(total))"
    }

    var body: some View {
        Gauge(value: fraction) {
            Text("Memory")
        } currentValueLabel: {
            Text(readout)
                .monospacedDigit()
        }
        .gaugeStyle(.linearCapacity)
        .tint(tint)
        .font(.caption)
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.25), value: fraction)
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .help("Unified memory in use: \(readout)")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Memory in use")
        .accessibilityValue(readout)
    }
}
