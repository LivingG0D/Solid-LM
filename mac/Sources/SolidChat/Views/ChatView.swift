import AppKit
import SwiftUI

// MARK: - Cross-view signals

enum ChatViewSignals {
    /// Posted by the chat empty state's "Browse Models" button when no explicit
    /// handler was injected. `RootView` owns the section picker, so it has to
    /// observe this to switch to the Models section:
    ///
    ///     .onReceive(NotificationCenter.default.publisher(for: ChatViewSignals.showModels)) { _ in
    ///         tab = .models
    ///     }
    static let showModels = Notification.Name("solidchat.showModels")
}

// MARK: - ChatView

/// The conversation surface: transcript on top, composer pinned to the bottom.
@MainActor
struct ChatView: View {
    /// Optional injection point for the "Browse Models" button. `RootView` calls
    /// `ChatView()`, in which case the button falls back to `ChatViewSignals.showModels`.
    var onShowModels: (() -> Void)?

    @Environment(AppStore.self) private var store
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var draft: String = ""
    @State private var isNearBottom: Bool = true
    @State private var editingMessageID: UUID?

    private static let bottomAnchorID = "solidchat.transcript.bottom"

    // MARK: Store surface (every use of AppStore lives in this one block)

    private var messages: [Msg] { store.current.messages }
    /// Scoped to the conversation on screen. `store.isStreaming` alone would put a
    /// spinner on this chat's finished message while a different chat generates.
    private var isStreaming: Bool { store.isStreaming(in: store.current.id) }
    /// Any generation anywhere — there is one engine, so the composer must block on it.
    private var engineBusy: Bool { store.isStreaming }
    private var modelReady: Bool { store.engineState.isReady }
    private var modelName: String { store.activeModel?.name ?? "" }

    private func sendDraft() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, modelReady, !engineBusy else { return }
        draft = ""
        isNearBottom = true
        store.send(text)
    }

    private func stop() { store.stop() }

    private func regenerate() {
        isNearBottom = true
        store.regenerate()
    }

    /// `AppStore.editAndResend` truncates by index, so resolve the row's index
    /// against the live transcript rather than trusting a captured position.
    private func resend(_ message: Msg, text: String) {
        editingMessageID = nil
        guard let index = messages.firstIndex(where: { $0.id == message.id }) else { return }
        isNearBottom = true
        store.editAndResend(at: index, text: text)
    }

    private func showModels() {
        if let onShowModels {
            onShowModels()
        } else {
            NotificationCenter.default.post(name: ChatViewSignals.showModels, object: nil)
        }
    }

    /// The speech engine's state as it applies to *one* row. There is a single
    /// synthesiser, so every message other than the one being spoken is idle —
    /// without this scoping every row would sprout a stop button at once.
    private func speechState(for message: Msg) -> SpeechState {
        store.speech.speakingMessageID == message.id ? store.speech.state : .idle
    }

    private func speak(_ message: Msg) {
        store.speech.toggle(message.content, messageID: message.id)
    }

    /// Hand-made rather than `@Bindable`: `speech` is a separate observable object
    /// hanging off the store, and the composer should not have to know that.
    private var autoSpeak: Binding<Bool> {
        Binding(get: { store.speech.settings.autoSpeak },
                set: { store.speech.settings.autoSpeak = $0 })
    }

    // MARK: Body

    var body: some View {
        VStack(spacing: 0) {
            if messages.isEmpty {
                ChatEmptyState(modelReady: modelReady, modelName: modelName, onOpenModels: showModels)
            } else {
                transcript
            }
            Divider()
            ChatComposer(
                text: $draft,
                isStreaming: engineBusy,
                modelReady: modelReady,
                modelName: modelName,
                autoSpeak: autoSpeak,
                onSend: sendDraft,
                onStop: stop
            )
        }
        .background(.background)
    }

    // MARK: Transcript

    private var transcript: some View {
        GeometryReader { geo in
            ScrollViewReader { proxy in
                ScrollView(.vertical) {
                    LazyVStack(alignment: .leading, spacing: 22) {
                        ForEach(messages) { message in
                            ChatMessageRow(
                                message: message,
                                maxBubbleWidth: bubbleWidth(for: geo.size.width),
                                isStreaming: isStreamingMessage(message),
                                canRegenerate: canRegenerate(message),
                                isEditing: editingMessageID == message.id,
                                speechState: speechState(for: message),
                                beginEditing: { editingMessageID = message.id },
                                cancelEditing: { editingMessageID = nil },
                                onRegenerate: regenerate,
                                onSaveEdit: { newText in resend(message, text: newText) },
                                onSpeak: { speak(message) }
                            )
                            .id(message.id)
                        }
                        Color.clear
                            .frame(height: 1)
                            .id(Self.bottomAnchorID)
                            .accessibilityHidden(true)
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 20)
                    .padding(.bottom, 14)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .defaultScrollAnchor(.bottom, for: .initialOffset)
                // Track whether the reader is parked at the bottom. If they scrolled up
                // to read, streaming tokens must not yank the view back down.
                .onScrollGeometryChange(for: Bool.self) { scroll in
                    scroll.visibleRect.maxY >= scroll.contentSize.height - 80
                } action: { _, near in
                    if isNearBottom != near { isNearBottom = near }
                }
                // New message appended: follow with a short animation.
                .onChange(of: messages.count) { _, _ in
                    guard isNearBottom else { return }
                    scrollToBottom(proxy, animated: !reduceMotion)
                }
                // Tokens arriving inside the last message: follow without animating
                // (animating every token is both janky and expensive).
                .onChange(of: streamSignature) { _, _ in
                    guard isNearBottom else { return }
                    scrollToBottom(proxy, animated: false)
                }
                .overlay(alignment: .bottomTrailing) {
                    if !isNearBottom {
                        Button {
                            isNearBottom = true
                            scrollToBottom(proxy, animated: !reduceMotion)
                        } label: {
                            Image(systemName: "arrow.down")
                                .font(.callout.weight(.semibold))
                                .padding(9)
                                .background(.regularMaterial, in: Circle())
                                .overlay(Circle().strokeBorder(.separator))
                        }
                        .buttonStyle(.plain)
                        .padding(16)
                        .help("Scroll to the latest message")
                        .accessibilityLabel("Scroll to the latest message")
                    }
                }
            }
        }
    }

    // MARK: Helpers

    private func bubbleWidth(for containerWidth: CGFloat) -> CGFloat {
        max(220, (containerWidth - 40) * 0.78)
    }

    private func isStreamingMessage(_ message: Msg) -> Bool {
        isStreaming && message.role == .assistant && message.id == messages.last?.id
    }

    private func canRegenerate(_ message: Msg) -> Bool {
        !isStreaming && message.role == .assistant && message.id == messages.last?.id
    }

    /// Cheap value that changes whenever the tail of the transcript grows.
    private var streamSignature: Int {
        guard let last = messages.last else { return 0 }
        return (messages.count &* 1_000_003) &+ (last.content.count &* 31) &+ last.thinking.count
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy, animated: Bool) {
        if animated {
            withAnimation(Animation.easeOut(duration: 0.18)) {
                proxy.scrollTo(Self.bottomAnchorID, anchor: .bottom)
            }
        } else {
            proxy.scrollTo(Self.bottomAnchorID, anchor: .bottom)
        }
    }
}

// MARK: - Empty state

@MainActor
private struct ChatEmptyState: View {
    let modelReady: Bool
    let modelName: String
    let onOpenModels: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 46, weight: .light))
                .foregroundStyle(.tertiary)
                .accessibilityHidden(true)

            Text("SolidChat")
                .font(.title2.weight(.semibold))

            if modelReady {
                Text("\(modelName) is ready. Ask anything below.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            } else {
                Text("Load a model to start")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Button("Browse Models", action: onOpenModels)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .help("Open the Models list and load a model")
            }
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Message row

@MainActor
private struct ChatMessageRow: View {
    let message: Msg
    let maxBubbleWidth: CGFloat
    let isStreaming: Bool
    let canRegenerate: Bool
    let isEditing: Bool
    /// Already scoped to this message by `ChatView` — `.idle` on every other row.
    let speechState: SpeechState
    let beginEditing: () -> Void
    let cancelEditing: () -> Void
    let onRegenerate: () -> Void
    let onSaveEdit: (String) -> Void
    let onSpeak: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var hovering = false
    @State private var copied = false
    @State private var editDraft = ""
    @State private var thinkingExpanded = false

    var body: some View {
        Group {
            if message.role == .user {
                userRow
            } else {
                assistantRow
            }
        }
        .onHover { inside in
            if reduceMotion {
                hovering = inside
            } else {
                withAnimation(Animation.easeInOut(duration: 0.12)) { hovering = inside }
            }
        }
        .onChange(of: isEditing, initial: true) { _, editing in
            if editing { editDraft = message.content }
        }
    }

    // MARK: User

    private var userRow: some View {
        HStack(alignment: .top, spacing: 0) {
            Spacer(minLength: 40)
            VStack(alignment: .trailing, spacing: 6) {
                if isEditing {
                    editor
                } else {
                    Text(message.content)
                        .textSelection(.enabled)
                        .multilineTextAlignment(.leading)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    actions
                }
            }
            .frame(maxWidth: maxBubbleWidth, alignment: .trailing)
        }
    }

    private var editor: some View {
        VStack(alignment: .trailing, spacing: 8) {
            TextEditor(text: $editDraft)
                .font(.body)
                .scrollContentBackground(.hidden)
                .frame(minHeight: 76)
                .padding(6)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .accessibilityLabel("Edit message")
            HStack(spacing: 8) {
                Button("Cancel", action: cancelEditing)
                    .keyboardShortcut(.cancelAction)
                // Disabled while streaming: editAndResend is a no-op mid-generation, so
                // leaving it enabled silently discards what the user just typed.
                Button("Save & Resend") { onSaveEdit(editDraft) }
                    .disabled(editDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                              || isStreaming)
                    .help(isStreaming ? "Stop generating first" : "Resend from this message")
            }
            .controlSize(.small)
        }
    }

    // MARK: Assistant

    private var assistantRow: some View {
        VStack(alignment: .leading, spacing: 10) {
            if !message.thinking.isEmpty {
                thinkingBlock
            }

            if !message.content.isEmpty {
                MarkdownMessageView(text: message.content)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else if isStreaming && message.thinking.isEmpty {
                ProgressView()
                    .controlSize(.small)
                    .accessibilityLabel("Generating a response")
            }

            if let failure = message.error {
                Label(failure, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .accessibilityLabel("Error: \(failure)")
            }

            if let stats = message.stats, !isStreaming {
                Text(statsLine(stats))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }

            actions
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var thinkingBlock: some View {
        DisclosureGroup(isExpanded: $thinkingExpanded) {
            Text(message.thinking)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
        } label: {
            Label(isThinkingLive ? "Thinking…" : "Thoughts", systemImage: "brain")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .onChange(of: isThinkingLive, initial: true) { _, live in
            let animation: Animation? = reduceMotion ? nil : Animation.easeInOut(duration: 0.15)
            withAnimation(animation) { thinkingExpanded = live }
        }
    }

    /// The model is still reasoning and has produced no answer text yet.
    private var isThinkingLive: Bool { isStreaming && message.content.isEmpty }

    private func statsLine(_ stats: MsgStats) -> String {
        String(format: "≈%.1f tok/s · %d tok · TTFT %.1fs", stats.tokPerSec, stats.tokens, stats.ttft)
    }

    // MARK: Hover actions

    private var actions: some View {
        HStack(spacing: 2) {
            iconButton(
                systemImage: copied ? "checkmark" : "doc.on.doc",
                help: copied ? "Copied" : "Copy this message",
                label: "Copy message",
                action: copyMessage
            )
            if canRegenerate {
                iconButton(
                    systemImage: "arrow.clockwise",
                    help: "Regenerate this response",
                    label: "Regenerate response",
                    action: onRegenerate
                )
            }
            if message.role == .user {
                iconButton(
                    systemImage: "pencil",
                    help: "Edit and resend this message",
                    label: "Edit message",
                    action: beginEditing
                )
            }
            if canSpeak {
                speakButton
            }
        }
        .controlSize(.small)
        .foregroundStyle(.secondary)
        // Revealed on hover for the pointer, but deliberately left in the
        // accessibility tree — VoiceOver users never hover. Pinned open while this
        // message is being spoken: the pointer wanders, and a stop button that
        // fades out while audio is still playing is unreachable.
        .opacity(hovering || speechState.isActive ? 1 : 0)
        .allowsHitTesting(hovering || speechState.isActive)
    }

    /// Nothing to read aloud on a user turn, an empty placeholder, or a reply that
    /// only produced an error.
    private var canSpeak: Bool {
        message.role == .assistant
            && !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var isSpeaking: Bool { speechState == .speaking }

    @ViewBuilder
    private var speakButton: some View {
        if speechState == .preparing {
            // Kokoro's worker takes ~4s to load and ~1.8s for the first utterance.
            // A dead-looking button with no audio for six seconds reads as broken.
            ProgressView()
                .controlSize(.small)
                .frame(width: 18, height: 18)
                .help("Preparing speech…")
                .accessibilityLabel("Preparing speech")
        } else {
            iconButton(
                systemImage: isSpeaking ? "stop.circle.fill" : "speaker.wave.2",
                // A failed attempt keeps the speak affordance but says why, rather
                // than silently doing nothing the next time it is pressed.
                help: isSpeaking ? "Stop speaking"
                                 : (speechState.errorText.map { "Speech failed: \($0)" }
                                    ?? "Speak this message"),
                label: isSpeaking ? "Stop speaking" : "Speak this message",
                action: onSpeak
            )
        }
    }

    private func iconButton(
        systemImage: String,
        help: String,
        label: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .frame(width: 18, height: 18)
                .contentShape(Rectangle())
        }
        .buttonStyle(.borderless)
        .help(help)
        .accessibilityLabel(label)
    }

    private func copyMessage() {
        let board = NSPasteboard.general
        board.clearContents()
        board.setString(message.content, forType: .string)
        copied = true
        Task {
            try? await Task.sleep(for: .seconds(1.2))
            copied = false
        }
    }
}

// MARK: - Composer

@MainActor
private struct ChatComposer: View {
    @Binding var text: String
    let isStreaming: Bool
    let modelReady: Bool
    let modelName: String
    @Binding var autoSpeak: Bool
    let onSend: () -> Void
    let onStop: () -> Void

    private var trimmed: String { text.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var canSend: Bool { modelReady && !isStreaming && !trimmed.isEmpty }

    private var placeholder: String {
        guard modelReady else { return "Load a model to start chatting" }
        return modelName.isEmpty ? "Send a message…" : "Message \(modelName)…"
    }

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
            ZStack(alignment: .topLeading) {
                if text.isEmpty {
                    Text(placeholder)
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 8)
                        .allowsHitTesting(false)
                        .accessibilityHidden(true)
                }
                ChatComposerTextView(
                    text: $text,
                    isEditable: modelReady,
                    onSubmit: { if canSend { onSend() } },
                    onEscape: { if isStreaming { onStop() } }
                )
                .padding(.horizontal, 5)
                .padding(.vertical, 4)
                .accessibilityLabel("Message")
                .accessibilityHint("Return sends, Shift-Return starts a new line")
            }
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 11, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 11, style: .continuous).strokeBorder(.separator))
            .opacity(modelReady ? 1 : 0.6)

            autoSpeakButton
            sendOrStop
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.bar)
    }

    /// Deliberately smaller and un-tinted when off, so it reads as a preference
    /// sitting next to the send button rather than a second action.
    private var autoSpeakButton: some View {
        Button {
            autoSpeak.toggle()
        } label: {
            Image(systemName: autoSpeak ? "speaker.wave.2.circle.fill" : "speaker.wave.2.circle")
                .font(.system(size: 21))
                .foregroundStyle(autoSpeak ? Color.accentColor : Color.secondary)
        }
        .buttonStyle(.plain)
        .help("Speak each reply out loud as it finishes")
        .accessibilityLabel("Speak replies automatically")
        .accessibilityValue(autoSpeak ? "On" : "Off")
        .accessibilityAddTraits(autoSpeak ? .isSelected : [])
    }

    @ViewBuilder
    private var sendOrStop: some View {
        if isStreaming {
            Button(action: onStop) {
                Image(systemName: "stop.circle.fill")
                    .font(.system(size: 26))
                    .foregroundStyle(Color.red)
            }
            .buttonStyle(.plain)
            .help("Stop generating (Escape)")
            .accessibilityLabel("Stop generating")

            // Escape stops generation even when focus is not in the text view.
            Button("Stop", action: onStop)
                .keyboardShortcut(.cancelAction)
                .frame(width: 0, height: 0)
                .opacity(0)
                .accessibilityHidden(true)
        } else {
            Button(action: onSend) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 26))
                    .foregroundStyle(canSend ? Color.accentColor : Color.secondary)
            }
            .buttonStyle(.plain)
            .disabled(!canSend)
            .help(modelReady ? "Send (Return) — Shift-Return for a new line" : "Load a model first")
            .accessibilityLabel("Send message")
        }
    }
}

// MARK: - Composer text view

/// An `NSTextView` that grows with its content up to `maxLines` and then scrolls.
/// Return submits, Shift-Return (and Option-Return) insert a newline, Escape cancels.
@MainActor
private struct ChatComposerTextView: NSViewRepresentable {
    @Binding var text: String
    var isEditable: Bool
    var minLines: Int = 1
    var maxLines: Int = 8
    var onSubmit: () -> Void
    var onEscape: () -> Void

    private static let insetH: CGFloat = 2
    private static let insetV: CGFloat = 4
    private static let fragmentPadding: CGFloat = 4

    private static var font: NSFont { .preferredFont(forTextStyle: .body) }

    private static var lineHeight: CGFloat {
        let f = font
        return ceil(f.ascender - f.descender + f.leading)
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        // TextKit 1 keeps `usedRect` measurement and `doCommandBy` behaviour predictable.
        let textView = NSTextView(usingTextLayoutManager: false)
        textView.delegate = context.coordinator
        textView.string = text
        textView.font = Self.font
        textView.textColor = .labelColor
        textView.insertionPointColor = .controlAccentColor
        textView.isRichText = false
        textView.importsGraphics = false
        textView.allowsUndo = true
        textView.isEditable = isEditable
        textView.isSelectable = true
        textView.drawsBackground = false
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainerInset = NSSize(width: Self.insetH, height: Self.insetV)
        textView.minSize = NSSize(width: 0, height: 0)
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude,
                                  height: CGFloat.greatestFiniteMagnitude)
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.lineFragmentPadding = Self.fragmentPadding
        textView.textContainer?.containerSize = NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude)
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false

        let scrollView = NSScrollView()
        scrollView.documentView = textView
        scrollView.drawsBackground = false
        scrollView.borderType = .noBorder
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = true
        scrollView.verticalScrollElasticity = .allowed
        scrollView.horizontalScrollElasticity = .none
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let textView = scrollView.documentView as? NSTextView else { return }

        if textView.string != text {
            textView.string = text
            let end = (text as NSString).length
            textView.setSelectedRange(NSRange(location: end, length: 0))
            if text.isEmpty { textView.scroll(.zero) }
        }
        if textView.isEditable != isEditable { textView.isEditable = isEditable }

        // Take focus once the view is in a window so typing works immediately.
        if !context.coordinator.didFocus, isEditable, let window = scrollView.window {
            context.coordinator.didFocus = true
            window.makeFirstResponder(textView)
        }
    }

    /// Height is derived from the text itself, so growth needs no extra state.
    func sizeThatFits(_ proposal: ProposedViewSize, nsView: NSScrollView, context: Context) -> CGSize? {
        let proposed = proposal.width ?? nsView.bounds.width
        guard proposed.isFinite, proposed > 0 else { return nil }

        let line = Self.lineHeight
        let chrome = Self.insetV * 2
        let minHeight = line * CGFloat(minLines) + chrome
        let maxHeight = line * CGFloat(maxLines) + chrome
        let textWidth = max(1, proposed - Self.insetH * 2 - Self.fragmentPadding * 2)

        var probe = text
        if probe.isEmpty { probe = " " }
        if probe.hasSuffix("\n") { probe += " " }

        let bounds = (probe as NSString).boundingRect(
            with: NSSize(width: textWidth, height: CGFloat.greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            attributes: [.font: Self.font]
        )
        let content = ceil(bounds.height) + chrome
        return CGSize(width: proposed, height: min(max(content, minHeight), maxHeight))
    }

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: ChatComposerTextView
        var didFocus = false

        init(_ parent: ChatComposerTextView) {
            self.parent = parent
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            if parent.text != textView.string { parent.text = textView.string }
        }

        func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            switch commandSelector {
            case #selector(NSResponder.insertNewline(_:)):
                let shift = NSApplication.shared.currentEvent?.modifierFlags.contains(.shift) ?? false
                if shift {
                    insertNewline(in: textView)
                } else {
                    parent.onSubmit()
                }
                return true

            case NSSelectorFromString("insertLineBreak:"),
                 NSSelectorFromString("insertNewlineIgnoringFieldEditor:"):
                insertNewline(in: textView)
                return true

            case #selector(NSResponder.cancelOperation(_:)):
                parent.onEscape()
                return true

            default:
                return false
            }
        }

        private func insertNewline(in textView: NSTextView) {
            textView.insertText("\n", replacementRange: textView.selectedRange())
            if parent.text != textView.string { parent.text = textView.string }
        }
    }
}
