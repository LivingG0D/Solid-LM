import AppKit
import SwiftUI

// ---------------------------------------------------------------------------
// Surface consumed from the rest of the module:
//
//   AppStore        let engine: EngineManager
//   EngineManager   var logLines: [String]      // newest last, capped at 500
//                   func clearLog()
// ---------------------------------------------------------------------------

/// The engine child's stdout and stderr. When a load fails, the explanation is
/// here — so this view is optimised for reading back, not just watching.
@MainActor
struct LogsView: View {
    @Environment(AppStore.self) private var store
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Off by default would be wrong (a live load is the common case), but the
    /// user must be able to stop the view yanking itself downwards mid-read.
    @State private var follow = true

    private var lines: [String] { store.engine.logLines }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
        }
    }

    // MARK: Header

    private var header: some View {
        HStack(spacing: 12) {
            Toggle(isOn: $follow) {
                Text("Follow")
            }
            .toggleStyle(.switch)
            .controlSize(.small)
            .help("Keep scrolling to the newest line. Switch off to read back without being pulled down.")
            .accessibilityLabel("Follow newest output")

            Text(lines.count == 1 ? "1 line" : "\(lines.count) lines")
                .font(.caption)
                .monospacedDigit()
                .foregroundStyle(.secondary)

            Spacer(minLength: 8)

            Button {
                copyAll()
            } label: {
                Label("Copy All", systemImage: "doc.on.doc")
            }
            .disabled(lines.isEmpty)
            .help("Copy the whole log to the clipboard")
            .accessibilityLabel("Copy all engine output")

            Button(role: .destructive) {
                store.engine.clearLog()
            } label: {
                Label("Clear", systemImage: "trash")
            }
            .disabled(lines.isEmpty)
            .help("Clear the lines shown here. The engine keeps running.")
            .accessibilityLabel("Clear the log view")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(.bar)
    }

    // MARK: Content

    @ViewBuilder
    private var content: some View {
        if lines.isEmpty {
            ContentUnavailableView {
                Label("No Engine Output", systemImage: "text.alignleft")
            } description: {
                Text("No engine output yet — load a model.")
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollViewReader { proxy in
                ScrollView([.vertical, .horizontal]) {
                    LazyVStack(alignment: .leading, spacing: 1) {
                        ForEach(Array(lines.enumerated()), id: \.offset) { index, line in
                            Text(line)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(tint(for: line))
                                .textSelection(.enabled)
                                .lineLimit(1)
                                .id(index)
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .onAppear { scrollToEnd(proxy, animated: false) }
                .onChange(of: lines.count) { _, _ in
                    scrollToEnd(proxy, animated: true)
                }
                .onChange(of: follow) { _, isOn in
                    if isOn { scrollToEnd(proxy, animated: true) }
                }
            }
        }
    }

    /// llama.cpp prefixes its own failures; colouring them saves a lot of squinting.
    private func tint(for line: String) -> Color {
        let lowered = line.lowercased()
        if lowered.contains("error") || lowered.contains("failed") || lowered.contains("traceback") {
            return .red
        }
        if lowered.contains("warn") {
            return .orange
        }
        return .primary
    }

    private func scrollToEnd(_ proxy: ScrollViewProxy, animated: Bool) {
        guard follow, !lines.isEmpty else { return }
        let last = lines.count - 1
        if animated && !reduceMotion {
            withAnimation(.easeOut(duration: 0.15)) {
                proxy.scrollTo(last, anchor: .bottom)
            }
        } else {
            proxy.scrollTo(last, anchor: .bottom)
        }
    }

    private func copyAll() {
        let text = lines.joined(separator: "\n")
        guard !text.isEmpty else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }
}
