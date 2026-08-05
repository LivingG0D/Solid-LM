import AppKit
import SwiftUI

// ---------------------------------------------------------------------------
// AppStore surface consumed by this file (AppStore lives in Core/Store.swift):
//
//   init()
//   func bootstrap()                 // idempotent, synchronous kick-off
//   func newConversation()
//   func rescanModels() async
//   func unload()
//   func stop()                      // cancel the in-flight response
//   func persistCurrent()
//   var  isStreaming: Bool
//   var  engineState: EngineState
//   let  engine: EngineManager       // .shutdownBlocking() — bounded, synchronous
//   let  images: ImageStore          // .unload(), .engine.shutdownBlocking()
//   let  speech                      // .stop(), .shutdownBlocking()
//
// SettingsView is assumed to exist with a no-argument initialiser and to read
// AppStore out of the environment.
// ---------------------------------------------------------------------------

private let helpWindowID = "solidchat.help"

@main
struct SolidChatApp: App {
    @State private var store = AppStore()
    @NSApplicationDelegateAdaptor(SolidChatAppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(store)
                .task {
                    // The delegate is what kills the engine child on quit; hand it the
                    // store before anything can be launched.
                    appDelegate.store = store
                    store.bootstrap()
                }
        }
        .defaultSize(width: 1180, height: 800)
        .windowToolbarStyle(.unified)
        .commands {
            SolidChatCommands(store: store)
        }

        Window("SolidChat Help", id: helpWindowID) {
            HelpView()
        }
        .defaultSize(width: 480, height: 480)
        .windowResizability(.contentMinSize)

        Settings {
            SettingsView()
                .environment(store)
        }
    }
}

// MARK: - Menu bar

@MainActor
private struct SolidChatCommands: Commands {
    let store: AppStore

    var body: some Commands {
        // Replaces "New" / "New Window" in the File menu: this app has one window.
        CommandGroup(replacing: .newItem) {
            Button("New Chat") {
                store.newConversation()
            }
            .keyboardShortcut("n", modifiers: .command)
        }

        CommandMenu("Model") {
            Button("Rescan Models") {
                Task { await store.rescanModels() }
            }
            .keyboardShortcut("r", modifiers: .command)

            Divider()

            Button("Unload Model") {
                store.unload()
            }
            .keyboardShortcut("u", modifiers: [.command, .shift])
            .disabled(store.engineState == .idle)

            // Ungated: unloading with no image model loaded is a no-op.
            Button("Unload Image Model") {
                store.images.unload()
            }

            Divider()

            Button("Stop Generating") {
                store.stop()
            }
            .keyboardShortcut(".", modifiers: .command)
            .disabled(!store.isStreaming)

            // Ungated, like "Unload Image Model": stopping silence is a no-op, and
            // gating it on speech state would make the item flicker mid-sentence.
            Button("Stop Speaking") {
                store.speech.stop()
            }
            .keyboardShortcut(".", modifiers: [.command, .shift])
        }

        CommandGroup(replacing: .help) {
            HelpMenuButton()
        }
    }
}

/// `openWindow` is read from the environment, which resolves reliably inside a
/// view — so the Help item is a view rather than a bare `Button` in `Commands`.
private struct HelpMenuButton: View {
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button("SolidChat Help") {
            openWindow(id: helpWindowID)
        }
        .keyboardShortcut("?", modifiers: .command)
    }
}

// MARK: - Help window

private struct HelpView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("SolidChat")
                    .font(.title2.weight(.semibold))

                Text("A local chat client. One model is loaded at a time and served over an OpenAI-compatible API on port 8181.")
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                section("Sections") {
                    row("Chat", "Talk to the loaded model. Per-message tokens/sec, time to first token and token count.")
                    row("Images", "Generate, edit and upscale with a diffusion model. Separate engine, separate memory.")
                    row("Models", "Everything found under the models folder. Load, unload, per-model settings.")
                    row("Downloads", "Fetch a model from a HuggingFace repo straight into the models folder.")
                    row("Logs", "Live engine output. A failed load explains itself here.")
                }

                section("Keyboard") {
                    shortcut("⌘N", "New chat")
                    shortcut("⌘R", "Rescan the models folder")
                    shortcut("⇧⌘U", "Unload the current model")
                    shortcut("⌘.", "Stop generating")
                    shortcut("⇧⌘.", "Stop speaking")
                    shortcut("⌘,", "Settings")
                }

                section("Notes") {
                    row("Memory", "Unified memory is shared between the app and the model. Loading a second model evicts the first.")
                    row("GPU layers", "Full offload by default — a partial offload measured roughly ten times slower.")
                }
            }
            .padding(24)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(minWidth: 420, minHeight: 380)
    }

    @ViewBuilder
    private func section(_ title: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            content()
        }
    }

    private func row(_ term: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(term)
                .font(.callout.weight(.medium))
            Text(detail)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func shortcut(_ keys: String, _ detail: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(keys)
                .font(.system(.callout, design: .monospaced))
                .frame(width: 52, alignment: .leading)
            Text(detail)
                .font(.callout)
                .foregroundStyle(.secondary)
            Spacer(minLength: 0)
        }
    }
}

// MARK: - Lifecycle

/// Owns nothing except the guarantee that the engine child process dies with the app.
/// A leaked llama-server keeps its whole model resident in unified memory.
@MainActor
final class SolidChatAppDelegate: NSObject, NSApplicationDelegate {
    var store: AppStore?

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        // flushToDisk, not persistCurrent: it cancels the pending debounce and writes
        // settings AND the open chat synchronously. persistCurrent leaves a settings
        // change made in the last 500 ms unsaved, and its save is detached — on quit
        // there is no "moment to land".
        store?.flushToDisk()
        // SIGTERM, then SIGKILL, bounded to a few seconds so quitting never hangs.
        store?.engine.shutdownBlocking()
        // sd-server is a second child holding its own model in unified memory.
        store?.images.engine.shutdownBlocking()
        // The Kokoro worker is a third child; a leaked python holds the TTS model.
        store?.speech.shutdownBlocking()
    }
}
