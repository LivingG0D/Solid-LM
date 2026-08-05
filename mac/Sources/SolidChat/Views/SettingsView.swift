import AppKit
import SwiftUI

// ---------------------------------------------------------------------------
// Surface consumed from the rest of the module:
//
//   AppStore    var samplers: Samplers        // settable; persists (debounced)
//               var hfToken: String           // settable; persists (debounced)
//               func rescanModels() async
//               let speech                    // @Observable @MainActor, see below
//   speech      var settings: SpeechSettings  // settable
//               func voices(for: SpeechEngineKind) -> [SpeechVoice]
//               func previewVoice(_: SpeechVoice)     // fire-and-forget
//               func warmUpKokoro()                   // fire-and-forget
//   Paths       Key.modelsRoot / Key.llamaServer / Key.venvPython   (UserDefaults keys)
//               modelsRoot / llamaServer / venvPython               (effective URLs)
//               defaultLlamaServer / defaultVenvPython
// ---------------------------------------------------------------------------

private let hfTokensPageURL = URL(string: "https://huggingface.co/settings/tokens")!

/// The Settings scene (⌘,). Four tabs, because the groups have nothing to do with
/// each other: what the model does, how it is read aloud, where the files are, and
/// who we are to HuggingFace.
@MainActor
struct SettingsView: View {
    var body: some View {
        TabView {
            GenerationSettings()
                .tabItem { Label("Generation", systemImage: "slider.horizontal.3") }

            SpeechSettingsTab()
                .tabItem { Label("Speech", systemImage: "waveform") }

            PathsSettings()
                .tabItem { Label("Paths", systemImage: "folder") }

            HuggingFaceSettings()
                .tabItem { Label("HuggingFace", systemImage: "arrow.down.circle") }
        }
        .frame(width: 580, height: 580)
    }
}

// MARK: - Generation

@MainActor
private struct GenerationSettings: View {
    @Environment(AppStore.self) private var store

    var body: some View {
        @Bindable var store = store

        Form {
            Section("System prompt") {
                VStack(alignment: .leading, spacing: 6) {
                    TextEditor(text: $store.samplers.system)
                        .font(.body)
                        .scrollContentBackground(.hidden)
                        .frame(minHeight: 96)
                        .padding(6)
                        .background(.background, in: RoundedRectangle(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(.quaternary))
                        .accessibilityLabel("System prompt")

                    Text("Sent ahead of every conversation, before your first message. Leave it empty to use whatever the model was trained to do by default.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.vertical, 2)
            }

            Section("Sampling") {
                DoubleSettingRow(
                    title: "Temperature",
                    explanation: "How much risk the model takes when picking the next word. 0 is repeatable and flat; above ~1.2 it starts to wander.",
                    range: 0...2,
                    step: 0.05,
                    decimals: 2,
                    value: $store.samplers.temperature)

                IntSettingRow(
                    title: "Top-k",
                    explanation: "Only the k most likely next words are considered at all. 0 turns the cut-off off entirely.",
                    range: 0...200,
                    step: 1,
                    value: $store.samplers.topK)

                DoubleSettingRow(
                    title: "Top-p",
                    explanation: "Keeps the smallest group of words whose probabilities add up to p, and ignores the rest. 1 disables it.",
                    range: 0...1,
                    step: 0.01,
                    decimals: 2,
                    value: $store.samplers.topP)

                DoubleSettingRow(
                    title: "Min-p",
                    explanation: "Drops any word less likely than this fraction of the best one. A gentler filter than top-p, and usually enough on its own.",
                    range: 0...0.5,
                    step: 0.01,
                    decimals: 2,
                    value: $store.samplers.minP)

                DoubleSettingRow(
                    title: "Repeat penalty",
                    explanation: "Pushes down words that already appeared. 1 is off; much above 1.2 and the model starts dodging ordinary words like \"the\".",
                    range: 1...1.6,
                    step: 0.01,
                    decimals: 2,
                    value: $store.samplers.repeatPenalty)

                IntSettingRow(
                    title: "Max tokens",
                    explanation: "Hard ceiling on a single reply, in tokens (roughly ¾ of a word each). The model usually stops well before it.",
                    range: 64...32768,
                    step: 64,
                    value: $store.samplers.maxTokens)
            }

            Section {
                HStack {
                    Button("Reset to Defaults") { resetSamplers() }
                        .help("Restore the shipped sampling values")
                    Spacer(minLength: 0)
                }

                Text("Restores temperature, top-k, top-p, min-p, repeat penalty and max tokens. Your system prompt is left alone.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .formStyle(.grouped)
    }

    /// Everything except the system prompt — losing a carefully written prompt to a
    /// button labelled "reset the sliders" would be a nasty surprise.
    private func resetSamplers() {
        let defaults = Samplers()
        var updated = store.samplers
        updated.temperature = defaults.temperature
        updated.topK = defaults.topK
        updated.topP = defaults.topP
        updated.minP = defaults.minP
        updated.repeatPenalty = defaults.repeatPenalty
        updated.maxTokens = defaults.maxTokens
        store.samplers = updated
    }
}

// MARK: - Generation rows

private struct DoubleSettingRow: View {
    let title: String
    let explanation: String
    let range: ClosedRange<Double>
    let step: Double
    let decimals: Int
    @Binding var value: Double

    /// Both the field and the slider write through this, so a typed-in 9999 lands
    /// inside the range instead of pinning the slider off its own track.
    private var clamped: Binding<Double> {
        Binding(get: { min(max(value, range.lowerBound), range.upperBound) },
                set: { value = min(max($0, range.lowerBound), range.upperBound) })
    }

    private var readout: String {
        String(format: "%.\(decimals)f", clamped.wrappedValue)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 10) {
                Text(title)
                Spacer(minLength: 8)
                TextField(title, value: clamped, format: .number.precision(.fractionLength(decimals)))
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
                    .monospacedDigit()
                    .frame(width: 78)
                    .labelsHidden()
                    .accessibilityLabel(title)
            }

            Slider(value: clamped, in: range, step: step)
                .accessibilityLabel(title)
                .accessibilityValue(readout)

            Text(explanation)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 3)
    }
}

private struct IntSettingRow: View {
    let title: String
    let explanation: String
    let range: ClosedRange<Int>
    let step: Int
    @Binding var value: Int

    private var clamped: Binding<Int> {
        Binding(get: { min(max(value, range.lowerBound), range.upperBound) },
                set: { value = min(max($0, range.lowerBound), range.upperBound) })
    }

    private var sliderValue: Binding<Double> {
        Binding(get: { Double(clamped.wrappedValue) },
                set: { clamped.wrappedValue = Int($0.rounded()) })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 10) {
                Text(title)
                Spacer(minLength: 8)
                TextField(title, value: clamped, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
                    .monospacedDigit()
                    .frame(width: 78)
                    .labelsHidden()
                    .accessibilityLabel(title)
            }

            Slider(value: sliderValue,
                   in: Double(range.lowerBound)...Double(range.upperBound),
                   step: Double(step))
                .accessibilityLabel(title)
                .accessibilityValue("\(clamped.wrappedValue)")

            Text(explanation)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 3)
    }
}

// MARK: - Speech

/// Opens System Settings ▸ Accessibility, which is where Read & Speak lives — the
/// section macOS called "Spoken Content" before this release.
///
/// What is actually verified on this machine, rather than assumed: the pane still
/// answers to the legacy id. AccessibilitySettingsExtension.appex declares
/// `legacyBundleIdentifier = com.apple.preference.universalaccess` and
/// `allowsXAppleSystemPreferencesURLScheme = true`, and `open` on this URL succeeds.
///
/// The `?SpokenContent` anchor is a different matter and is best treated as a hint:
/// the only anchors that appear in the macOS 26 pane binary are of the
/// `Seeing_VoiceOver` / `Media_Descriptions` shape, and there is no speech one among
/// them, so the deep link most likely lands on Accessibility's root rather than
/// scrolling to Read & Speak. It is kept because it costs nothing and still works on
/// older systems. That is exactly why the button is labelled for what it reliably
/// does, and why the prose beside it spells out the remaining path in full.
private let spokenContentSettingsURL =
    URL(string: "x-apple.systempreferences:com.apple.preference.universalaccess?SpokenContent")!

/// Named `…Tab` because `SpeechSettings` is the Codable value type in SpeechTypes.swift.
@MainActor
private struct SpeechSettingsTab: View {
    @Environment(AppStore.self) private var store

    var body: some View {
        Form {
            engineSection

            // Only the controls the selected engine actually obeys: Kokoro ignores
            // AVSpeechUtterance's rate and pitch entirely, and showing dead sliders
            // is how a user concludes the app is broken.
            if store.speech.settings.engine == .system {
                systemVoiceSection
                systemToneSection
                systemQualitySection
            } else {
                kokoroVoiceSection
                kokoroSpeedSection
                kokoroWarmUpSection
            }

            autoSpeakSection
        }
        .formStyle(.grouped)
    }

    // MARK: Engine

    private var engineSection: some View {
        @Bindable var speech = store.speech

        // Radio group rather than a popup: there are two choices, and the whole
        // decision rests on the trade-off spelled out in `detail`.
        return Section("Engine") {
            Picker("Engine", selection: $speech.settings.engine) {
                ForEach(SpeechEngineKind.allCases, id: \.self) { kind in
                    VStack(alignment: .leading, spacing: 1) {
                        Text(kind.label)
                        Text(kind.detail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .tag(kind)
                }
            }
            .pickerStyle(.radioGroup)
            .labelsHidden()
            .accessibilityLabel("Speech engine")
        }
    }

    // MARK: Voice

    private var systemVoiceSection: some View {
        @Bindable var speech = store.speech
        let voices = options(for: .system)

        return Section("Voice") {
            HStack(spacing: 10) {
                Picker("Voice", selection: $speech.settings.systemVoiceID) {
                    Text("System Default").tag("")
                    voiceEntries(voices)
                }
                .labelsHidden()
                .accessibilityLabel("System voice")

                previewButton(for: .system)
            }

            Text(voices.isEmpty
                 ? "No system voices were found, which should not happen — macOS installs several with the OS."
                 : "All \(voices.count) voices macOS has installed, English first. Each is labelled with its quality: Compact is the small robotic kind, Enhanced and Premium are the good ones.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var kokoroVoiceSection: some View {
        @Bindable var speech = store.speech
        let voices = options(for: .kokoro)
        // The 54 names only exist once the worker has announced itself, so before the
        // first warm-up the list holds nothing but the saved id.
        let known = store.speech.voices(for: .kokoro).isEmpty == false

        return Section("Voice") {
            HStack(spacing: 10) {
                Picker("Voice", selection: $speech.settings.kokoroVoiceID) {
                    voiceEntries(voices)
                }
                .labelsHidden()
                .accessibilityLabel("Kokoro voice")

                previewButton(for: .kokoro)
            }

            Text(known
                 ? "\(voices.count) neural voices. The first letters of the raw name are the accent and the gender — af_heart is an American female voice."
                 : "The full list of voices arrives when the Kokoro worker starts. Warm it up below, or just press Preview and wait a few seconds.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// English first, then everything else, each group behind its own header.
    /// A flat 180-item menu is unusable, and 139 of those items are not English.
    @ViewBuilder
    private func voiceEntries(_ voices: [SpeechVoice]) -> some View {
        let english = voices.filter { $0.language.hasPrefix("en") }
        let other = voices.filter { !$0.language.hasPrefix("en") }

        if !english.isEmpty, !other.isEmpty {
            Section("English") {
                ForEach(english) { Text(Self.entryLabel($0)).tag($0.id) }
            }
            Section("Other languages") {
                ForEach(other) { Text(Self.entryLabel($0)).tag($0.id) }
            }
        } else {
            ForEach(voices) { Text(Self.entryLabel($0)).tag($0.id) }
        }
    }

    private func previewButton(for kind: SpeechEngineKind) -> some View {
        let voice = selectedVoice(for: kind)

        return Button("Preview") {
            if let voice { store.speech.previewVoice(voice) }
        }
        .disabled(voice == nil)
        .help(voice == nil
              ? "Pick a named voice to hear it"
              : "Speak a short sample in this voice")
        .accessibilityLabel("Preview voice")
    }

    /// Voices for `kind`, English first, plus a stand-in for a saved id the engine has
    /// not listed. Without the stand-in a picker whose selection is absent shows blank,
    /// and the user's saved choice silently looks unset.
    private func options(for kind: SpeechEngineKind) -> [SpeechVoice] {
        var voices = store.speech.voices(for: kind).sorted { a, b in
            let aEnglish = a.language.hasPrefix("en")
            let bEnglish = b.language.hasPrefix("en")
            if aEnglish != bEnglish { return aEnglish }
            return a.displayName.localizedStandardCompare(b.displayName) == .orderedAscending
        }

        let saved = store.speech.settings.voiceID(for: kind)
        if !saved.isEmpty, !voices.contains(where: { $0.id == saved }) {
            voices.insert(SpeechVoice(id: saved, name: saved, language: "", engine: kind), at: 0)
        }
        return voices
    }

    private func selectedVoice(for kind: SpeechEngineKind) -> SpeechVoice? {
        let id = store.speech.settings.voiceID(for: kind)
        guard !id.isEmpty else { return nil }
        return options(for: kind).first { $0.id == id }
    }

    private static func entryLabel(_ voice: SpeechVoice) -> String {
        switch voice.engine {
        case .kokoro:
            return voice.displayName
        case .system:
            // Names repeat across locales — "Karen" is both en-AU and en-US — and the
            // quality is the one fact that decides whether the voice is worth using.
            let language = voice.language.isEmpty ? "" : " · \(voice.language)"
            return "\(voice.displayName)\(language) · \(voice.quality.label)"
        }
    }

    // MARK: Tone

    private var systemToneSection: some View {
        @Bindable var speech = store.speech

        return Section("How it sounds") {
            DoubleSettingRow(
                title: "Rate",
                explanation: "Speaking speed on AVSpeechUtterance's own scale, where 0.5 is the macOS default — not a multiplier. Much below 0.35 it drawls; much above 0.75 it runs its words together.",
                range: 0...1,
                step: 0.01,
                decimals: 2,
                value: $speech.settings.rate)

            DoubleSettingRow(
                title: "Pitch",
                explanation: "Multiplies the voice's natural pitch. 1 leaves it alone; the extremes sound like a cartoon either way.",
                range: 0.5...2,
                step: 0.05,
                decimals: 2,
                value: $speech.settings.pitch)

            DoubleSettingRow(
                title: "Volume",
                explanation: "Loudness of the spoken audio only. It does not touch the system volume.",
                range: 0...1,
                step: 0.05,
                decimals: 2,
                value: $speech.settings.volume)
        }
    }

    private var kokoroSpeedSection: some View {
        @Bindable var speech = store.speech

        return Section("How it sounds") {
            DoubleSettingRow(
                title: "Speed",
                explanation: "Kokoro's own multiplier, applied while it synthesises rather than to the finished audio, so the voice keeps its pitch at either end. 1 is its natural pace.",
                range: 0.5...2,
                step: 0.05,
                decimals: 2,
                value: $speech.settings.kokoroSpeed)

            Text("Rate, pitch and volume belong to the system engine and are not shown here — Kokoro has no equivalent knobs.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: Honest notes

    /// Counted from the voices actually installed, not hard-coded: this Mac has 41
    /// English voices and every one is Compact, but a user who downloads a Premium
    /// voice tomorrow should stop being told otherwise.
    private var systemQualitySection: some View {
        let english = store.speech.voices(for: .system).filter { $0.language.hasPrefix("en") }
        let better = english.filter { $0.quality != .compact }

        return Section("Voice quality") {
            if !english.isEmpty, better.isEmpty {
                Label {
                    Text("All \(english.count) English voices installed on this Mac are Compact — the small, robotic ones macOS ships with. No setting on this screen can improve them. Enhanced and Premium voices exist, but they are a separate download.")
                        .font(.caption)
                        .fixedSize(horizontal: false, vertical: true)
                } icon: {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                }
                .accessibilityElement(children: .combine)
            } else if !better.isEmpty {
                Text("\(better.count) of \(english.count) English voices here are Enhanced or Premium. Pick one of those — the Compact ones sound robotic.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Text("To download better ones: System Settings ▸ Accessibility ▸ Read & Speak (called Spoken Content before macOS 26) ▸ System voice ▸ Manage Voices. Anything marked Premium is worth the few hundred megabytes.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack {
                Button("Open Accessibility Settings") {
                    NSWorkspace.shared.open(spokenContentSettingsURL)
                }
                .help("Opens System Settings at Accessibility, where Read & Speak lives")
                Spacer(minLength: 0)
            }

            Text("Or switch to Kokoro at the top of this tab: it needs nothing from Apple and already sounds better than any Compact voice.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var kokoroWarmUpSection: some View {
        Section("Warm-up") {
            Label {
                Text("The first thing Kokoro says waits for the model to load — about four seconds on this Mac. The worker then stays resident and each sentence takes under two. Warming up now moves that wait to a moment you are not listening.")
                    .font(.caption)
                    .fixedSize(horizontal: false, vertical: true)
            } icon: {
                Image(systemName: "clock")
                    .foregroundStyle(.secondary)
            }
            .accessibilityElement(children: .combine)

            HStack {
                Button("Warm Up Now") { store.speech.warmUpKokoro() }
                    .help("Start the Kokoro worker now so the first reply does not wait for it")
                Spacer(minLength: 0)
            }
        }
    }

    // MARK: Auto-speak

    private var autoSpeakSection: some View {
        @Bindable var speech = store.speech

        return Section("Automatic speech") {
            Toggle("Speak replies as they arrive", isOn: $speech.settings.autoSpeak)
                .accessibilityLabel("Speak replies automatically")

            Text("Reads every assistant reply aloud without being asked. The speak button in Chat works either way, and Model ▸ Stop Speaking silences whatever is being said.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - Paths

@MainActor
private struct PathsSettings: View {
    @Environment(AppStore.self) private var store

    // These three keys are what `Paths` reads; writing them here is the whole
    // mechanism — there is no second copy of the value anywhere.
    @AppStorage(Paths.Key.modelsRoot) private var modelsRootOverride = ""
    @AppStorage(Paths.Key.llamaServer) private var llamaServerOverride = ""
    @AppStorage(Paths.Key.venvPython) private var venvPythonOverride = ""

    var body: some View {
        Form {
            Section("Models folder") {
                PathSettingRow(
                    explanation: "Scanned as publisher/model. LM Studio's own folder works as-is — models are shared, never copied.",
                    effective: Paths.modelsRoot,
                    status: PathCheck.directory(Paths.modelsRoot),
                    picksDirectory: true,
                    override: $modelsRootOverride,
                    afterChange: { Task { await store.rescanModels() } })
            }

            Section("llama-server binary") {
                PathSettingRow(
                    explanation: "The llama.cpp server executable. Its dylibs must sit in the same folder — the app points DYLD_LIBRARY_PATH there when it launches.",
                    effective: Paths.llamaServer,
                    status: PathCheck.executable(Paths.llamaServer),
                    picksDirectory: false,
                    override: $llamaServerOverride,
                    afterChange: {})
            }

            Section("Python (virtualenv)") {
                PathSettingRow(
                    explanation: "The interpreter used for MLX and vLLM — normally .venv/bin/python. It needs mlx-lm installed to serve MLX models.",
                    effective: Paths.venvPython,
                    status: PathCheck.executable(Paths.venvPython),
                    picksDirectory: false,
                    override: $venvPythonOverride,
                    afterChange: {})
            }

            Section {
                Text("Changes take effect on the next load. A running engine keeps using the paths it was started with.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .formStyle(.grouped)
    }
}

private struct PathStatus {
    var ok: Bool
    var note: String
}

private enum PathCheck {
    static func directory(_ url: URL) -> PathStatus {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        let exists = fm.fileExists(atPath: url.path(percentEncoded: false), isDirectory: &isDirectory)
        if exists && isDirectory.boolValue { return PathStatus(ok: true, note: "Folder found.") }
        if exists { return PathStatus(ok: false, note: "That path is a file, not a folder.") }
        return PathStatus(ok: false, note: "No folder at this path.")
    }

    static func executable(_ url: URL) -> PathStatus {
        let fm = FileManager.default
        let path = url.path(percentEncoded: false)
        if fm.isExecutableFile(atPath: path) { return PathStatus(ok: true, note: "Executable found.") }
        if fm.fileExists(atPath: path) { return PathStatus(ok: false, note: "Found, but not executable — chmod +x it.") }
        return PathStatus(ok: false, note: "Nothing at this path.")
    }
}

@MainActor
private struct PathSettingRow: View {
    let explanation: String
    let effective: URL
    let status: PathStatus
    let picksDirectory: Bool
    @Binding var override: String
    let afterChange: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 7) {
                Image(systemName: status.ok ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    .foregroundStyle(status.ok ? Color.green : Color.red)
                    .accessibilityHidden(true)

                Text(effective.path(percentEncoded: false))
                    .font(.system(.caption, design: .monospaced))
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                    .help(effective.path(percentEncoded: false))

                Spacer(minLength: 8)

                if !override.isEmpty {
                    Button("Use Default") {
                        override = ""
                        afterChange()
                    }
                    .help("Forget this override and go back to the built-in location")
                }

                Button("Browse…") { browse() }
                    .help(picksDirectory ? "Choose a folder" : "Choose a file")
            }
            .accessibilityElement(children: .contain)
            .accessibilityLabel(status.ok ? "Path found" : "Path not found")
            .accessibilityValue(effective.path(percentEncoded: false))

            Text(status.note)
                .font(.caption)
                .foregroundStyle(status.ok ? AnyShapeStyle(.secondary) : AnyShapeStyle(Color.red))

            Text(explanation)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 3)
    }

    private func browse() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = picksDirectory
        panel.canChooseFiles = !picksDirectory
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = picksDirectory
        panel.treatsFilePackagesAsDirectories = true
        // .venv and friends are hidden; without this the panel cannot reach them.
        panel.showsHiddenFiles = true
        panel.prompt = "Choose"
        panel.directoryURL = picksDirectory ? effective : effective.deletingLastPathComponent()

        guard panel.runModal() == .OK, let picked = panel.url else { return }
        override = picked.path(percentEncoded: false)
        afterChange()
    }
}

// MARK: - HuggingFace

@MainActor
private struct HuggingFaceSettings: View {
    @Environment(AppStore.self) private var store

    var body: some View {
        @Bindable var store = store

        Form {
            Section("Access token") {
                VStack(alignment: .leading, spacing: 6) {
                    SecureField("hf_…", text: $store.hfToken)
                        .textFieldStyle(.roundedBorder)
                        .accessibilityLabel("HuggingFace access token")

                    Text("Only needed for gated or private repos. Public models download without one.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.vertical, 2)

                HStack {
                    Button("Clear Token") { store.hfToken = "" }
                        .disabled(store.hfToken.isEmpty)
                        .help("Remove the stored token")
                    Spacer(minLength: 0)
                    Link("huggingface.co/settings/tokens", destination: hfTokensPageURL)
                        .help("Open the HuggingFace token page in your browser")
                }
            }

            Section("Where it is kept") {
                Label {
                    Text("This token is stored in plain text, in settings.json inside ~/Library/Application Support/SolidChat. It is not in the Keychain, and anyone who can read your home folder can read it.")
                        .font(.caption)
                        .fixedSize(horizontal: false, vertical: true)
                } icon: {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                }
                .accessibilityElement(children: .combine)

                Text("Create a fine-grained token with read access only. A read token can pull gated weights and nothing else, so a leak costs you nothing beyond revoking it. Never paste a write token here.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                Text("The token is sent only to huggingface.co, as an Authorization header.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .formStyle(.grouped)
    }
}
