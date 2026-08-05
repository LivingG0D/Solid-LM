import SwiftUI

// ---------------------------------------------------------------------------
// Surface consumed from the rest of the module:
//
//   AppStore      let downloader: Downloader
//                 var hfToken: String
//                 var modelsRoot: URL
//                 let stats: SystemStats            // .totalBytes
//                 func rescanModels() async
//   Downloader    var jobs: [DownloadJob]
//                 func inspect(_ urlOrRepo: String, token: String) async throws -> HFRepo
//                 func start(repo: HFRepo, quant: String, token: String, modelsRoot: URL)
//                 func cancel(_ id: UUID)
//                 static func filesToDownload(repo: HFRepo, quant: String) throws -> [HFFile]
//                 static func message(_ error: Error) -> String
//   ModelScanner  static func fitClass(size: Int64) -> FitClass
// ---------------------------------------------------------------------------

/// Quants worth preselecting, best first. Q4_K_M is the usual sweet spot: close to
/// full-precision quality at roughly a quarter of the weights.
private let preferredQuants = ["Q4_K_M", "Q4_K_S", "IQ4_XS", "Q5_K_M", "Q8_0"]

private let tokensPageURL = URL(string: "https://huggingface.co/settings/tokens")!

// MARK: - Downloads

@MainActor
struct DownloadsView: View {
    @Environment(AppStore.self) private var store

    @State private var query = ""
    @State private var repo: HFRepo?
    @State private var selectedQuant = ""
    @State private var isInspecting = false
    @State private var inspectError = ""
    /// Jobs whose completion has already triggered a rescan.
    @State private var rescanned: Set<UUID> = []

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                inspectSection
                if let repo { repoSection(repo) }
                transfersSection
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onChange(of: store.downloader.jobs) { _, jobs in
            noteFinishedJobs(jobs)
        }
    }

    // MARK: Inspect

    private var inspectSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Download a model")
                .font(.headline)

            HStack(spacing: 10) {
                TextField("org/name — or a huggingface.co model URL", text: $query)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1)
                    .onSubmit { inspect() }
                    .accessibilityLabel("HuggingFace repository")

                if isInspecting {
                    ProgressView()
                        .progressViewStyle(.circular)
                        .controlSize(.small)
                        .accessibilityLabel("Inspecting repository")
                }

                // Return in the field submits (see .onSubmit above); deliberately not
                // the window's default action, so Return cannot re-inspect out from
                // under the Download button once a repo is on screen.
                Button("Inspect") { inspect() }
                    .disabled(isInspecting || query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .help("Ask HuggingFace what is inside this repo")
            }

            Text("Files land in \(store.modelsRoot.path(percentEncoded: false)), laid out as publisher/model.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)

            if !inspectError.isEmpty {
                NoticeRow(text: inspectError,
                          systemImage: "exclamationmark.triangle.fill",
                          tint: .red)
            }
        }
    }

    private func inspect() {
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isInspecting else { return }

        isInspecting = true
        inspectError = ""

        Task {
            do {
                let found = try await store.downloader.inspect(text, token: store.hfToken)
                repo = found
                selectedQuant = Self.preferredQuant(in: found)
            } catch {
                repo = nil
                selectedQuant = ""
                inspectError = Downloader.message(error)
            }
            isInspecting = false
        }
    }

    /// First of `preferredQuants` the repo actually ships, else its first quant.
    private static func preferredQuant(in repo: HFRepo) -> String {
        guard repo.kind == .gguf, !repo.quants.isEmpty else { return "" }
        let available = Set(repo.quants)
        return preferredQuants.first(where: { available.contains($0) }) ?? repo.quants[0]
    }

    // MARK: Repo card

    @ViewBuilder
    private func repoSection(_ repo: HFRepo) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(repo.repo)
                        .font(.system(.title3, design: .monospaced))
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)

                    Chip(text: kindLabel(repo.kind))

                    if repo.gated {
                        Chip(text: "gated", tint: .orange)
                    }

                    Spacer(minLength: 0)
                }

                if repo.gated {
                    VStack(alignment: .leading, spacing: 4) {
                        NoticeRow(text: gatedNote,
                                  systemImage: "lock.fill",
                                  tint: store.hfToken.isEmpty ? .red : .orange)
                        if store.hfToken.isEmpty { TokenHint() }
                    }
                }

                if repo.kind == .gguf, !repo.quants.isEmpty {
                    quantPicker(repo)
                } else if repo.kind == .mlx {
                    Text("MLX repos are downloaded whole — weights, tokenizer and config.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                sizeSummary

                HStack(spacing: 12) {
                    Button {
                        startDownload(repo)
                    } label: {
                        Label("Download & Install", systemImage: "arrow.down.circle")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(selectionFiles.isEmpty || hasActiveJob(for: repo))
                    .help("Fetch the selected files into the models folder")

                    if hasActiveJob(for: repo) {
                        Text("Already downloading.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Spacer(minLength: 0)
                }
            }
            .padding(4)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func kindLabel(_ kind: HFRepo.Kind) -> String {
        switch kind {
        case .gguf: return "GGUF · llama.cpp"
        case .mlx: return "MLX · safetensors"
        }
    }

    private var gatedNote: String {
        store.hfToken.isEmpty
            ? "This repo is gated or private. Accept its licence on HuggingFace, then add a read-only token in Settings → HuggingFace — the download will fail without one."
            : "This repo is gated or private. Your token is being sent; if you have not accepted its licence on HuggingFace the download will still be refused."
    }

    @ViewBuilder
    private func quantPicker(_ repo: HFRepo) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Quantisation")
                .font(.subheadline.weight(.medium))

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 84, maximum: 150), spacing: 8)],
                      alignment: .leading,
                      spacing: 8) {
                ForEach(repo.quants, id: \.self) { quant in
                    QuantChip(quant: quant,
                              isSelected: quant == selectedQuant,
                              size: bytes(of: repo, quant: quant)) {
                        selectedQuant = quant
                    }
                }
            }

            Text("Smaller quants use less memory and run faster; larger ones answer a little better. Q4_K_M is the usual compromise.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: Size / fit

    private var selectionFiles: [HFFile] {
        guard let repo else { return [] }
        return (try? Downloader.filesToDownload(repo: repo, quant: selectedQuant)) ?? []
    }

    private var selectionBytes: Int64 {
        selectionFiles.reduce(Int64(0)) { $0 + max(0, $1.size) }
    }

    private func bytes(of repo: HFRepo, quant: String) -> Int64 {
        let files = (try? Downloader.filesToDownload(repo: repo, quant: quant)) ?? []
        return files.reduce(Int64(0)) { $0 + max(0, $1.size) }
    }

    @ViewBuilder
    private var sizeSummary: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Text(Fmt.bytes(selectionBytes))
                    .font(.title3.weight(.semibold))
                    .monospacedDigit()
                Text(selectionFiles.count == 1 ? "in 1 file" : "in \(selectionFiles.count) files")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Download size \(Fmt.bytes(selectionBytes))")

            if let warning = fitWarning {
                NoticeRow(text: warning.text, systemImage: warning.icon, tint: warning.tint)
            } else if selectionBytes > 0 {
                Text("Fits comfortably in \(Fmt.bytes(store.stats.totalBytes)) of unified memory.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// Same thresholds the Models list uses, so a model does not read "tight" in one
    /// place and "fine" in the other.
    private var fitWarning: (text: String, icon: String, tint: Color)? {
        guard selectionBytes > 0 else { return nil }
        let installed = Fmt.bytes(store.stats.totalBytes)
        let size = Fmt.bytes(selectionBytes)

        switch ModelScanner.fitClass(size: selectionBytes) {
        case .fits:
            return nil
        case .tight:
            return ("\(size) is tight on \(installed) of unified memory — it will load, but close other apps first and keep the context short.",
                    "exclamationmark.triangle.fill", .orange)
        case .tooBig:
            return ("\(size) is larger than \(installed) of unified memory can usefully hold. It will download, but loading it will swap or fail.",
                    "exclamationmark.octagon.fill", .red)
        }
    }

    // MARK: Start

    private func startDownload(_ repo: HFRepo) {
        store.downloader.start(repo: repo,
                               quant: selectedQuant,
                               token: store.hfToken,
                               modelsRoot: store.modelsRoot)
    }

    private func hasActiveJob(for repo: HFRepo) -> Bool {
        store.downloader.jobs.contains {
            $0.repo == repo.repo && $0.quant == selectedQuant && Self.isActive($0.state)
        }
    }

    private static func isActive(_ state: DownloadJob.State) -> Bool {
        switch state {
        case .queued, .downloading, .installing: return true
        case .done, .failed, .cancelled: return false
        }
    }

    // MARK: Transfers

    @ViewBuilder
    private var transfersSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Transfers")
                .font(.headline)

            if store.downloader.jobs.isEmpty {
                Text("Nothing downloading. Inspect a repo above to start one.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(store.downloader.jobs) { job in
                    JobRow(job: job) {
                        store.downloader.cancel(job.id)
                    }
                }
            }
        }
    }

    /// A finished job means a new model on disk — pick it up without making the
    /// user hit Rescan.
    private func noteFinishedJobs(_ jobs: [DownloadJob]) {
        let finished = Set(jobs.filter { $0.state == .done }.map(\.id))
        let fresh = finished.subtracting(rescanned)
        guard !fresh.isEmpty else { return }
        rescanned.formUnion(fresh)
        Task { await store.rescanModels() }
    }
}

// MARK: - One transfer

@MainActor
private struct JobRow: View {
    let job: DownloadJob
    let cancel: () -> Void

    private var isActive: Bool {
        switch job.state {
        case .queued, .downloading, .installing: return true
        case .done, .failed, .cancelled: return false
        }
    }

    private var percent: Int { Int((job.fraction * 100).rounded()) }

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                header

                if isActive {
                    ProgressView(value: job.fraction)
                        .progressViewStyle(.linear)
                        .accessibilityLabel("Download progress")
                        .accessibilityValue("\(percent) percent")
                }

                detailLine

                if job.state == .failed, !job.error.isEmpty {
                    Text(job.error)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if job.state == .done {
                    Text(job.destination.path(percentEncoded: false))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                }
            }
            .padding(4)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text(job.repo)
                .font(.callout.weight(.medium))
                .lineLimit(1)
                .truncationMode(.middle)

            if !job.quant.isEmpty {
                Chip(text: job.quant)
            }

            Spacer(minLength: 8)

            stateLabel

            if isActive {
                Button(action: cancel) {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.borderless)
                .help("Cancel this download")
                .accessibilityLabel("Cancel download of \(job.repo)")
            }
        }
    }

    @ViewBuilder
    private var stateLabel: some View {
        switch job.state {
        case .queued:
            Label("Queued", systemImage: "clock")
                .font(.caption)
                .foregroundStyle(.secondary)
        case .downloading:
            Text("\(percent)%")
                .font(.caption)
                .monospacedDigit()
                .foregroundStyle(.secondary)
        case .installing:
            HStack(spacing: 5) {
                ProgressView()
                    .progressViewStyle(.circular)
                    .controlSize(.small)
                Text("Installing…")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Installing")
        case .done:
            Label("Installed", systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.green)
        case .failed:
            Label("Failed", systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.red)
        case .cancelled:
            Label("Cancelled", systemImage: "xmark.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var detailLine: some View {
        HStack(spacing: 10) {
            Text("\(Fmt.bytes(job.done)) / \(Fmt.bytes(job.total)) (\(percent)%)")
                .monospacedDigit()

            if job.state == .downloading, job.bytesPerSec > 0 {
                Text(Fmt.speed(job.bytesPerSec))
                    .monospacedDigit()
                let eta = Fmt.duration(job.eta)
                if !eta.isEmpty {
                    Text("\(eta) left")
                        .monospacedDigit()
                }
            }

            Spacer(minLength: 8)

            if !job.currentFile.isEmpty {
                Text(job.currentFile)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }
}

// MARK: - Small pieces

private struct Chip: View {
    let text: String
    var tint: Color = .secondary

    var body: some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .foregroundStyle(tint)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(.quaternary, in: Capsule())
            .accessibilityLabel(text)
    }
}

@MainActor
private struct QuantChip: View {
    let quant: String
    let isSelected: Bool
    let size: Int64
    let select: () -> Void

    var body: some View {
        Button(action: select) {
            VStack(spacing: 1) {
                Text(quant)
                    .font(.system(.callout, design: .monospaced))
                if size > 0 {
                    Text(Fmt.bytes(size))
                        .font(.caption2)
                        .monospacedDigit()
                        .opacity(0.8)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
            .padding(.horizontal, 8)
            .background(isSelected ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.quaternary),
                        in: RoundedRectangle(cornerRadius: 7))
            .foregroundStyle(isSelected ? AnyShapeStyle(Color.white) : AnyShapeStyle(Color.primary))
            .contentShape(RoundedRectangle(cornerRadius: 7))
        }
        .buttonStyle(.plain)
        .help(size > 0 ? "\(quant) — \(Fmt.bytes(size))" : quant)
        .accessibilityLabel("Quantisation \(quant)")
        .accessibilityValue(size > 0 ? Fmt.bytes(size) : "")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }
}

private struct NoticeRow: View {
    let text: String
    let systemImage: String
    let tint: Color

    var body: some View {
        Label {
            Text(text)
                .font(.caption)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: systemImage)
                .foregroundStyle(tint)
        }
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Token help

/// Kept next to the gated-repo warning so the Settings link is discoverable from
/// the place the user actually hits the wall.
private struct TokenHint: View {
    var body: some View {
        Link("Create a read-only token", destination: tokensPageURL)
            .font(.caption)
    }
}
