import SwiftUI

/// Browse HuggingFace from inside the app.
///
/// Finding a repo is all this does. Once one is picked it hands off to the existing
/// `Downloader` — quant selection, parallel ranged fetches, resume and atomic install
/// already live there and are not worth a second implementation.
@MainActor
struct MarketView: View {
    @Environment(AppStore.self) private var store

    @State private var query = ""
    @State private var format: MarketFormat = .gguf
    @State private var sort: MarketSort = .trending
    @State private var results: [MarketModel] = []
    @State private var loading = false
    @State private var error: String?

    /// The repo whose files are being resolved, and the result once they are.
    @State private var opening: String?
    @State private var detail: HFRepo?
    @State private var detailFor: MarketModel?
    @State private var selectedQuant = ""

    private var installedIDs: Set<String> {
        // A repo is "installed" when a local model's publisher/name matches it.
        Set(store.models.map { $0.id.lowercased() })
    }

    var body: some View {
        VStack(spacing: 0) {
            controls
            Divider()
            content
        }
        .background(.background)
        .task(id: TaskKey(query: "", format: format, sort: sort)) { await load() }
        .sheet(item: $detailFor) { model in
            MarketDetailSheet(model: model,
                              repo: detail,
                              selectedQuant: $selectedQuant,
                              onDownload: { start(model, $0) },
                              onClose: { detailFor = nil; detail = nil })
        }
    }

    /// `.task(id:)` needs a single Equatable value. Search text is deliberately not in
    /// here — retriggering on every keystroke would hammer the API.
    private struct TaskKey: Equatable { var query: String; var format: MarketFormat; var sort: MarketSort }

    // MARK: Controls

    private var controls: some View {
        HStack(spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search HuggingFace", text: $query)
                    .textFieldStyle(.plain)
                    .onSubmit { Task { await load() } }
                if !query.isEmpty {
                    Button {
                        query = ""
                        Task { await load() }
                    } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                        .help("Clear search")
                        .accessibilityLabel("Clear search")
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .frame(maxWidth: 320)

            Picker("Format", selection: $format) {
                ForEach(MarketFormat.allCases, id: \.self) { f in
                    Text("\(f.label) · \(f.engineNote)").tag(f)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()

            Picker("Sort", selection: $sort) {
                ForEach(MarketSort.allCases, id: \.self) { Text($0.label).tag($0) }
            }
            .labelsHidden()
            .fixedSize()

            Spacer()

            if loading { ProgressView().controlSize(.small) }
            Button {
                Task { await load() }
            } label: { Image(systemName: "arrow.clockwise") }
                .help("Refresh")
                .accessibilityLabel("Refresh results")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    // MARK: Content

    @ViewBuilder
    private var content: some View {
        if let error {
            ContentUnavailableView {
                Label("Could not reach HuggingFace", systemImage: "wifi.exclamationmark")
            } description: {
                Text(error).textSelection(.enabled)
            } actions: {
                Button("Try again") { Task { await load() } }
            }
        } else if results.isEmpty && !loading {
            ContentUnavailableView {
                Label("Nothing found", systemImage: "magnifyingglass")
            } description: {
                Text(query.isEmpty
                     ? "No \(format.label) models came back. Check the connection and try again."
                     : "No \(format.label) models match “\(query)”.")
            }
        } else {
            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(results) { model in
                        MarketRow(model: model,
                                  installed: installedIDs.contains(model.id.lowercased()),
                                  busy: opening == model.id,
                                  onOpen: { open(model) })
                    }
                }
                .padding(16)
            }
        }
    }

    // MARK: Actions

    private func load() async {
        loading = true
        error = nil
        defer { loading = false }
        do {
            results = try await MarketClient.search(query: query, format: format,
                                                    sort: sort, token: store.hfToken)
        } catch {
            self.error = error.localizedDescription
            results = []
        }
    }

    /// Resolve the repo's actual files so the user picks a real quant rather than guessing.
    private func open(_ model: MarketModel) {
        opening = model.id
        Task {
            defer { opening = nil }
            do {
                let repo = try await store.downloader.inspect(model.id, token: store.hfToken)
                detail = repo
                selectedQuant = Self.preferredQuant(from: repo.quants)
                detailFor = model
            } catch {
                self.error = error.localizedDescription
            }
        }
    }

    private func start(_ model: MarketModel, _ quant: String) {
        guard let repo = detail else { return }
        store.downloader.start(repo: repo, quant: quant,
                               token: store.hfToken, modelsRoot: store.modelsRoot)
        detailFor = nil
        detail = nil
    }

    /// Q4_K_M first: the usual quality-per-byte sweet spot for language models.
    /// (Image models are the opposite — see `ImageModel.quantWarning`.)
    static func preferredQuant(from quants: [String]) -> String {
        let order = ["Q4_K_M", "Q4_K_S", "IQ4_XS", "Q5_K_M", "Q6_K", "Q8_0"]
        for want in order where quants.contains(want) { return want }
        return quants.first ?? ""
    }
}

// MARK: - Row

private struct MarketRow: View {
    let model: MarketModel
    let installed: Bool
    let busy: Bool
    let onOpen: () -> Void

    @State private var hovering = false

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(model.name)
                        .font(.system(.body, design: .monospaced))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if installed {
                        Label("Installed", systemImage: "checkmark.circle.fill")
                            .labelStyle(.iconOnly)
                            .foregroundStyle(.green)
                            .help("Already in your models folder")
                    }
                    if model.gated {
                        Label("Gated", systemImage: "lock.fill")
                            .labelStyle(.iconOnly)
                            .foregroundStyle(.orange)
                            .help("Gated repo — needs a HuggingFace token with access")
                    }
                }

                HStack(spacing: 8) {
                    Text(model.author)
                    Label(model.downloadsLabel, systemImage: "arrow.down.circle")
                    Label("\(model.likes)", systemImage: "heart")
                    ForEach(model.displayTags, id: \.self) { tag in
                        Text(tag)
                            .font(.caption2)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(.quaternary, in: Capsule())
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            }

            Spacer(minLength: 8)

            Button(action: onOpen) {
                if busy {
                    ProgressView().controlSize(.small)
                } else {
                    Text(installed ? "Add quant" : "Get")
                }
            }
            .disabled(busy)
            .frame(minWidth: 74)
            .help(installed
                  ? "Already installed — download a different quantisation"
                  : "Look at the files in this repo and pick a quantisation")

            Link(destination: URL(string: "https://huggingface.co/\(model.id)")!) {
                Image(systemName: "arrow.up.forward.square")
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .help("Open the model card on huggingface.co")
            .accessibilityLabel("Open on HuggingFace")
        }
        .padding(12)
        .background(hovering ? AnyShapeStyle(.quaternary) : AnyShapeStyle(.clear),
                    in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(.separator))
        .onHover { hovering = $0 }
    }
}

// MARK: - Detail sheet

private struct MarketDetailSheet: View {
    let model: MarketModel
    let repo: HFRepo?
    @Binding var selectedQuant: String
    let onDownload: (String) -> Void
    let onClose: () -> Void

    private var totalBytes: Int64 {
        guard let repo else { return 0 }
        guard repo.kind == .gguf, !selectedQuant.isEmpty else {
            return repo.files.reduce(0) { $0 + $1.size }
        }
        // Mirror what Downloader actually fetches. A substring match would also count
        // `…-mmproj-Q8_0.gguf`, which is how this first showed "600 MB" for a 27B model.
        let match = repo.files.filter { f in
            guard f.path.lowercased().hasSuffix(".gguf") else { return true }   // configs, tokenizer
            return !Downloader.isCompanionGGUF(f.path)
                && (Downloader.parseQuant(fromPath: f.path) ?? "") == selectedQuant.uppercased()
        }
        return match.reduce(0) { $0 + $1.size }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 3) {
                Text(model.name).font(.headline).textSelection(.enabled)
                Text(model.author).font(.subheadline).foregroundStyle(.secondary)
            }

            if let repo {
                if repo.kind == .gguf && !repo.quants.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Quantisation").font(.caption).foregroundStyle(.secondary)
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 6) {
                                ForEach(repo.quants, id: \.self) { q in
                                    Button(q) { selectedQuant = q }
                                        .buttonStyle(.bordered)
                                        .tint(q == selectedQuant ? .accentColor : .secondary)
                                }
                            }
                        }
                    }
                } else {
                    Text("MLX repo — the whole folder is downloaded.")
                        .font(.caption).foregroundStyle(.secondary)
                }

                HStack(spacing: 6) {
                    Text(Fmt.bytes(totalBytes)).font(.system(.body, design: .monospaced))
                    if totalBytes > 19 * 1_073_741_824 {
                        Label("Larger than usable memory on a 24 GB machine",
                              systemImage: "exclamationmark.triangle.fill")
                            .font(.caption).foregroundStyle(.orange)
                    }
                }

                if model.gated {
                    Label("This repo is gated. Without an authorised token in Settings the download will fail.",
                          systemImage: "lock.fill")
                        .font(.caption).foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else {
                HStack(spacing: 8) { ProgressView().controlSize(.small); Text("Reading the repo…") }
                    .font(.caption).foregroundStyle(.secondary)
            }

            HStack {
                Link("Model card", destination: URL(string: "https://huggingface.co/\(model.id)")!)
                    .font(.caption)
                Spacer()
                Button("Cancel", action: onClose).keyboardShortcut(.cancelAction)
                Button("Download") { onDownload(selectedQuant) }
                    .keyboardShortcut(.defaultAction)
                    .disabled(repo == nil)
            }
        }
        .padding(20)
        .frame(width: 460)
    }
}
