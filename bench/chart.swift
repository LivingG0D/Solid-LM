import AppKit
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

// Renders the measured benchmark as a PNG for the README. Values are the real
// numbers from the run; nothing here is illustrative.
struct Bar { let label: String; let value: Double; let highlight: Bool; let note: String }

func render(title: String, subtitle: String, bars: [Bar], unit: String, out: String) {
    let W = 1000, rowH = 52, top = 104, bottom = 44
    let H = top + rowH * bars.count + bottom
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    guard let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8, bytesPerRow: 0,
                              space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return }

    // GitHub-dark friendly background.
    ctx.setFillColor(CGColor(red: 0.05, green: 0.06, blue: 0.09, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: W, height: H))

    func text(_ s: String, x: CGFloat, y: CGFloat, size: CGFloat, color: CGColor,
              bold: Bool = false, mono: Bool = false, rightAlignedTo: CGFloat? = nil) {
        let font = mono
            ? NSFont.monospacedSystemFont(ofSize: size, weight: bold ? .bold : .regular)
            : NSFont.systemFont(ofSize: size, weight: bold ? .semibold : .regular)
        let attrs: [NSAttributedString.Key: Any] = [
            .font: font, .foregroundColor: NSColor(cgColor: color) ?? .white,
        ]
        let str = NSAttributedString(string: s, attributes: attrs)
        let line = CTLineCreateWithAttributedString(str)
        var drawX = x
        if let right = rightAlignedTo {
            drawX = right - CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
        }
        ctx.textPosition = CGPoint(x: drawX, y: y)
        CTLineDraw(line, ctx)
    }

    let white = CGColor(red: 0.94, green: 0.96, blue: 0.98, alpha: 1)
    let grey  = CGColor(red: 0.55, green: 0.60, blue: 0.68, alpha: 1)
    let blue  = CGColor(red: 0.23, green: 0.51, blue: 0.96, alpha: 1)
    let green = CGColor(red: 0.20, green: 0.78, blue: 0.45, alpha: 1)

    text(title, x: 32, y: CGFloat(H - 44), size: 24, color: white, bold: true)
    text(subtitle, x: 32, y: CGFloat(H - 70), size: 14, color: grey)

    let labelW: CGFloat = 300
    let barX = labelW + 44
    let barMaxW = CGFloat(W) - barX - 290   // room for the value label AND the right-aligned note
    let maxVal = bars.map(\.value).max() ?? 1

    for (i, b) in bars.enumerated() {
        let y = CGFloat(H - top - (i + 1) * rowH) + 14
        text(b.label, x: 32, y: y + 6, size: 14, color: b.highlight ? white : grey,
             bold: b.highlight, mono: true)

        let w = max(2, CGFloat(b.value / maxVal) * barMaxW)
        let rect = CGRect(x: barX, y: y, width: w, height: 24)
        ctx.setFillColor(b.highlight ? green : blue.copy(alpha: 0.55)!)
        ctx.addPath(CGPath(roundedRect: rect, cornerWidth: 5, cornerHeight: 5, transform: nil))
        ctx.fillPath()

        text(String(format: "%.2f %@", b.value, unit), x: barX + w + 12, y: y + 6,
             size: 14, color: b.highlight ? green : white, bold: b.highlight, mono: true)
        if !b.note.isEmpty {
            text(b.note, x: 0, y: y + 6, size: 13, color: grey, mono: true,
                 rightAlignedTo: CGFloat(W) - 24)
        }
    }

    guard let img = ctx.makeImage(),
          let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: out) as CFURL,
                                                     UTType.png.identifier as CFString, 1, nil)
    else { return }
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
    print("wrote \(out)")
}

let dir = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "."

render(title: "Gemma4-12B QAT Q4_K_M — dense",
       subtitle: "Identical model file and draft head. 200 tokens, temp 0, mean of 3 runs. Apple M5, 24 GB.",
       bars: [
        Bar(label: "LM Studio 2.27.1",       value: 14.12, highlight: false, note: "baseline"),
        Bar(label: "upstream b10229",        value: 14.14, highlight: false, note: "1.00x"),
        Bar(label: "PrismML b9599",          value: 14.04, highlight: false, note: "0.99x"),
        Bar(label: "upstream + MTP",         value: 20.04, highlight: false, note: "1.42x"),
        Bar(label: "LM Studio + MTP",        value: 21.05, highlight: false, note: "1.49x"),
        Bar(label: "PrismML + MTP",          value: 25.51, highlight: true,  note: "1.81x"),
       ], unit: "tok/s", out: "\(dir)/bench-12b.png")

render(title: "Gemma4-26B-A4B QAT Q4_K_M — mixture of experts",
       subtitle: "Same file, same draft. On the MoE the ranking flips — LM Studio edges ahead.",
       bars: [
        Bar(label: "LM Studio 2.27.1",  value: 28.86, highlight: false, note: "baseline"),
        Bar(label: "PrismML + MTP",     value: 36.90, highlight: false, note: "1.28x"),
        Bar(label: "LM Studio + MTP",   value: 37.99, highlight: true,  note: "1.32x"),
       ], unit: "tok/s", out: "\(dir)/bench-26b.png")

render(title: "Speculative decoding is the only knob that moves",
       subtitle: "Qwen3.5-9B Q4_K_M for the flags, Gemma4-12B for the draft depths. Everything else sits on the memory wall.",
       bars: [
        Bar(label: "threads 10",        value: 19.40, highlight: false, note: "worse than 4"),
        Bar(label: "mmap off",          value: 19.30, highlight: false, note: ""),
        Bar(label: "flash-attn off",    value: 20.03, highlight: false, note: "no effect"),
        Bar(label: "KV cache Q8",       value: 20.03, highlight: false, note: "no effect"),
        Bar(label: "stock (baseline)",  value: 20.05, highlight: false, note: "bandwidth limit"),
        Bar(label: "predicted ceiling", value: 20.50, highlight: false, note: "115 GB/s / 5.23 GiB"),
       ], unit: "tok/s", out: "\(dir)/bench-knobs.png")

render(title: "MTP draft depth — Gemma4-12B",
       subtitle: "Deeper drafting loses: past 3 the acceptance rate falls faster than the batching gain.",
       bars: [
        Bar(label: "off",       value: 14.14, highlight: false, note: "1.00x"),
        Bar(label: "n-max 6",   value: 15.49, highlight: false, note: "1.10x"),
        Bar(label: "n-max 5",   value: 17.65, highlight: false, note: "1.25x"),
        Bar(label: "n-max 4",   value: 21.58, highlight: false, note: "1.53x"),
        Bar(label: "n-max 2",   value: 22.00, highlight: false, note: "1.56x"),
        Bar(label: "n-max 3",   value: 25.51, highlight: true,  note: "1.80x  <- default"),
       ], unit: "tok/s", out: "\(dir)/bench-draft.png")
