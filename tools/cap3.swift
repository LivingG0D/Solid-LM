import Foundation
import ScreenCaptureKit
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

@main struct Cap {
    static func main() async {
        let target = CGWindowID(UInt32(CommandLine.arguments[1])!)
        let out = URL(fileURLWithPath: CommandLine.arguments[2])
        do {
            // This is the sanctioned path on macOS 26. If TCC has not been granted
            // to the responsible app it throws rather than returning a blank frame.
            let content = try await SCShareableContent.excludingDesktopWindows(false,
                                                                               onScreenWindowsOnly: true)
            guard let win = content.windows.first(where: { $0.windowID == target }) else {
                print("window \(target) not in shareable content"); exit(1)
            }
            let cfg = SCStreamConfiguration()
            cfg.width = Int(win.frame.width * 2)
            cfg.height = Int(win.frame.height * 2)
            cfg.showsCursor = false
            let filter = SCContentFilter(desktopIndependentWindow: win)
            let img = try await SCScreenshotManager.captureImage(contentFilter: filter,
                                                                configuration: cfg)
            guard let d = CGImageDestinationCreateWithURL(out as CFURL,
                                                         UTType.png.identifier as CFString, 1, nil)
            else { print("dest fail"); exit(1) }
            CGImageDestinationAddImage(d, img, nil)
            CGImageDestinationFinalize(d)
            print("wrote \(img.width)x\(img.height)")
        } catch {
            print("capture failed: \(error.localizedDescription)"); exit(1)
        }
    }
}
