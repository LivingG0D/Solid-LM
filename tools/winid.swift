import CoreGraphics
import Foundation
// Window id of the SolidChat main window, largest first.
let opts: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
let list = (CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String: Any]]) ?? []
let mine = list.filter { ($0[kCGWindowOwnerName as String] as? String)?.contains("SolidChat") == true }
    .compactMap { w -> (Int, CGFloat)? in
        guard let n = w[kCGWindowNumber as String] as? Int,
              let b = w[kCGWindowBounds as String] as? [String: Any],
              let width = b["Width"] as? CGFloat, let height = b["Height"] as? CGFloat,
              width > 300, height > 300 else { return nil }
        return (n, width * height)
    }
    .sorted { $0.1 > $1.1 }
if let best = mine.first { print(best.0) }
