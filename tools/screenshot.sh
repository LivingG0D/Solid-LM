#!/bin/bash
# Capture the SolidChat window for the README.
#
# Window ONLY, never the full screen: nothing else on the desktop — other windows,
# menu-bar extras, file names — can leak into a published image.
#
# Needs Screen Recording for the app running this shell (System Settings >
# Privacy & Security > Screen Recording). macOS caches that decision at process
# launch, so after granting it you must quit and reopen that app.
#
#   tools/screenshot.sh docs chat
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: screenshot.sh <outdir> <name>}"
NAME="${2:?usage: screenshot.sh <outdir> <name>}"
mkdir -p "$OUT"

for bin in winid cap3; do
  [ -x "$HERE/$bin" ] || swiftc -parse-as-library -O -o "$HERE/$bin" "$HERE/$bin.swift" 2>/dev/null \
    || swiftc -O -o "$HERE/$bin" "$HERE/$bin.swift"
done

WID="$("$HERE/winid" | head -1)"
[ -n "$WID" ] || { echo "no SolidChat window — is the app running?" >&2; exit 1; }

# ScreenCaptureKit rather than the screencapture CLI: it reports a TCC denial as a
# real error instead of writing a blank frame.
"$HERE/cap3" "$WID" "$OUT/$NAME.png"

W=$(sips -g pixelWidth "$OUT/$NAME.png" | awk '/pixelWidth/{print $2}')
[ "$W" -gt 1800 ] && sips -Z 1600 "$OUT/$NAME.png" >/dev/null   # Retina 2x -> sane repo size
sips -g pixelWidth -g pixelHeight "$OUT/$NAME.png" | awk '/pixel/{printf "%s ", $2} END{print ""}'
