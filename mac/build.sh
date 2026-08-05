#!/bin/bash
# Build SolidChat.app with the Command Line Tools toolchain (no Xcode required).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$HERE/build/SolidChat.app"
SRC="$HERE/Sources/SolidChat"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# Whole-module build: every .swift under Sources compiled as one module.
swiftc -parse-as-library \
  -O -whole-module-optimization \
  -target arm64-apple-macosx26.0 \
  -o "$APP/Contents/MacOS/SolidChat" \
  $(find "$SRC" -name '*.swift')

[ -f "$HERE/SolidChat.icns" ] && cp "$HERE/SolidChat.icns" "$APP/Contents/Resources/"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>SolidChat</string>
  <key>CFBundleIdentifier</key><string>local.solidchat.app</string>
  <key>CFBundleName</key><string>SolidChat</string>
  <key>CFBundleDisplayName</key><string>SolidChat</string>
  <key>CFBundleIconFile</key><string>SolidChat</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>26.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSSupportsAutomaticGraphicsSwitching</key><true/>
</dict></plist>
PLIST

# Ad-hoc signature. Unsigned SwiftUI binaries are killed on launch by Gatekeeper.
codesign --force --sign - "$APP" 2>/dev/null

echo "built $APP"
