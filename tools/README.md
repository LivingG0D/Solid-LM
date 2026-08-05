# Screenshot tooling

How the UI screenshots in the top-level README were captured, so they can be regenerated
rather than hand-cropped.

```bash
open -a /Applications/SolidChat.app
tools/screenshot.sh docs chat        # -> docs/chat.png
```

`winid.swift` finds the SolidChat main window; `cap3.swift` captures **that window only** through
ScreenCaptureKit. Window-scoped on purpose — a full-screen grab would put whatever else is on the
desktop into a public image.

ScreenCaptureKit rather than the `screencapture` CLI because it reports a permission denial as a
real error (`The user declined TCCs for application, window, display capture`) instead of silently
writing a blank frame.

## Permission

Screen Recording must be granted to whichever app runs this shell, in
**System Settings ▸ Privacy & Security ▸ Screen Recording**.

macOS caches that decision **at process launch**. Granting it while the app is running is not
enough — quit and reopen the app, or the capture keeps failing with the denial above even though
the checkbox is ticked.
