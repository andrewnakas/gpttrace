# GPTTrace for Safari (macOS and iOS)

Generated with `xcrun safari-web-extension-converter extension/dist`. The Xcode
project does not copy the extension: it references `../../../../extension/dist`,
so build the extension first and every Chrome build feeds Safari too.

    npm run build:extension
    open apps/safari/GPTTrace/GPTTrace.xcodeproj

- Set your Apple team under Signing & Capabilities on all four targets
  (GPTTrace iOS/macOS and the two Extension targets). The bundle id is
  `com.gpttrace.app`; the extensions are `com.gpttrace.app.Extension`.
- Local test on a Mac: run the "GPTTrace (macOS)" scheme. In Safari, go to
  Settings › Advanced and turn on "Show features for web developers", then
  Develop › "Allow Unsigned Extensions". Enable GPTTrace in Settings › Extensions.
- iPhone/iPad: run "GPTTrace (iOS)" on a device, then Settings › Apps › Safari ›
  Extensions › GPTTrace.
- Release: Product › Archive for each platform, then upload to App Store Connect.

Checked so far: the macOS scheme builds unsigned (`xcodebuild ... CODE_SIGN_IDENTITY=-`).
Not yet checked in Safari itself: `windows.create({type: "popup"})` on iOS (iOS
Safari has no popup windows, so the result page opens as a tab) and ORT WASM
memory on iPhone.
