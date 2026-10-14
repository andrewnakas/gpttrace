# GPTTrace apps

All apps run the same on-device detection engine as gpttrace.com. The web
bundle they ship is built from the site by `node tools/build-app.js` (from the
repo root) → `apps/web/`. Models are downloaded from gpttrace.com on first use
and cached, as the extension does; nothing the user checks is uploaded.

| Folder | Platforms | Shell |
|---|---|---|
| `safari/` | Safari on macOS and iOS | Xcode project from `safari-web-extension-converter` (see its README) |
| `mobile/` | iOS, Android | Capacitor (`npm run ios`, `npm run android` inside `mobile/`) |
| `desktop/` | macOS, Windows, Linux | Tauri |

The native shells serve the bundle from a local scheme that maps
extensionless paths to the root `index.html`. Because of that, `build-app.js`
rewrites page links to `/slug/index.html`.

## Share to GPTTrace

| Platform | How a share arrives | Status |
|---|---|---|
| PWA | Manifest `share_target` → `sw.js` | `node tests/pwa.js` |
| Android | `SEND` intent filter, then `MainActivity.java` → `window.gtShare` | Compiled into the debug APK |
| iOS | Share extension, see below | Code only so far |

**iOS share extension (one-time Xcode setup):**

1. Open `mobile/ios/App/App.xcodeproj`.
2. File › New › Target › Share Extension, named "GPTTraceShare". When asked, don't activate the scheme.
3. Replace the generated `ShareViewController.swift` with `mobile/ios-share/ShareViewController.swift`. Delete `MainInterface.storyboard`. In the extension's Info.plist, remove `NSExtensionMainStoryboard` and set `NSExtensionPrincipalClass` to `$(PRODUCT_MODULE_NAME).ShareViewController`.
4. In the extension's Info.plist, set `NSExtensionAttributes › NSExtensionActivationRule` to accept text, URLs, one image, one movie and one audio file.
5. Add the App Groups capability with `group.com.gpttrace.app` to both the App target and the GPTTraceShare target.

`SharedInbox` in `SceneDelegate.swift` picks the item up when the app comes to the front.
