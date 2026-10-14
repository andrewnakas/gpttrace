import UIKit
import Capacitor

class SceneDelegate: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?

    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
        guard let windowScene = scene as? UIWindowScene else { return }

        window = UIWindow(windowScene: windowScene)
        window?.rootViewController = CAPBridgeViewController()
        window?.makeKeyAndVisible()

        SceneDelegateProxy.shared.scene(scene, willConnectTo: session, options: connectionOptions)
    }

    func scene(_ scene: UIScene, openURLContexts URLContexts: Set<UIOpenURLContext>) {
        SceneDelegateProxy.shared.scene(scene, openURLContexts: URLContexts)
    }

    func scene(_ scene: UIScene, continue userActivity: NSUserActivity) {
        SceneDelegateProxy.shared.scene(scene, continue: userActivity)
    }

    // "Share to GPTTrace": the share extension (ios-share/) parks the shared
    // text or file in the App Group container; pick it up whenever the app
    // comes to the front and hand it to the page (window.gtShare).
    func sceneDidBecomeActive(_ scene: UIScene) {
        SharedInbox.deliver(to: window?.rootViewController as? CAPBridgeViewController)
    }
}

enum SharedInbox {
    static let group = "group.com.gpttrace.app"

    static func deliver(to bridge: CAPBridgeViewController?) {
        guard let bridge = bridge,
              let box = FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: group)?.appendingPathComponent("share"),
              let items = try? FileManager.default.contentsOfDirectory(at: box, includingPropertiesForKeys: nil),
              let item = items.first else { return }
        var payload: [String: String] = [:]
        if item.lastPathComponent == "shared-text.txt" {
            payload["text"] = (try? String(contentsOf: item, encoding: .utf8)) ?? ""
        } else {
            // Copy out of the group container into the app's own temp folder,
            // which Capacitor serves to the page as a local file URL.
            let dest = FileManager.default.temporaryDirectory.appendingPathComponent(item.lastPathComponent)
            try? FileManager.default.removeItem(at: dest)
            guard (try? FileManager.default.copyItem(at: item, to: dest)) != nil else { return }
            payload["url"] = "capacitor://localhost/_capacitor_file_" + dest.path
            payload["name"] = item.lastPathComponent
        }
        for i in items { try? FileManager.default.removeItem(at: i) }
        guard let data = try? JSONSerialization.data(withJSONObject: payload), let json = String(data: data, encoding: .utf8) else { return }
        let js = "(function f(){window.gtShare?gtShare(\(json)):setTimeout(f,200)})()"
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { bridge.webView?.evaluateJavaScript(js) }
    }
}
