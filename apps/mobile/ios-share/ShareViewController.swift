import UIKit
import UniformTypeIdentifiers

/* GPTTrace share extension: receives text, an image, audio or a video from
 * another app, saves it into the App Group container shared with the main
 * app, and tells the person to open GPTTrace, which picks it up
 * (SharedInbox in SceneDelegate.swift) and checks it on the device. Nothing
 * is uploaded. */
class ShareViewController: UIViewController {
    private let group = "group.com.gpttrace.app"
    private let label = UILabel()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = UIColor(red: 0.04, green: 0.05, blue: 0.07, alpha: 1)
        label.textColor = .white
        label.numberOfLines = 0
        label.textAlignment = .center
        label.font = .preferredFont(forTextStyle: .headline)
        label.text = "Saving for GPTTrace…"
        label.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(label)
        NSLayoutConstraint.activate([
            label.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            label.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            label.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 24),
            label.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -24)
        ])
        Task { await save() }
    }

    private func save() async {
        guard let box = FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: group)?.appendingPathComponent("share") else { return finish("GPTTrace isn't set up for sharing (App Group missing).") }
        try? FileManager.default.removeItem(at: box)
        try? FileManager.default.createDirectory(at: box, withIntermediateDirectories: true)
        let providers = (extensionContext?.inputItems as? [NSExtensionItem] ?? []).flatMap { $0.attachments ?? [] }
        for type in [UTType.image, .movie, .audio] {
            if let p = providers.first(where: { $0.hasItemConformingToTypeIdentifier(type.identifier) }),
               let url = try? await p.loadFileCopy(type) {
                try? FileManager.default.copyItem(at: url, to: box.appendingPathComponent(url.lastPathComponent))
                return finish("Saved. Open GPTTrace to see the result.")
            }
        }
        for type in [UTType.plainText, .url] {
            if let p = providers.first(where: { $0.hasItemConformingToTypeIdentifier(type.identifier) }),
               let item = try? await p.loadItem(forTypeIdentifier: type.identifier) {
                let text = (item as? String) ?? (item as? URL)?.absoluteString ?? ""
                try? text.write(to: box.appendingPathComponent("shared-text.txt"), atomically: true, encoding: .utf8)
                return finish("Saved. Open GPTTrace to see the result.")
            }
        }
        finish("GPTTrace can check text, images, audio and video.")
    }

    @MainActor private func finish(_ message: String) {
        label.text = message
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { self.extensionContext?.completeRequest(returningItems: nil) }
    }
}

private extension NSItemProvider {
    // The provider's file is deleted when the handler returns, so copy it.
    func loadFileCopy(_ type: UTType) async throws -> URL {
        try await withCheckedThrowingContinuation { cont in
            _ = loadFileRepresentation(forTypeIdentifier: type.identifier) { url, err in
                guard let url = url else { return cont.resume(throwing: err ?? CocoaError(.fileNoSuchFile)) }
                let copy = FileManager.default.temporaryDirectory.appendingPathComponent(url.lastPathComponent)
                try? FileManager.default.removeItem(at: copy)
                do { try FileManager.default.copyItem(at: url, to: copy); cont.resume(returning: copy) } catch { cont.resume(throwing: error) }
            }
        }
    }
}
