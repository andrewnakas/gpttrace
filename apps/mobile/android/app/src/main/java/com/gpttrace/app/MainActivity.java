package com.gpttrace.app;

import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import com.getcapacitor.BridgeActivity;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import org.json.JSONObject;

/* "Share to GPTTrace": text, images, audio and video shared from other apps
 * are handed to the page's window.gtShare (js/ui/mount.js). A shared file is
 * copied into the app's private cache and read by the page through
 * Capacitor's local file URL; nothing leaves the device. */
public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        handleShare(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        handleShare(intent);
    }

    private void handleShare(Intent intent) {
        if (intent == null || !Intent.ACTION_SEND.equals(intent.getAction())) return;
        try {
            JSONObject d = new JSONObject();
            Uri uri = intent.getParcelableExtra(Intent.EXTRA_STREAM);
            if (uri != null) {
                String name = displayName(uri);
                File dir = new File(getCacheDir(), "share");
                dir.mkdirs();
                for (File old : dir.listFiles()) old.delete();
                File out = new File(dir, name.replaceAll("[^A-Za-z0-9._-]", "_"));
                try (InputStream in = getContentResolver().openInputStream(uri); OutputStream o = new FileOutputStream(out)) {
                    byte[] buf = new byte[65536];
                    for (int n; (n = in.read(buf)) > 0; ) o.write(buf, 0, n);
                }
                d.put("url", bridge.getLocalUrl() + "/_capacitor_file_" + out.getAbsolutePath());
                d.put("name", name);
                d.put("type", intent.getType());
            } else {
                String text = intent.getStringExtra(Intent.EXTRA_TEXT);
                if (text == null || text.trim().isEmpty()) return;
                d.put("text", text);
            }
            intent.setAction(null); // handle each share once
            String js = "(function f(){window.gtShare?gtShare(" + d + "):setTimeout(f,200)})()";
            bridge.getWebView().post(() -> bridge.getWebView().evaluateJavascript(js, null));
        } catch (Exception e) {
            android.util.Log.w("GPTTrace", "share failed", e);
        }
    }

    private String displayName(Uri uri) {
        try (Cursor c = getContentResolver().query(uri, new String[] { OpenableColumns.DISPLAY_NAME }, null, null, null)) {
            if (c != null && c.moveToFirst()) return c.getString(0);
        } catch (Exception ignored) {}
        return "shared";
    }
}
