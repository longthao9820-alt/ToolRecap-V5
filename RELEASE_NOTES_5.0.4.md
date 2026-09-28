# ToolRecap V5.0.4

- Skips VoiceStudio and Narration Fit completely for Highlight projects.
- Replaces misleading Highlight narration/mix activity with original-source-audio status.
- Prevents Recap gain and ducking controls from changing Highlight audio.
- Preserves Highlight source loudness by default; adds an explicit opt-in loudness-normalization setting.
- Accepts and preserves valid Tailscale endpoints such as `https://desktop-name.tailnet.ts.net:8443`.
- Bypasses machine HTTP proxies for `.ts.net` traffic while retaining normal HTTPS certificate verification.
- Validates ports as `1-65535` and gives guidance for Tailscale Serve versus direct non-TLS backend ports.
- Applies the same endpoint policy to AI Gateway and VoiceStudio connections.
