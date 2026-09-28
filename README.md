# ToolRecap V5

ToolRecap V5 is a portable Windows desktop application with one deliberately small workflow:

```text
Prepare sources locally
→ AI Gateway analyses the user's prompt and returns final Render JSON
→ validate the Render JSON technically
→ optional creator review
→ VoiceStudio/FFmpeg render and publish
```

The Gateway owns scene selection, story structure, Highlight selection, and narration writing. The desktop application does not run separate Scanner, Catalog, Planner, Vision, Writer, or Finalizer editorial phases. Recap and Highlight share the same pipeline and Render JSON contract.

The existing AI Gateway and VoiceStudio settings remain compatible. V5 also retains saved-project resume, optional review, narration fitting, original-audio mixing, subtitle generation, and NVIDIA-accelerated rendering where supported.

## Install the portable release

1. Open the repository's **Releases** page.
2. Download `ToolRecapV5-v5.0.4-windows-portable.zip` from the latest release.
3. Extract the entire ZIP to a writable local folder.
4. Keep the complete extracted `ToolRecapV5` folder together.
5. Run `ToolRecapV5.exe` from that folder.
6. Open **Settings** and configure the required AI Gateway models/credentials. Configure VoiceStudio when using Recap narration.
7. Choose **Recap** or **Highlight**, select source media, enter the corresponding prompt, and start the workflow.

For creator review, enable **Review before render** in the selected workspace. The app pauses after validating the Gateway's Render JSON; approve it to continue without repeating the AI request.

No separate Python installation is required for normal portable use. FFmpeg, FFprobe, the Python runtime, and required application libraries are included in the one-folder package.

RapidOCR and its local ONNX models are included for extracting image-based subtitles. This is source preparation, not editorial analysis. Source preparation does not invoke Whisper automatically. If no usable English subtitle can be extracted, the app reports the track and extraction error.

V5.0.1 disables bitmap-subtitle OCR in the active V5 workflow. Only directly extractable English text subtitles are accepted; otherwise source preparation stops with an explicit error. Unfinished projects can be deleted from either workspace without removing already-published files.

V5.0.2 makes folder discovery visible as it happens: episode rows appear incrementally, **Real-Time Status / Activity** reports the scan and the latest detected file, and **Create Highlights** becomes available as soon as the completed source list is ready. Saved-project activity can no longer overwrite an active source scan.

V5.0.3 places automatic output inside the selected source folder and names it from that folder: `<source-folder>/Output_<source-folder-name>_Highlight` or `<source-folder>/Output_<source-folder-name>_Recap`. A manually configured output directory still takes precedence.

V5.0.4 gives Highlight a source-audio-only downstream path: Voice and Narration Fit are skipped, Recap gain/duck controls are not applied, and Highlight loudness normalization is opt-in. Tailscale `.ts.net` service URLs with valid ports are normalized consistently, bypass machine HTTP proxies, retain TLS verification, and report actionable connection guidance.

## Windows and hardware requirements

- 64-bit Windows 10 or Windows 11.
- Sufficient local storage for source preparation, caches, temporary render files, and published video.
- Network access to any configured AI Gateway or remote VoiceStudio service.
- NVIDIA acceleration requires a compatible NVIDIA GPU and current NVIDIA driver.
- The full CUDA Toolkit is **not** required or bundled.

ToolRecap can use CPU rendering when GPU acceleration is deliberately disabled. If NVIDIA mode is selected, NVENC initialization failures are reported explicitly.

## Portable state and privacy

The portable application folder contains only application runtime files. User settings, encrypted credentials, project checkpoints, caches, and update staging data are stored separately under:

```text
%LOCALAPPDATA%\ToolRecapV4\
```

API credentials are stored with Windows DPAPI and are not included in release packages. Copying the portable folder to another computer does not transfer user credentials or projects.

## Self-check

The frozen executable supports a zero-project runtime check:

```text
ToolRecapV5.exe --selfcheck --selfcheck-json selfcheck.json
```

Exit code `0` means all mandatory portable-runtime checks passed. Optional models, credentials, or hardware may be reported as warnings.

## Development

Source tests:

```powershell
pytest -q
python -m compileall -q toolrecap_v4
```

Build the one-folder portable package on Windows:

```powershell
python build_portable.py
```

The generated `dist/`, `build/`, and `release/` directories are intentionally excluded from Git history. Downloadable binaries are distributed through GitHub Releases.
