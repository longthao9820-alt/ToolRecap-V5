# ToolRecap V5.0.1

- Adds a Delete action for unfinished projects and their private checkpoints while preserving published MP4/SRT files.
- Disables bitmap subtitle OCR and stops source preparation when no usable English text subtitle is available.
- Uses `Output_<last-source-name>_Highlight` or `Output_<last-source-name>_Recap` for automatic output folders.
- Verifies cross-drive publication through a destination-volume temporary file, avoiding Windows error 17.
