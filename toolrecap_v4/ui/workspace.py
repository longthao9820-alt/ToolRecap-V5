"""Shared Recap/Highlight workspace UI components."""
from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk
from typing import Callable


@dataclass(frozen=True)
class WorkspaceLabels:
    source_title: str
    select_file: str
    select_folder: str
    saved_project: str
    resume_saved: str
    prompt_title: str
    queue_title: str
    start: str
    clear: str


class WorkspacePage(ttk.Frame):
    """One reusable workspace shell; business commands are injected by mode."""

    COLUMNS = ("episode", "source_video", "stage", "progress", "status")
    COLUMN_LAYOUT = {
        "episode": (90, "center"), "source_video": (260, "w"),
        "stage": (150, "center"), "progress": (90, "center"), "status": (420, "w"),
    }

    def __init__(
        self, parent: tk.Misc, *, labels: WorkspaceLabels,
        source_variable: tk.StringVar, prompt: str,
        on_select_file: Callable[[], None], on_select_folder: Callable[[], None],
        on_saved_selected: Callable[..., None], on_resume_saved: Callable[[], None],
        on_delete_saved: Callable[[], None],
        on_start: Callable[[], None], on_stop: Callable[[], None],
        on_open_output: Callable[[], None], on_resume: Callable[[], None],
        on_clear: Callable[[], None], on_use_json: Callable[[], None] | None = None,
        on_review: Callable[[], None] | None = None, review_before_render: bool = False,
    ) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        self.source_panel = ttk.LabelFrame(self, text=labels.source_title, padding=12)
        self.source_panel.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.source_panel.columnconfigure(0, weight=1)
        source_buttons = ttk.Frame(self.source_panel)
        source_buttons.pack(fill="x", pady=(0, 6))
        self.select_file_button = ttk.Button(source_buttons, text=labels.select_file, width=20, command=on_select_file)
        self.select_file_button.pack(side="left", padx=(0, 6))
        self.select_folder_button = ttk.Button(source_buttons, text=labels.select_folder, width=20, command=on_select_folder)
        self.select_folder_button.pack(side="left")
        self.source_label = ttk.Label(
            self.source_panel, textvariable=source_variable, font=("Segoe UI", 9),
            foreground="#0969da", wraplength=900,
        )
        self.source_label.pack(fill="x")
        self.saved_panel = ttk.Frame(self.source_panel)
        ttk.Label(self.saved_panel, text=labels.saved_project, width=27).pack(side="left", padx=(0, 8))
        self.saved_combo = ttk.Combobox(self.saved_panel, state="readonly", width=54)
        self.saved_combo.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.saved_combo.bind("<<ComboboxSelected>>", on_saved_selected)
        self.resume_saved_button = ttk.Button(self.saved_panel, text=labels.resume_saved, width=25, command=on_resume_saved)
        self.resume_saved_button.pack(side="left", padx=(0, 6))
        self.delete_saved_button = ttk.Button(
            self.saved_panel, text="🗑 Delete", width=12, command=on_delete_saved,
        )
        self.delete_saved_button.pack(side="left")

        self.prompt_panel = ttk.LabelFrame(self, text=labels.prompt_title, padding=8)
        self.prompt_panel.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.prompt_panel.columnconfigure(0, weight=1)
        prompt_host = ttk.Frame(self.prompt_panel)
        prompt_host.grid(row=0, column=0, sticky="ew")
        prompt_host.columnconfigure(0, weight=1)
        self.prompt_text = tk.Text(prompt_host, height=4, wrap="word", font=("Segoe UI", 9))
        self.prompt_text.insert("1.0", prompt)
        prompt_scroll = ttk.Scrollbar(prompt_host, orient="vertical", command=self.prompt_text.yview)
        self.prompt_text.configure(yscrollcommand=prompt_scroll.set)
        self.prompt_text.grid(row=0, column=0, sticky="ew")
        prompt_scroll.grid(row=0, column=1, sticky="ns")

        self.queue_panel = ttk.LabelFrame(self, text=labels.queue_title, padding=8)
        self.queue_panel.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
        self.queue_panel.columnconfigure(0, weight=1)
        self.queue_panel.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(self.queue_panel, columns=self.COLUMNS, show="headings", selectmode="browse")
        for name, title in zip(self.COLUMNS, ("Episode", "Source Video", "Stage", "Progress", "Status")):
            self.tree.heading(name, text=title)
            width, anchor = self.COLUMN_LAYOUT[name]
            self.tree.column(name, width=width, anchor=anchor)
        queue_scroll = ttk.Scrollbar(self.queue_panel, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=queue_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        queue_scroll.grid(row=0, column=1, sticky="ns")

        self.action_toolbar = ttk.Frame(self)
        self.action_toolbar.grid(row=3, column=0, sticky="ew")
        self.start_button = ttk.Button(self.action_toolbar, text=labels.start, style="Primary.TButton", command=on_start)
        self.start_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(self.action_toolbar, text="⏹ Stop", style="Danger.TButton", command=on_stop, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 8))
        self.open_output_button = ttk.Button(self.action_toolbar, text="📂 Open Output Folder", command=on_open_output)
        self.open_output_button.pack(side="left", padx=(0, 8))
        self.resume_button = ttk.Button(self.action_toolbar, text="⏯ Resume", command=on_resume, state="disabled")
        self.resume_button.pack(side="left", padx=(0, 8))
        self.review_button = None
        if on_review is not None:
            self.review_button = ttk.Button(self.action_toolbar, text="Review Plan", command=on_review)
            self.review_button.pack(side="left", padx=(0, 8))
        self.review_var = tk.BooleanVar(value=review_before_render)
        self.review_toggle = ttk.Checkbutton(
            self.action_toolbar, text="Review before render", variable=self.review_var,
        )
        self.review_toggle.pack(side="left", padx=(0, 8))
        self.use_json_button = None
        if on_use_json is not None:
            self.use_json_button = ttk.Button(self.action_toolbar, text="📄 Use JSON (0 AI)", command=on_use_json)
            self.use_json_button.pack(side="left", padx=(0, 8))
        self.clear_button = ttk.Button(self.action_toolbar, text=labels.clear, command=on_clear)
        self.clear_button.pack(side="right")

    def show_saved_projects(self, labels: list[str], selected: str = "") -> None:
        self.saved_combo.configure(values=labels)
        self.saved_combo.set(selected)
        if labels:
            if not self.saved_panel.winfo_manager():
                self.saved_panel.pack(fill="x", pady=(8, 0))
        else:
            self.saved_panel.pack_forget()
