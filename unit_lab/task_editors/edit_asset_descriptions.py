#!/usr/bin/env python3
"""Browse GenieSim benchmark assets and edit description.py name fields.

This tool is aimed at the asset root layout used under:
    <asset_root>/objects/benchmark/<category>/<asset_id>/

It lets you:
1. Filter and browse assets from the benchmark catalog.
2. Drag a slider or click a row to jump between assets quickly.
3. Review every snapshot image for the selected asset.
4. Inspect the current description.py content.
5. Add or modify english_name and chinese_name, then write them back safely.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk


DEFAULT_ASSET_ROOT = Path(
    "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets"
)
UI_FONT_CANDIDATES = [
    "Cabin",
    "Aptos",
    "Noto Sans",
    "Helvetica",
    "Segoe UI",
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
]
MONO_FONT_CANDIDATES = [
    "JetBrains Mono",
    "Menlo",
    "Consolas",
    "DejaVu Sans Mono",
    "Liberation Mono",
    "Courier New",
    "Courier",
]
STATUS_FILTERS = [
    "All",
    "Missing either name",
    "Missing english_name",
    "Missing chinese_name",
    "Has both names",
    "Parse errors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse assets and edit english_name/chinese_name in description.py."
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=DEFAULT_ASSET_ROOT,
        help="Root directory containing objects/benchmark.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def pick_first_text(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        return str(value[0])
    if value is None:
        return ""
    return str(value)


def list_snapshots(obj_dir: Path) -> tuple[Path, ...]:
    snapshot_dir = obj_dir / "snapshot"
    if not snapshot_dir.exists():
        return ()

    preferred = snapshot_dir / "Camera1.png"
    snapshot_paths = sorted(snapshot_dir.glob("*.png"))
    ordered: list[Path] = []
    if preferred.exists():
        ordered.append(preferred)
    ordered.extend(path for path in snapshot_paths if path != preferred)
    return tuple(ordered)


def load_description_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def parse_description_mapping(path: Path) -> tuple[dict[str, Any], str, str | None]:
    raw_text = load_description_text(path)
    if not raw_text.strip():
        return {}, raw_text, None

    try:
        data = ast.literal_eval(raw_text)
    except Exception as exc_ast:
        try:
            data = json.loads(raw_text)
        except Exception as exc_json:
            return {}, raw_text, f"ast: {exc_ast}; json: {exc_json}"

    if not isinstance(data, dict):
        return {}, raw_text, "description.py did not parse into a dict"

    return data, raw_text, None


def build_fallback_description_dict(
    object_parameters: dict[str, Any],
) -> dict[str, Any]:
    llm = object_parameters.get("llm_descriptions", {})
    fallback: dict[str, Any] = {}

    semantic_name = llm.get("semantic_name") or object_parameters.get("semantic_name")
    if semantic_name not in (None, "", []):
        fallback["semantic_name"] = semantic_name

    for key in [
        "object_category",
        "color",
        "shape",
        "materials",
        "dimensions",
        "unit",
        "descriptive_terms",
        "full_description",
    ]:
        value = llm.get(key)
        if value not in (None, "", []):
            fallback[key] = value

    return fallback


def normalize_name_text(value: str) -> str:
    return " ".join(value.strip().split())


def merge_name_fields(
    source: dict[str, Any],
    english_name: str,
    chinese_name: str,
) -> dict[str, Any]:
    english_name = normalize_name_text(english_name)
    chinese_name = normalize_name_text(chinese_name)

    merged: dict[str, Any] = {}
    inserted = False

    for key, value in source.items():
        if key in {"english_name", "chinese_name"}:
            continue
        merged[key] = value
        if key == "semantic_name":
            if english_name:
                merged["english_name"] = english_name
            if chinese_name:
                merged["chinese_name"] = chinese_name
            inserted = True

    if not inserted:
        if english_name:
            merged["english_name"] = english_name
        if chinese_name:
            merged["chinese_name"] = chinese_name

    return merged


def serialize_description_mapping(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=4, ensure_ascii=False) + "\n"


def write_description_mapping(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_description_mapping(data), encoding="utf-8")


def safe_tree_iid(record_key: str) -> str:
    return record_key.replace("/", "|")


@dataclass(frozen=True)
class AssetRecord:
    key: str
    asset_id: str
    category: str
    data_info_dir: str
    obj_dir: Path
    description_path: Path
    object_parameters_path: Path
    snapshot_paths: tuple[Path, ...]
    semantic_name: str
    english_name: str
    chinese_name: str
    color: str
    object_category: str
    full_description: str
    raw_description_text: str
    description_dict: dict[str, Any]
    fallback_description_dict: dict[str, Any]
    description_error: str | None

    @property
    def status_text(self) -> str:
        if self.description_error:
            return "parse error"
        if self.english_name and self.chinese_name:
            return "ready"
        if self.english_name or self.chinese_name:
            return "partial"
        if self.description_path.exists():
            return "missing names"
        return "no description.py"

    @property
    def missing_summary(self) -> str:
        missing: list[str] = []
        if not self.english_name:
            missing.append("english")
        if not self.chinese_name:
            missing.append("chinese")
        return ", ".join(missing) if missing else "none"

    @property
    def snapshot_count(self) -> int:
        return len(self.snapshot_paths)


def load_asset_catalog(asset_root: Path) -> list[AssetRecord]:
    objects_root = asset_root / "objects" / "benchmark"
    if not objects_root.exists():
        raise FileNotFoundError(f"Missing benchmark objects root: {objects_root}")

    records: list[AssetRecord] = []
    for obj_dir in sorted(objects_root.glob("*/*")):
        if not obj_dir.is_dir():
            continue

        object_parameters_path = obj_dir / "object_parameters.json"
        if not object_parameters_path.exists():
            continue

        object_parameters = load_json(object_parameters_path)
        fallback_description_dict = build_fallback_description_dict(object_parameters)
        description_path = obj_dir / "description.py"
        description_dict, raw_description_text, description_error = parse_description_mapping(
            description_path
        )

        llm = object_parameters.get("llm_descriptions", {})
        semantic_name = (
            pick_first_text(description_dict.get("semantic_name"))
            or pick_first_text(fallback_description_dict.get("semantic_name"))
            or pick_first_text(object_parameters.get("semantic_name"))
            or pick_first_text(llm.get("semantic_name"))
        )
        color = (
            pick_first_text(description_dict.get("color"))
            or pick_first_text(fallback_description_dict.get("color"))
            or pick_first_text(llm.get("color"))
        )
        object_category = (
            pick_first_text(description_dict.get("object_category"))
            or pick_first_text(fallback_description_dict.get("object_category"))
            or pick_first_text(llm.get("object_category"))
        )
        full_description = (
            pick_first_text(description_dict.get("full_description"))
            or pick_first_text(fallback_description_dict.get("full_description"))
            or pick_first_text(llm.get("full_description"))
        )

        category = obj_dir.parent.name
        asset_id = obj_dir.name
        data_info_dir = f"objects/benchmark/{category}/{asset_id}/"
        key = f"{category}/{asset_id}"

        records.append(
            AssetRecord(
                key=key,
                asset_id=asset_id,
                category=category,
                data_info_dir=data_info_dir,
                obj_dir=obj_dir,
                description_path=description_path,
                object_parameters_path=object_parameters_path,
                snapshot_paths=list_snapshots(obj_dir),
                semantic_name=semantic_name,
                english_name=pick_first_text(description_dict.get("english_name")),
                chinese_name=pick_first_text(description_dict.get("chinese_name")),
                color=color,
                object_category=object_category,
                full_description=full_description,
                raw_description_text=raw_description_text,
                description_dict=description_dict,
                fallback_description_dict=fallback_description_dict,
                description_error=description_error,
            )
        )

    return records


class AssetDescriptionEditor:
    def __init__(self, root: tk.Tk, asset_root: Path) -> None:
        self.root = root
        self.asset_root = asset_root

        self.catalog: list[AssetRecord] = []
        self.filtered_records: list[AssetRecord] = []
        self.tree_record_map: dict[str, AssetRecord] = {}
        self.selected_record: AssetRecord | None = None
        self.selected_snapshot_index = 0
        self.current_preview_image: tk.PhotoImage | None = None
        self.snapshot_thumb_images: list[tk.PhotoImage | None] = []
        self._suspend_tree_event = False
        self._suspend_slider_event = False
        self._suspend_field_updates = False
        self._closing = False
        self._tree_render_token = 0
        self._pending_tree_after_id: str | None = None
        self._pending_selection_after_id: str | None = None
        self._pending_strip_after_id: str | None = None
        self._snapshot_render_token = 0

        self.asset_root_var = tk.StringVar(value=str(asset_root))
        self.search_var = tk.StringVar()
        self.category_var = tk.StringVar(value="All")
        self.status_filter_var = tk.StringVar(value="All")
        self.count_var = tk.StringVar(value="0 assets")
        self.selection_var = tk.StringVar(value="No asset selected")
        self.description_path_var = tk.StringVar(value="-")
        self.data_info_dir_var = tk.StringVar(value="-")
        self.summary_var = tk.StringVar(value="-")
        self.snapshot_info_var = tk.StringVar(value="snapshots 0/0")
        self.parse_status_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Loading asset catalog...")
        self.slider_info_var = tk.StringVar(value="0 / 0")
        self.english_name_var = tk.StringVar()
        self.chinese_name_var = tk.StringVar()

        self.tree: ttk.Treeview | None = None
        self.slider: ttk.Scale | None = None
        self.preview_canvas: tk.Canvas | None = None
        self.snapshot_strip_canvas: tk.Canvas | None = None
        self.snapshot_strip_scrollbar: ttk.Scrollbar | None = None
        self.full_description_text: tk.Text | None = None
        self.raw_description_text: tk.Text | None = None
        self.save_button: ttk.Button | None = None
        self.save_next_button: ttk.Button | None = None
        self.revert_button: ttk.Button | None = None
        self.semantic_to_english_button: ttk.Button | None = None
        self.selected_title_label: ttk.Label | None = None

        self.root.title("GenieSim Asset Description Editor")
        self.root.geometry("1700x980")
        self.root.minsize(1360, 840)

        style = ttk.Style(self.root)
        self._configure_fonts(style)
        self._configure_styles(style)
        self._build_ui()
        self._bind_events()
        self.status_var.set("Starting editor...")
        self.root.after(10, lambda: self.reload_catalog(check_unsaved=False))

    def _configure_fonts(self, style: ttk.Style) -> None:
        self.ui_font_family = self._choose_font_family(
            UI_FONT_CANDIDATES,
            "TkDefaultFont",
        )
        self.mono_font_family = self._choose_font_family(
            MONO_FONT_CANDIDATES,
            "TkFixedFont",
        )

        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family=self.ui_font_family, size=11)

        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(family=self.ui_font_family, size=10)

        fixed_font = tkfont.nametofont("TkFixedFont")
        fixed_font.configure(family=self.mono_font_family, size=10)

        heading_font = tkfont.nametofont("TkHeadingFont")
        heading_font.configure(family=self.ui_font_family, size=11, weight="bold")

        caption_font = tkfont.nametofont("TkCaptionFont")
        caption_font.configure(family=self.ui_font_family, size=10)

        style.configure(".", font=(self.ui_font_family, 11))
        style.configure("Treeview", font=(self.ui_font_family, 10), rowheight=24)
        style.configure("Treeview.Heading", font=(self.ui_font_family, 11, "bold"))
        style.configure("TLabelframe.Label", font=(self.ui_font_family, 11, "bold"))

    def _configure_styles(self, style: ttk.Style) -> None:
        available_themes = set(style.theme_names())
        if "clam" in available_themes:
            style.theme_use("clam")

        bg = "#f6f1e8"
        panel = "#fbf7f0"
        ink = "#1f2937"
        line = "#d6cfc2"
        accent = "#0f766e"
        accent_dark = "#115e59"
        warn = "#b45309"

        self.root.configure(bg=bg)
        style.configure(".", background=bg, foreground=ink)
        style.configure("TFrame", background=bg)
        style.configure("Panel.TFrame", background=panel)
        style.configure("TLabelframe", background=bg, bordercolor=line, relief="solid")
        style.configure("TLabelframe.Label", background=bg, foreground=ink)
        style.configure("TLabel", background=bg, foreground=ink)
        style.configure("Muted.TLabel", background=bg, foreground="#6b7280")
        style.configure("Value.TLabel", background=bg, foreground="#111827")
        style.configure("Title.TLabel", background=bg, foreground="#111827", font=(self.ui_font_family, 16, "bold"))
        style.configure("Path.TLabel", background=bg, foreground="#374151")
        style.configure("TEntry", fieldbackground="#fffdf8")
        style.configure("TCombobox", fieldbackground="#fffdf8")
        style.configure("TButton", padding=(10, 6))
        style.configure("Accent.TButton", background=accent, foreground="#ffffff")
        style.map(
            "Accent.TButton",
            background=[("active", accent_dark), ("pressed", accent_dark)],
            foreground=[("active", "#ffffff"), ("pressed", "#ffffff")],
        )
        style.configure("Warn.TButton", background=warn, foreground="#ffffff")
        style.map(
            "Warn.TButton",
            background=[("active", "#92400e"), ("pressed", "#92400e")],
            foreground=[("active", "#ffffff"), ("pressed", "#ffffff")],
        )
        style.map(
            "Treeview",
            background=[("selected", "#d97706")],
            foreground=[("selected", "#ffffff")],
        )

    def _choose_font_family(
        self,
        candidates: list[str],
        fallback_named_font: str,
    ) -> str:
        available = {
            family.lower(): family
            for family in self.root.tk.call("font", "families")
        }
        for candidate in candidates:
            family = available.get(candidate.lower())
            if family:
                return family
        return str(tkfont.nametofont(fallback_named_font).actual("family"))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        toolbar = ttk.LabelFrame(outer, text="Catalog")
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(1, weight=1)

        ttk.Label(toolbar, text="Asset Root").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=8)
        ttk.Entry(toolbar, textvariable=self.asset_root_var).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(0, 8),
            pady=8,
        )
        ttk.Button(toolbar, text="Browse", command=self.browse_asset_root).grid(
            row=0,
            column=2,
            padx=(0, 8),
            pady=8,
        )
        ttk.Button(toolbar, text="Reload", command=lambda: self.reload_catalog()).grid(
            row=0,
            column=3,
            pady=8,
        )

        content = ttk.Frame(outer)
        content.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=5)
        content.columnconfigure(1, weight=8)

        self._build_library_panel(content)
        self._build_detail_panel(content)

        ttk.Label(
            outer,
            textvariable=self.status_var,
            anchor="w",
            relief=tk.SUNKEN,
            padding=6,
        ).grid(row=2, column=0, sticky="ew", pady=(10, 0))

    def _build_library_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Asset Library")
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)

        controls = ttk.Frame(panel)
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)
        controls.columnconfigure(5, weight=1)

        ttk.Label(controls, text="Search").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(controls, textvariable=self.search_var).grid(row=0, column=1, sticky="ew", padx=(0, 10))

        ttk.Label(controls, text="Category").grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.category_combo = ttk.Combobox(
            controls,
            textvariable=self.category_var,
            values=["All"],
            state="readonly",
        )
        self.category_combo.grid(row=0, column=3, sticky="ew", padx=(0, 10))

        ttk.Label(controls, text="Status").grid(row=0, column=4, sticky="w", padx=(0, 6))
        self.status_combo = ttk.Combobox(
            controls,
            textvariable=self.status_filter_var,
            values=STATUS_FILTERS,
            state="readonly",
        )
        self.status_combo.grid(row=0, column=5, sticky="ew")

        meta = ttk.Frame(panel)
        meta.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 0))
        meta.columnconfigure(0, weight=1)
        meta.columnconfigure(1, weight=1)
        ttk.Label(meta, textvariable=self.count_var, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(meta, textvariable=self.slider_info_var, style="Muted.TLabel").grid(row=0, column=1, sticky="e")

        tree_frame = ttk.Frame(panel)
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 6))
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        columns = (
            "idx",
            "category",
            "asset_id",
            "semantic_name",
            "english_name",
            "chinese_name",
            "status",
        )
        self.tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
        )
        headings = {
            "idx": "#",
            "category": "Category",
            "asset_id": "Asset ID",
            "semantic_name": "Semantic",
            "english_name": "English",
            "chinese_name": "Chinese",
            "status": "Status",
        }
        widths = {
            "idx": 44,
            "category": 110,
            "asset_id": 210,
            "semantic_name": 150,
            "english_name": 130,
            "chinese_name": 130,
            "status": 120,
        }
        for key in columns:
            self.tree.heading(key, text=headings[key])
            anchor = tk.CENTER if key in {"idx", "status"} else tk.W
            self.tree.column(key, width=widths[key], stretch=key not in {"idx"}, anchor=anchor)

        self.tree.tag_configure("ready", background="#edf7ed")
        self.tree.tag_configure("partial", background="#fff7db")
        self.tree.tag_configure("missing", background="#fff1e7")
        self.tree.tag_configure("error", background="#fdecec")

        tree_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_y.set, xscrollcommand=tree_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_y.grid(row=0, column=1, sticky="ns")
        tree_x.grid(row=1, column=0, sticky="ew")

        slider_frame = ttk.LabelFrame(panel, text="Asset Slider")
        slider_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=(2, 8))
        slider_frame.columnconfigure(1, weight=1)

        ttk.Button(slider_frame, text="Prev", command=lambda: self.select_relative_asset(-1)).grid(
            row=0,
            column=0,
            padx=(0, 8),
            pady=8,
        )
        self.slider = ttk.Scale(
            slider_frame,
            from_=1,
            to=1,
            orient=tk.HORIZONTAL,
            command=self.on_slider_moved,
        )
        self.slider.grid(row=0, column=1, sticky="ew", pady=8)
        ttk.Button(slider_frame, text="Next", command=lambda: self.select_relative_asset(1)).grid(
            row=0,
            column=2,
            padx=(8, 8),
            pady=8,
        )
        ttk.Button(
            slider_frame,
            text="Next Missing",
            style="Warn.TButton",
            command=self.select_next_missing,
        ).grid(row=0, column=3, pady=8)

    def _build_detail_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.rowconfigure(1, weight=3)
        panel.rowconfigure(2, weight=2)
        panel.columnconfigure(0, weight=1)

        header = ttk.LabelFrame(panel, text="Selected Asset")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        header.columnconfigure(3, weight=1)

        self.selected_title_label = ttk.Label(header, text="No asset selected", style="Title.TLabel")
        self.selected_title_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(10, 6))

        ttk.Label(header, text="Selection").grid(row=1, column=0, sticky="nw", padx=(10, 8), pady=2)
        ttk.Label(header, textvariable=self.selection_var, style="Value.TLabel", wraplength=600).grid(
            row=1,
            column=1,
            sticky="nw",
            pady=2,
        )
        ttk.Label(header, text="Status").grid(row=1, column=2, sticky="nw", padx=(20, 8), pady=2)
        ttk.Label(header, textvariable=self.summary_var, style="Value.TLabel", wraplength=500).grid(
            row=1,
            column=3,
            sticky="nw",
            pady=2,
        )

        ttk.Label(header, text="Data Info Dir").grid(row=2, column=0, sticky="nw", padx=(10, 8), pady=2)
        ttk.Label(header, textvariable=self.data_info_dir_var, style="Path.TLabel", wraplength=600).grid(
            row=2,
            column=1,
            sticky="nw",
            pady=2,
        )
        ttk.Label(header, text="Description Path").grid(row=2, column=2, sticky="nw", padx=(20, 8), pady=2)
        ttk.Label(header, textvariable=self.description_path_var, style="Path.TLabel", wraplength=500).grid(
            row=2,
            column=3,
            sticky="nw",
            pady=2,
        )

        ttk.Label(header, text="Parse").grid(row=3, column=0, sticky="nw", padx=(10, 8), pady=(2, 10))
        ttk.Label(header, textvariable=self.parse_status_var, style="Value.TLabel", wraplength=1000).grid(
            row=3,
            column=1,
            columnspan=3,
            sticky="nw",
            pady=(2, 10),
        )

        preview_frame = ttk.LabelFrame(panel, text="Snapshots")
        preview_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 12))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(
            preview_frame,
            background="#fcfbf7",
            highlightthickness=1,
            highlightbackground="#d6cfc2",
            bd=0,
        )
        self.preview_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 6))

        strip_meta = ttk.Frame(preview_frame)
        strip_meta.grid(row=1, column=0, sticky="ew", padx=10)
        strip_meta.columnconfigure(0, weight=1)
        ttk.Label(strip_meta, textvariable=self.snapshot_info_var, style="Muted.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 6),
        )

        strip_frame = ttk.Frame(preview_frame)
        strip_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        strip_frame.columnconfigure(0, weight=1)

        self.snapshot_strip_canvas = tk.Canvas(
            strip_frame,
            height=158,
            background="#f3eee5",
            highlightthickness=1,
            highlightbackground="#d6cfc2",
            bd=0,
        )
        self.snapshot_strip_canvas.grid(row=0, column=0, sticky="ew")
        self.snapshot_strip_scrollbar = ttk.Scrollbar(
            strip_frame,
            orient=tk.HORIZONTAL,
            command=self.snapshot_strip_canvas.xview,
        )
        self.snapshot_strip_scrollbar.grid(row=1, column=0, sticky="ew")
        self.snapshot_strip_canvas.configure(xscrollcommand=self.snapshot_strip_scrollbar.set)

        notebook = ttk.Notebook(panel)
        notebook.grid(row=2, column=0, sticky="nsew")

        fields_tab = ttk.Frame(notebook)
        raw_tab = ttk.Frame(notebook)
        fields_tab.columnconfigure(0, weight=1)
        raw_tab.columnconfigure(0, weight=1)
        raw_tab.rowconfigure(0, weight=1)
        notebook.add(fields_tab, text="Fields")
        notebook.add(raw_tab, text="description.py")

        edit_frame = ttk.LabelFrame(fields_tab, text="Name Editor")
        edit_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 10))
        edit_frame.columnconfigure(1, weight=1)
        edit_frame.columnconfigure(3, weight=1)

        ttk.Label(edit_frame, text="english_name").grid(row=0, column=0, sticky="w", padx=(10, 8), pady=(10, 6))
        ttk.Entry(edit_frame, textvariable=self.english_name_var).grid(
            row=0,
            column=1,
            sticky="ew",
            pady=(10, 6),
        )
        ttk.Label(edit_frame, text="chinese_name").grid(row=0, column=2, sticky="w", padx=(16, 8), pady=(10, 6))
        ttk.Entry(edit_frame, textvariable=self.chinese_name_var).grid(
            row=0,
            column=3,
            sticky="ew",
            pady=(10, 6),
            padx=(0, 10),
        )

        button_bar = ttk.Frame(edit_frame)
        button_bar.grid(row=1, column=0, columnspan=4, sticky="ew", padx=10, pady=(0, 10))
        button_bar.columnconfigure(5, weight=1)

        self.save_button = ttk.Button(button_bar, text="Save Current", style="Accent.TButton", command=self.save_current)
        self.save_button.grid(row=0, column=0, padx=(0, 8))

        self.save_next_button = ttk.Button(
            button_bar,
            text="Save + Next Missing",
            style="Warn.TButton",
            command=self.save_current_and_next_missing,
        )
        self.save_next_button.grid(row=0, column=1, padx=(0, 8))

        self.revert_button = ttk.Button(button_bar, text="Revert Fields", command=self.revert_fields)
        self.revert_button.grid(row=0, column=2, padx=(0, 8))

        self.semantic_to_english_button = ttk.Button(
            button_bar,
            text="Use semantic_name",
            command=self.use_semantic_as_english,
        )
        self.semantic_to_english_button.grid(row=0, column=3, padx=(0, 8))

        ttk.Button(button_bar, text="Reload Catalog", command=lambda: self.reload_catalog()).grid(
            row=0,
            column=4,
        )

        description_frame = ttk.LabelFrame(fields_tab, text="Current Description Summary")
        description_frame.grid(row=1, column=0, sticky="nsew")
        description_frame.columnconfigure(0, weight=1)
        description_frame.rowconfigure(1, weight=1)

        summary_meta = ttk.Frame(description_frame)
        summary_meta.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        summary_meta.columnconfigure(1, weight=1)
        summary_meta.columnconfigure(3, weight=1)

        ttk.Label(summary_meta, text="semantic_name").grid(row=0, column=0, sticky="nw", padx=(0, 8))
        self.semantic_value_label = ttk.Label(summary_meta, text="-", style="Value.TLabel", wraplength=360)
        self.semantic_value_label.grid(row=0, column=1, sticky="nw")

        ttk.Label(summary_meta, text="object_category").grid(row=0, column=2, sticky="nw", padx=(18, 8))
        self.object_category_value_label = ttk.Label(summary_meta, text="-", style="Value.TLabel", wraplength=360)
        self.object_category_value_label.grid(row=0, column=3, sticky="nw")

        ttk.Label(summary_meta, text="color").grid(row=1, column=0, sticky="nw", padx=(0, 8), pady=(8, 0))
        self.color_value_label = ttk.Label(summary_meta, text="-", style="Value.TLabel", wraplength=360)
        self.color_value_label.grid(row=1, column=1, sticky="nw", pady=(8, 0))

        ttk.Label(summary_meta, text="snapshots").grid(row=1, column=2, sticky="nw", padx=(18, 8), pady=(8, 0))
        self.snapshot_count_value_label = ttk.Label(summary_meta, text="-", style="Value.TLabel")
        self.snapshot_count_value_label.grid(row=1, column=3, sticky="nw", pady=(8, 0))

        self.full_description_text = tk.Text(
            description_frame,
            wrap=tk.WORD,
            height=10,
            state=tk.DISABLED,
            background="#fffdf8",
            relief=tk.SOLID,
            bd=1,
        )
        self.full_description_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        self.raw_description_text = tk.Text(
            raw_tab,
            wrap=tk.NONE,
            height=20,
            state=tk.DISABLED,
            font=(self.mono_font_family, 10),
            background="#fbfaf7",
            relief=tk.SOLID,
            bd=1,
        )
        self.raw_description_text.grid(row=0, column=0, sticky="nsew")
        raw_y = ttk.Scrollbar(raw_tab, orient=tk.VERTICAL, command=self.raw_description_text.yview)
        raw_x = ttk.Scrollbar(raw_tab, orient=tk.HORIZONTAL, command=self.raw_description_text.xview)
        self.raw_description_text.configure(yscrollcommand=raw_y.set, xscrollcommand=raw_x.set)
        raw_y.grid(row=0, column=1, sticky="ns")
        raw_x.grid(row=1, column=0, sticky="ew")

    def _bind_events(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Control-s>", lambda event: self.save_current())
        self.root.bind("<Control-r>", lambda event: self.reload_catalog())
        self.search_var.trace_add("write", lambda *_: self.refresh_library())
        self.category_var.trace_add("write", lambda *_: self.refresh_library())
        self.status_filter_var.trace_add("write", lambda *_: self.refresh_library())
        self.english_name_var.trace_add("write", lambda *_: self.update_dirty_state())
        self.chinese_name_var.trace_add("write", lambda *_: self.update_dirty_state())

        if self.tree is not None:
            self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        if self.preview_canvas is not None:
            self.preview_canvas.bind("<Configure>", lambda event: self.render_preview())
        if self.snapshot_strip_canvas is not None:
            self.snapshot_strip_canvas.bind("<Button-1>", self.on_snapshot_strip_click)

    def browse_asset_root(self) -> None:
        if not self.confirm_discard_changes():
            return

        selected = filedialog.askdirectory(
            parent=self.root,
            title="Select GenieSim asset root",
            initialdir=self.asset_root_var.get() or str(DEFAULT_ASSET_ROOT),
        )
        if not selected:
            return

        self.asset_root = Path(selected)
        self.asset_root_var.set(str(self.asset_root))
        self.reload_catalog(check_unsaved=False)

    def reload_catalog(self, check_unsaved: bool = True) -> None:
        if check_unsaved and not self.confirm_discard_changes():
            return
        preserve_key = self.selected_record.key if self.selected_record else None
        asset_root = Path(self.asset_root_var.get()).expanduser()
        self.asset_root = asset_root
        self.status_var.set(f"Loading asset catalog from {asset_root} ...")
        self.count_var.set("Loading assets...")
        self.cancel_pending_tree_render()
        self.root.update_idletasks()
        try:
            self.catalog = load_asset_catalog(asset_root)
        except Exception as exc:
            messagebox.showerror("Catalog Load Failed", str(exc))
            self.status_var.set(f"Catalog load failed: {exc}")
            self.count_var.set("Load failed")
            return

        categories = ["All"] + sorted({record.category for record in self.catalog})
        self.category_combo.configure(values=categories)
        if self.category_var.get() not in categories:
            self.category_var.set("All")

        self.status_var.set(f"Loaded {len(self.catalog)} assets from {self.asset_root}")
        self.refresh_library(preserve_key=preserve_key)

    def refresh_library(self, preserve_key: str | None = None) -> None:
        if preserve_key is None and self.selected_record is not None:
            preserve_key = self.selected_record.key
        search = self.search_var.get().strip().lower()
        category = self.category_var.get()
        status_filter = self.status_filter_var.get()

        self.filtered_records = [
            record
            for record in self.catalog
            if self.record_matches_filters(record, search, category, status_filter)
        ]
        self.count_var.set(f"{len(self.filtered_records)} filtered / {len(self.catalog)} total")

        if self.tree is None:
            return

        self.cancel_pending_tree_render()
        self._tree_render_token += 1
        render_token = self._tree_render_token

        self._suspend_tree_event = True
        self.tree.delete(*self.tree.get_children())
        self.tree_record_map.clear()
        self._suspend_tree_event = False

        self.update_slider_bounds()

        if not self.filtered_records:
            self.clear_selection()
            return

        chunk_size = 60
        target_record = None
        if preserve_key is not None:
            target_record = next(
                (record for record in self.filtered_records if record.key == preserve_key),
                None,
            )
        if target_record is None and self.selected_record is None:
            target_record = self.filtered_records[0]
        selection_scheduled = False
        selection_needs_apply = (
            target_record is not None
            and (self.selected_record is None or self.selected_record.key != target_record.key)
        )
        insert_errors: list[str] = []

        def render_chunk(start_index: int) -> None:
            if self._closing or render_token != self._tree_render_token or self.tree is None:
                return

            end_index = min(start_index + chunk_size, len(self.filtered_records))
            self._suspend_tree_event = True
            for index, record in enumerate(self.filtered_records[start_index:end_index], start=start_index + 1):
                item_id = safe_tree_iid(record.key)
                if record.description_error:
                    tag = "error"
                elif record.english_name and record.chinese_name:
                    tag = "ready"
                elif record.english_name or record.chinese_name:
                    tag = "partial"
                else:
                    tag = "missing"

                try:
                    self.tree.insert(
                        "",
                        tk.END,
                        iid=item_id,
                        values=(
                            index,
                            record.category,
                            record.asset_id,
                            record.semantic_name,
                            record.english_name,
                            record.chinese_name,
                            record.status_text,
                        ),
                        tags=(tag,),
                    )
                except tk.TclError as exc:
                    fallback_item_id = f"{item_id}|{index}"
                    try:
                        self.tree.insert(
                            "",
                            tk.END,
                            iid=fallback_item_id,
                            values=(
                                index,
                                record.category,
                                record.asset_id,
                                record.semantic_name,
                                record.english_name,
                                record.chinese_name,
                                record.status_text,
                            ),
                            tags=(tag,),
                        )
                        item_id = fallback_item_id
                        insert_errors.append(f"{record.key}: {exc}")
                    except tk.TclError:
                        insert_errors.append(f"{record.key}: {exc}")
                        continue
                self.tree_record_map[item_id] = record
            self._suspend_tree_event = False

            self.status_var.set(
                f"Rendering asset list {end_index}/{len(self.filtered_records)} ..."
                if end_index < len(self.filtered_records)
                else f"Rendered {len(self.filtered_records)} filtered assets"
            )

            nonlocal selection_scheduled
            if not selection_scheduled and selection_needs_apply and target_record is not None:
                self.schedule_selection(target_record, force=True)
                selection_scheduled = True

            if end_index < len(self.filtered_records):
                self._pending_tree_after_id = self.root.after_idle(lambda: render_chunk(end_index))
                return

            self._pending_tree_after_id = None
            if target_record is not None and self.selected_record is not None and self.selected_record.key == target_record.key:
                self.restore_tree_selection()
                self.sync_slider_to_selection()
            elif selection_needs_apply and target_record is not None and not selection_scheduled:
                self.schedule_selection(target_record, force=True)
            status = f"Rendered {len(self.filtered_records)} filtered assets"
            if insert_errors:
                status = (
                    f"{status}; {len(insert_errors)} rows needed fallback ids"
                    if len(insert_errors) <= 3
                    else f"{status}; {len(insert_errors)} rows needed fallback ids"
                )
            self.status_var.set(status)

        render_chunk(0)

    def cancel_pending_tree_render(self) -> None:
        if self._pending_tree_after_id is None:
            return
        try:
            self.root.after_cancel(self._pending_tree_after_id)
        except tk.TclError:
            pass
        self._pending_tree_after_id = None

    def schedule_selection(self, record: AssetRecord, force: bool = False) -> None:
        self.cancel_pending_selection()
        try:
            self._pending_selection_after_id = self.root.after_idle(
                lambda: self._run_scheduled_selection(record, force)
            )
        except tk.TclError:
            self._pending_selection_after_id = None

    def _run_scheduled_selection(self, record: AssetRecord, force: bool) -> None:
        self._pending_selection_after_id = None
        if self._closing:
            return
        self.apply_selection(record, force=force)

    def cancel_pending_selection(self) -> None:
        if self._pending_selection_after_id is None:
            return
        try:
            self.root.after_cancel(self._pending_selection_after_id)
        except tk.TclError:
            pass
        self._pending_selection_after_id = None

    def record_matches_filters(
        self,
        record: AssetRecord,
        search: str,
        category: str,
        status_filter: str,
    ) -> bool:
        if category != "All" and record.category != category:
            return False

        if search:
            haystack = " ".join(
                [
                    record.asset_id,
                    record.category,
                    record.data_info_dir,
                    record.semantic_name,
                    record.english_name,
                    record.chinese_name,
                    record.color,
                    record.object_category,
                    record.full_description,
                ]
            ).lower()
            if search not in haystack:
                return False

        if status_filter == "All":
            return True
        if status_filter == "Missing either name":
            return not (record.english_name and record.chinese_name)
        if status_filter == "Missing english_name":
            return not record.english_name
        if status_filter == "Missing chinese_name":
            return not record.chinese_name
        if status_filter == "Has both names":
            return bool(record.english_name and record.chinese_name)
        if status_filter == "Parse errors":
            return record.description_error is not None
        return True

    def on_tree_select(self, _event: tk.Event[Any]) -> None:
        if self._suspend_tree_event or self.tree is None:
            return

        selection = self.tree.selection()
        if not selection:
            return

        item_id = selection[0]
        record = self.tree_record_map.get(item_id)
        if record is None:
            return

        if self.selected_record is not None and record.key == self.selected_record.key:
            self.sync_slider_to_selection()
            return

        if not self.apply_selection(record):
            self.restore_tree_selection()

    def restore_tree_selection(self) -> None:
        if self.selected_record is None or self.tree is None:
            return
        item_id = safe_tree_iid(self.selected_record.key)
        if item_id not in self.tree_record_map:
            return
        current_selection = self.tree.selection()
        if current_selection == (item_id,):
            self.tree.focus(item_id)
            self.tree.see(item_id)
            return
        self._suspend_tree_event = True
        self.tree.selection_set(item_id)
        self.tree.focus(item_id)
        self.tree.see(item_id)
        self._suspend_tree_event = False

    def update_slider_bounds(self) -> None:
        total = len(self.filtered_records)
        self.slider_info_var.set(f"{0 if total == 0 else 1} / {total}" if total else "0 / 0")
        if self.slider is None:
            return
        slider_to = max(total, 1)
        self.slider.configure(from_=1, to=slider_to, state=(tk.NORMAL if total else tk.DISABLED))
        if not total:
            self._suspend_slider_event = True
            self.slider.set(1)
            self._suspend_slider_event = False

    def on_slider_moved(self, value: str) -> None:
        if self._suspend_slider_event:
            return
        if not self.filtered_records:
            return

        index = int(round(float(value))) - 1
        index = max(0, min(index, len(self.filtered_records) - 1))
        target_record = self.filtered_records[index]
        if not self.apply_selection(target_record):
            self.sync_slider_to_selection()

    def select_relative_asset(self, delta: int) -> None:
        if not self.filtered_records:
            return
        if self.selected_record is None:
            self.apply_selection(self.filtered_records[0])
            return
        try:
            current_index = next(
                index
                for index, record in enumerate(self.filtered_records)
                if record.key == self.selected_record.key
            )
        except StopIteration:
            current_index = 0
        target_index = max(0, min(current_index + delta, len(self.filtered_records) - 1))
        self.apply_selection(self.filtered_records[target_index])

    def select_next_missing(self) -> None:
        if not self.filtered_records:
            return

        start_index = -1
        if self.selected_record is not None:
            try:
                start_index = next(
                    index
                    for index, record in enumerate(self.filtered_records)
                    if record.key == self.selected_record.key
                )
            except StopIteration:
                start_index = -1

        total = len(self.filtered_records)
        for offset in range(1, total + 1):
            record = self.filtered_records[(start_index + offset) % total]
            if not (record.english_name and record.chinese_name):
                self.apply_selection(record)
                return
        self.status_var.set("Every filtered asset already has both english_name and chinese_name.")

    def apply_selection(self, record: AssetRecord, force: bool = False) -> bool:
        if not force and self.selected_record is not None and record.key == self.selected_record.key:
            self.sync_slider_to_selection()
            return True

        if not force and not self.confirm_discard_changes():
            return False

        self.selected_record = record
        self.selected_snapshot_index = 0
        self.load_record_into_ui(record)
        self.restore_tree_selection()
        self.sync_slider_to_selection()
        return True

    def clear_selection(self) -> None:
        self.selected_record = None
        self.selected_snapshot_index = 0
        self.selection_var.set("No asset selected")
        self.description_path_var.set("-")
        self.data_info_dir_var.set("-")
        self.summary_var.set("-")
        self.parse_status_var.set("")
        self.snapshot_info_var.set("snapshots 0/0")
        if self.selected_title_label is not None:
            self.selected_title_label.configure(text="No asset selected")
        self.semantic_value_label.configure(text="-")
        self.object_category_value_label.configure(text="-")
        self.color_value_label.configure(text="-")
        self.snapshot_count_value_label.configure(text="-")
        self._suspend_field_updates = True
        self.english_name_var.set("")
        self.chinese_name_var.set("")
        self._suspend_field_updates = False
        self.set_text_widget(self.full_description_text, "")
        self.set_text_widget(self.raw_description_text, "")
        self.current_preview_image = None
        self.snapshot_thumb_images.clear()
        self.cancel_pending_strip_render()
        self._snapshot_render_token += 1
        self.render_preview()
        self.render_snapshot_strip()
        self.update_dirty_state()

    def load_record_into_ui(self, record: AssetRecord) -> None:
        self.selection_var.set(f"{record.category} / {record.asset_id}")
        self.description_path_var.set(str(record.description_path))
        self.data_info_dir_var.set(record.data_info_dir)
        self.summary_var.set(
            f"semantic={record.semantic_name or '-'} | "
            f"missing={record.missing_summary} | "
            f"status={record.status_text}"
        )
        if record.description_error:
            self.parse_status_var.set(
                f"Parse error in description.py. Saving will rewrite it with a clean dict: {record.description_error}"
            )
        elif record.description_path.exists():
            self.parse_status_var.set("description.py parsed successfully")
        else:
            self.parse_status_var.set("description.py does not exist yet. Saving will create it.")

        if self.selected_title_label is not None:
            self.selected_title_label.configure(text=record.asset_id)
        self.semantic_value_label.configure(text=record.semantic_name or "-")
        self.object_category_value_label.configure(text=record.object_category or "-")
        self.color_value_label.configure(text=record.color or "-")
        self.snapshot_count_value_label.configure(text=str(record.snapshot_count))

        self._suspend_field_updates = True
        self.english_name_var.set(record.english_name)
        self.chinese_name_var.set(record.chinese_name)
        self._suspend_field_updates = False

        full_description = record.full_description or "No full_description available."
        self.set_text_widget(self.full_description_text, full_description)

        if record.raw_description_text.strip():
            raw_text = record.raw_description_text
        else:
            raw_text = serialize_description_mapping(record.fallback_description_dict)
        self.set_text_widget(self.raw_description_text, raw_text)

        self.update_dirty_state()
        self.status_var.set(f"Selected {record.asset_id}, preparing previews ...")
        self.cancel_pending_strip_render()
        self._snapshot_render_token += 1
        try:
            self.root.after_idle(self.render_preview)
            self._pending_strip_after_id = self.root.after_idle(self.render_snapshot_strip)
        except tk.TclError:
            self._pending_strip_after_id = None

    def sync_slider_to_selection(self) -> None:
        if self.slider is None or self.selected_record is None or not self.filtered_records:
            return

        try:
            index = next(
                idx
                for idx, record in enumerate(self.filtered_records, start=1)
                if record.key == self.selected_record.key
            )
        except StopIteration:
            return

        self.slider_info_var.set(f"{index} / {len(self.filtered_records)}")
        self._suspend_slider_event = True
        self.slider.set(index)
        self._suspend_slider_event = False

    def render_preview(self) -> None:
        if self.preview_canvas is None:
            return

        canvas = self.preview_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 320)
        height = max(canvas.winfo_height(), 280)

        if self.selected_record is None:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Select an asset",
                font=(self.ui_font_family, 18, "bold"),
                fill="#6b7280",
            )
            return

        if not self.selected_record.snapshot_paths:
            canvas.create_rectangle(30, 30, width - 30, height - 30, outline="#d6cfc2", width=2)
            canvas.create_text(
                width / 2,
                height / 2,
                text=f"No snapshots\n{self.selected_record.asset_id}",
                font=(self.ui_font_family, 15, "bold"),
                fill="#6b7280",
                justify=tk.CENTER,
            )
            self.snapshot_info_var.set("snapshots 0/0")
            return

        self.selected_snapshot_index = min(
            self.selected_snapshot_index,
            len(self.selected_record.snapshot_paths) - 1,
        )
        snapshot_path = self.selected_record.snapshot_paths[self.selected_snapshot_index]
        self.snapshot_info_var.set(
            f"snapshots {self.selected_snapshot_index + 1}/{len(self.selected_record.snapshot_paths)}"
        )
        image = self.load_scaled_photo_image(snapshot_path, width - 40, height - 40)
        self.current_preview_image = image
        if image is None:
            canvas.create_rectangle(30, 30, width - 30, height - 30, outline="#d6cfc2", width=2)
            canvas.create_text(
                width / 2,
                height / 2,
                text=f"Could not load preview\n{snapshot_path.name}",
                font=(self.ui_font_family, 15, "bold"),
                fill="#6b7280",
                justify=tk.CENTER,
            )
            return

        x = max((width - image.width()) // 2, 16)
        y = max((height - image.height()) // 2, 16)
        canvas.create_rectangle(
            x - 10,
            y - 10,
            x + image.width() + 10,
            y + image.height() + 10,
            outline="#d6cfc2",
            width=2,
        )
        canvas.create_image(x, y, image=image, anchor=tk.NW)
        canvas.create_text(
            18,
            16,
            text=snapshot_path.name,
            anchor=tk.NW,
            font=(self.ui_font_family, 10, "bold"),
            fill="#374151",
        )

    def render_snapshot_strip(self) -> None:
        self._pending_strip_after_id = None
        if self.snapshot_strip_canvas is None:
            return

        canvas = self.snapshot_strip_canvas
        canvas.delete("all")
        self.snapshot_thumb_images.clear()
        height = max(canvas.winfo_height(), int(canvas.cget("height")))
        token = self._snapshot_render_token

        if self.selected_record is None or not self.selected_record.snapshot_paths:
            canvas.create_text(
                220,
                height / 2,
                text="No snapshot thumbnails",
                font=(self.ui_font_family, 13, "bold"),
                fill="#6b7280",
            )
            canvas.configure(scrollregion=(0, 0, 480, height))
            return

        thumb_w = 120
        thumb_h = 120
        gap = 12
        total = len(self.selected_record.snapshot_paths)
        total_width = 14 + total * (thumb_w + 16 + gap)
        canvas.configure(scrollregion=(0, 0, total_width, height))

        def render_chunk(start_index: int) -> None:
            if (
                self._closing
                or token != self._snapshot_render_token
                or self.snapshot_strip_canvas is None
                or self.selected_record is None
                or not self.selected_record.snapshot_paths
            ):
                self._pending_strip_after_id = None
                return

            end_index = min(start_index + 2, len(self.selected_record.snapshot_paths))
            for index in range(start_index, end_index):
                snapshot_path = self.selected_record.snapshot_paths[index]
                x = 14 + index * (thumb_w + 16 + gap)
                image = self.load_scaled_photo_image(snapshot_path, thumb_w, thumb_h)
                self.snapshot_thumb_images.append(image)
                is_selected = index == self.selected_snapshot_index
                outline = "#d97706" if is_selected else "#b8b0a4"
                width = 4 if is_selected else 2
                tile_tag = f"snapshot_{index}"
                tags = (tile_tag, f"snapshot_idx_{index}")

                canvas.create_rectangle(
                    x,
                    12,
                    x + thumb_w + 16,
                    12 + thumb_h + 22,
                    outline=outline,
                    width=width,
                    tags=tags,
                )
                if image is None:
                    canvas.create_rectangle(
                        x + 8,
                        26,
                        x + thumb_w + 8,
                        26 + thumb_h,
                        fill="#f7f3ec",
                        outline="#ddd6ca",
                        tags=tags,
                    )
                    canvas.create_text(
                        x + (thumb_w + 16) / 2,
                        26 + thumb_h / 2,
                        text="No image",
                        font=(self.ui_font_family, 11, "bold"),
                        fill="#6b7280",
                        tags=tags,
                    )
                else:
                    image_x = x + 8 + max((thumb_w - image.width()) // 2, 0)
                    image_y = 26 + max((thumb_h - image.height()) // 2, 0)
                    canvas.create_image(image_x, image_y, image=image, anchor=tk.NW, tags=tags)

                canvas.create_text(
                    x + (thumb_w + 16) / 2,
                    20,
                    text=snapshot_path.stem,
                    font=(self.ui_font_family, 9, "bold"),
                    fill=outline,
                    tags=tags,
                )

            if end_index < len(self.selected_record.snapshot_paths):
                self._pending_strip_after_id = self.root.after_idle(
                    lambda: render_chunk(end_index)
                )
                return

            self._pending_strip_after_id = None
            self.status_var.set(
                f"Selected {self.selected_record.asset_id}, loaded {len(self.selected_record.snapshot_paths)} snapshots"
            )

        render_chunk(0)

    def cancel_pending_strip_render(self) -> None:
        if self._pending_strip_after_id is None:
            return
        try:
            self.root.after_cancel(self._pending_strip_after_id)
        except tk.TclError:
            pass
        self._pending_strip_after_id = None

    def on_snapshot_strip_click(self, event: tk.Event[Any]) -> None:
        if self.snapshot_strip_canvas is None or self.selected_record is None:
            return

        tags = self.snapshot_strip_canvas.gettags(f"current")
        for tag in tags:
            if tag.startswith("snapshot_idx_"):
                try:
                    index = int(tag.split("_")[-1])
                except ValueError:
                    return
                self.selected_snapshot_index = index
                self.render_preview()
                self.render_snapshot_strip()
                return

    def load_scaled_photo_image(
        self,
        image_path: Path | None,
        max_width: int,
        max_height: int,
    ) -> tk.PhotoImage | None:
        if image_path is None or not image_path.exists():
            return None

        try:
            image = tk.PhotoImage(file=str(image_path))
        except tk.TclError:
            return None

        width_ratio = max_width / max(image.width(), 1)
        height_ratio = max_height / max(image.height(), 1)
        scale_ratio = max(min(width_ratio, height_ratio), 0.1)
        numerator = max(1, int(round(scale_ratio * 10)))
        image = image.zoom(numerator, numerator).subsample(10, 10)
        return image

    def set_text_widget(self, widget: tk.Text | None, content: str) -> None:
        if widget is None:
            return
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", content)
        widget.configure(state=tk.DISABLED)

    def confirm_discard_changes(self) -> bool:
        if not self.has_unsaved_changes():
            return True
        if self.selected_record is None:
            return True
        return messagebox.askyesno(
            "Discard changes?",
            (
                f"You have unsaved edits for {self.selected_record.asset_id}.\n\n"
                "Discard those edits and continue?"
            ),
            parent=self.root,
        )

    def has_unsaved_changes(self) -> bool:
        if self.selected_record is None or self._suspend_field_updates:
            return False
        return (
            normalize_name_text(self.english_name_var.get()) != self.selected_record.english_name
            or normalize_name_text(self.chinese_name_var.get()) != self.selected_record.chinese_name
        )

    def update_dirty_state(self) -> None:
        dirty = self.has_unsaved_changes()
        has_record = self.selected_record is not None
        save_state = tk.NORMAL if has_record and dirty else tk.DISABLED
        revert_state = tk.NORMAL if has_record and dirty else tk.DISABLED
        helper_state = tk.NORMAL if has_record else tk.DISABLED

        if self.save_button is not None:
            self.save_button.configure(state=save_state)
        if self.save_next_button is not None:
            self.save_next_button.configure(state=save_state)
        if self.revert_button is not None:
            self.revert_button.configure(state=revert_state)
        if self.semantic_to_english_button is not None:
            self.semantic_to_english_button.configure(state=helper_state)

        title_suffix = " *" if dirty else ""
        self.root.title(f"GenieSim Asset Description Editor{title_suffix}")

    def build_save_base_dict(self, record: AssetRecord) -> dict[str, Any]:
        if record.description_dict:
            return dict(record.description_dict)
        return dict(record.fallback_description_dict)

    def save_current(self) -> bool:
        if self.selected_record is None:
            return False

        record = self.selected_record
        base_dict = self.build_save_base_dict(record)
        merged = merge_name_fields(
            base_dict,
            self.english_name_var.get(),
            self.chinese_name_var.get(),
        )

        try:
            write_description_mapping(record.description_path, merged)
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc), parent=self.root)
            self.status_var.set(f"Save failed for {record.asset_id}: {exc}")
            return False

        serialized = serialize_description_mapping(merged)
        updated_record = replace(
            record,
            english_name=normalize_name_text(self.english_name_var.get()),
            chinese_name=normalize_name_text(self.chinese_name_var.get()),
            description_dict=merged,
            raw_description_text=serialized,
            description_error=None,
        )
        self.replace_record(updated_record)
        self.status_var.set(f"Saved {updated_record.description_path}")
        self.refresh_library(preserve_key=updated_record.key)
        return True

    def save_current_and_next_missing(self) -> None:
        if not self.save_current():
            return
        self.select_next_missing()

    def replace_record(self, updated_record: AssetRecord) -> None:
        self.catalog = [
            updated_record if record.key == updated_record.key else record
            for record in self.catalog
        ]
        self.filtered_records = [
            updated_record if record.key == updated_record.key else record
            for record in self.filtered_records
        ]
        self.selected_record = updated_record

    def revert_fields(self) -> None:
        if self.selected_record is None:
            return
        self._suspend_field_updates = True
        self.english_name_var.set(self.selected_record.english_name)
        self.chinese_name_var.set(self.selected_record.chinese_name)
        self._suspend_field_updates = False
        self.update_dirty_state()
        self.status_var.set(f"Reverted unsaved edits for {self.selected_record.asset_id}")

    def use_semantic_as_english(self) -> None:
        if self.selected_record is None:
            return
        if not self.selected_record.semantic_name:
            self.status_var.set("No semantic_name available for this asset.")
            return
        self.english_name_var.set(self.selected_record.semantic_name)
        self.status_var.set(f"Filled english_name from semantic_name for {self.selected_record.asset_id}")

    def on_close(self) -> None:
        if not self.confirm_discard_changes():
            return
        self._closing = True
        self.cancel_pending_tree_render()
        self.cancel_pending_selection()
        self.cancel_pending_strip_render()
        self.current_preview_image = None
        self.snapshot_thumb_images.clear()
        self.root.destroy()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    editor = AssetDescriptionEditor(root, args.asset_root)
    root.mainloop()


if __name__ == "__main__":
    main()
