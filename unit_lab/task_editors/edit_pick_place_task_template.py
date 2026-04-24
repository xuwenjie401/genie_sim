#!/usr/bin/env python3
"""Browse benchmark assets and generate a pick-place task template.

This editor is built around the existing GenieSim task JSON format. It clones a
reference task template, lets you replace the candidate pick/place assets, and
writes a new task file without touching robot, background, or workspace fields.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk


DEFAULT_ASSET_ROOT = Path(
    "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets"
)
DEFAULT_TEMPLATE_PATH = Path(
    "/home/agxi/RealityLab/genie_sim/source/data_collection/tasks/geniesim_2025/"
    "place_object_into_box_of_specific_color/galbot/"
    "place_object_into_box_of_specific_color_blue_galbot.json"
)

ROLE_ORDER = ["pick", "scene", "target_place", "wrong_place"]
ROLE_LABELS = {
    "pick": "Pick Candidate Pool",
    "scene": "Scene Object Pool",
    "target_place": "Target Place Pool",
    "wrong_place": "Wrong Place Pool",
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edit a GenieSim pick-place task template with a local UI."
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE_PATH,
        help="Reference task template to clone.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default output path shown in the UI.",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=DEFAULT_ASSET_ROOT,
        help="Root directory containing objects/benchmark and interaction.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def normalize_relative_dir(path: str) -> str:
    cleaned = path.strip().replace("\\", "/").lstrip("./").lstrip("/")
    if cleaned and not cleaned.endswith("/"):
        cleaned += "/"
    return cleaned


def sanitize_token(value: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def pick_first_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if value is None:
        return ""
    return str(value)


@dataclass(frozen=True)
class InteractionSummary:
    has_file: bool
    has_grasp: bool
    has_place: bool
    raw_path: Path | None = None
    grasp_labels: tuple[str, ...] = ()
    place_labels: tuple[str, ...] = ()
    detail_lines: tuple[str, ...] = ()

    @property
    def has_any(self) -> bool:
        return self.has_grasp or self.has_place

    def status_text(self) -> str:
        return (
            f"grasp={'yes' if self.has_grasp else 'no'} | "
            f"place={'yes' if self.has_place else 'no'}"
        )


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    category: str
    data_info_dir: str
    object_parameters_path: Path
    snapshot_paths: tuple[Path, ...]
    semantic_name: str
    color: str
    description: str
    mass: float
    interaction: InteractionSummary

    @property
    def snapshot_path(self) -> Path | None:
        return self.snapshot_paths[0] if self.snapshot_paths else None

@dataclass
class TemplateSlots:
    pick_index: int
    target_place_index: int
    wrong_place_indices: list[int]


def summarize_interaction(interaction_path: Path) -> InteractionSummary:
    if not interaction_path.exists():
        return InteractionSummary(has_file=False, has_grasp=False, has_place=False)

    try:
        data = load_json(interaction_path)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[warn] Failed to load interaction metadata: {interaction_path} ({exc})")
        return InteractionSummary(
            has_file=True,
            has_grasp=False,
            has_place=False,
            raw_path=interaction_path,
            detail_lines=(f"invalid interaction json: {exc}",),
        )

    interaction = data.get("interaction", {})
    active = interaction.get("active", {})
    passive = interaction.get("passive", {})

    grasp_labels = tuple(
        sorted(
            {
                *active.get("grasp", {}).keys(),
                *passive.get("grasp", {}).keys(),
            }
        )
    )
    place_labels = tuple(
        sorted(
            {
                *active.get("place", {}).keys(),
                *passive.get("place", {}).keys(),
            }
        )
    )

    detail_lines: list[str] = []
    if grasp_labels:
        detail_lines.append("grasp labels: " + ", ".join(grasp_labels))
    else:
        detail_lines.append("grasp labels: none")
    if place_labels:
        detail_lines.append("place labels: " + ", ".join(place_labels))
    else:
        detail_lines.append("place labels: none")

    return InteractionSummary(
        has_file=True,
        has_grasp=bool(grasp_labels),
        has_place=bool(place_labels),
        raw_path=interaction_path,
        grasp_labels=grasp_labels,
        place_labels=place_labels,
        detail_lines=tuple(detail_lines),
    )


def list_snapshots(obj_dir: Path) -> tuple[Path, ...]:
    snapshot_dir = obj_dir / "snapshot"
    if not snapshot_dir.exists():
        return ()

    preferred = snapshot_dir / "Camera1.png"
    snapshots = sorted(snapshot_dir.glob("*.png"))
    ordered: list[Path] = []
    if preferred.exists():
        ordered.append(preferred)
    ordered.extend(path for path in snapshots if path != preferred)
    return tuple(ordered)


def load_asset_catalog(
    asset_root: Path,
) -> tuple[list[AssetRecord], dict[str, AssetRecord], dict[str, AssetRecord]]:
    objects_root = asset_root / "objects" / "benchmark"
    interaction_root = asset_root / "interaction"
    records: list[AssetRecord] = []
    by_asset_id: dict[str, AssetRecord] = {}
    by_data_info_dir: dict[str, AssetRecord] = {}

    for obj_dir in sorted(objects_root.glob("*/*")):
        if not obj_dir.is_dir():
            continue
        object_parameters_path = obj_dir / "object_parameters.json"
        if not object_parameters_path.exists():
            continue

        object_parameters = load_json(object_parameters_path)
        llm = object_parameters.get("llm_descriptions", {})
        asset_id = obj_dir.name
        category = obj_dir.parent.name
        data_info_dir = normalize_relative_dir(
            f"objects/benchmark/{category}/{asset_id}/"
        )
        semantic_name = pick_first_text(
            object_parameters.get("semantic_name")
            or llm.get("semantic_name")
        )
        color = pick_first_text(llm.get("color"))
        description = pick_first_text(llm.get("full_description"))
        mass = float(object_parameters.get("mass") or 0.05)
        interaction = summarize_interaction(
            interaction_root / asset_id / "interaction.json"
        )

        record = AssetRecord(
            asset_id=asset_id,
            category=category,
            data_info_dir=data_info_dir,
            object_parameters_path=object_parameters_path,
            snapshot_paths=list_snapshots(obj_dir),
            semantic_name=semantic_name,
            color=color,
            description=description,
            mass=mass,
            interaction=interaction,
        )
        records.append(record)
        by_asset_id[asset_id] = record
        by_data_info_dir[data_info_dir] = record

    return records, by_asset_id, by_data_info_dir


def asset_object_id(record: AssetRecord) -> str:
    base = record.asset_id
    if base.startswith("benchmark_"):
        base = base[len("benchmark_") :]
    token = sanitize_token(base)
    if record.category == "storage_box" and record.color:
        color_token = sanitize_token(record.color)
        if color_token and not token.endswith(color_token):
            token = f"{token}_{color_token}"
    return f"geniesim_2025_{token}"


def build_candidate_entry(record: AssetRecord) -> dict[str, Any]:
    return {
        "data_info_dir": record.data_info_dir,
        "object_id": asset_object_id(record),
    }


def build_scene_entry(record: AssetRecord, default_mass: float) -> dict[str, Any]:
    return {
        "data_info_dir": record.data_info_dir,
        "object_id": asset_object_id(record),
        "mass": default_mass,
    }


def dedupe_records(records: list[AssetRecord]) -> list[AssetRecord]:
    seen: set[str] = set()
    deduped: list[AssetRecord] = []
    for record in records:
        if record.asset_id in seen:
            continue
        seen.add(record.asset_id)
        deduped.append(record)
    return deduped


def find_stage(template: dict[str, Any], action_name: str) -> dict[str, Any]:
    for stage in template.get("stages", []):
        if stage.get("action") == action_name:
            return stage
    raise ValueError(f"Template does not contain a '{action_name}' stage")


def detect_template_slots(template: dict[str, Any]) -> TemplateSlots:
    task_related_objects = template.get("objects", {}).get("task_related_objects", [])
    if not task_related_objects:
        raise ValueError("Template does not contain objects.task_related_objects")

    pick_stage = find_stage(template, "pick")
    place_stage = find_stage(template, "place")

    pick_object_id = pick_stage.get("passive", {}).get("object_id")
    if not pick_object_id:
        pick_object_id = pick_stage.get("checker", [{}])[0].get("params", {}).get(
            "object_id"
        )

    target_place_object_id = place_stage.get("passive", {}).get("object_id")
    if not target_place_object_id:
        for checker in place_stage.get("checker", []):
            target_place_object_id = checker.get("params", {}).get("target_id")
            if target_place_object_id:
                break

    pick_index = -1
    target_place_index = -1
    wrong_place_indices: list[int] = []

    for index, obj in enumerate(task_related_objects):
        object_id = obj.get("object_id", "")
        if object_id == pick_object_id:
            pick_index = index
        elif object_id == target_place_object_id:
            target_place_index = index
        else:
            wrong_place_indices.append(index)

    if pick_index < 0:
        raise ValueError(
            f"Could not match pick object_id '{pick_object_id}' inside task_related_objects"
        )
    if target_place_index < 0:
        raise ValueError(
            "Could not match place target object inside task_related_objects"
        )

    return TemplateSlots(
        pick_index=pick_index,
        target_place_index=target_place_index,
        wrong_place_indices=wrong_place_indices,
    )


def resolve_records_from_entries(
    entries: list[dict[str, Any]],
    catalog_by_data_info_dir: dict[str, AssetRecord],
) -> tuple[list[AssetRecord], list[str]]:
    records: list[AssetRecord] = []
    missing: list[str] = []
    for entry in entries:
        data_info_dir = normalize_relative_dir(entry.get("data_info_dir", ""))
        if not data_info_dir:
            continue
        record = catalog_by_data_info_dir.get(data_info_dir)
        if record is None:
            missing.append(data_info_dir)
            continue
        records.append(record)
    return dedupe_records(records), missing


def expand_slot_entries(slot: dict[str, Any]) -> list[dict[str, Any]]:
    if slot.get("candidate_objects"):
        return list(slot["candidate_objects"])
    if slot.get("data_info_dir"):
        return [slot]
    return []


def build_default_output_path(template_path: Path) -> Path:
    return template_path.with_name(template_path.stem + "_custom.json")


def build_task_template(
    reference_template: dict[str, Any],
    slots: TemplateSlots,
    role_records: dict[str, list[AssetRecord]],
    task_id: str,
    task_name_cn: str,
    task_name_en: str,
    init_scene_text: str,
    place_action_cn: str,
    place_action_en: str,
    pick_mass: float,
    place_mass: float,
    scene_mass: float,
) -> dict[str, Any]:
    if not role_records["pick"]:
        raise ValueError("Pick Candidate Pool is empty")
    if not role_records["target_place"]:
        raise ValueError("Target Place Pool is empty")

    template = copy.deepcopy(reference_template)
    task_related_objects = template["objects"]["task_related_objects"]

    pick_slot = copy.deepcopy(task_related_objects[slots.pick_index])
    pick_slot["candidate_objects"] = [
        build_candidate_entry(record)
        for record in role_records["pick"]
    ]
    pick_slot.pop("data_info_dir", None)
    pick_slot["mass"] = pick_mass

    target_place_slot = copy.deepcopy(task_related_objects[slots.target_place_index])
    target_place_slot["candidate_objects"] = [
        build_candidate_entry(record)
        for record in role_records["target_place"]
    ]
    target_place_slot.pop("data_info_dir", None)
    target_place_slot["mass"] = place_mass

    wrong_place_slots: list[dict[str, Any]] = []
    for index in slots.wrong_place_indices:
        wrong_slot = copy.deepcopy(task_related_objects[index])
        if role_records["wrong_place"]:
            wrong_slot["candidate_objects"] = [
                build_candidate_entry(record)
                for record in role_records["wrong_place"]
            ]
            wrong_slot.pop("data_info_dir", None)
            wrong_slot["mass"] = place_mass
            wrong_place_slots.append(wrong_slot)

    replacement_slots: dict[int, dict[str, Any] | None] = {
        slots.pick_index: pick_slot,
        slots.target_place_index: target_place_slot,
    }
    for index in slots.wrong_place_indices:
        replacement_slots[index] = None
    if wrong_place_slots:
        for index, wrong_slot in zip(slots.wrong_place_indices, wrong_place_slots):
            replacement_slots[index] = wrong_slot

    rebuilt_task_related_objects: list[dict[str, Any]] = []
    for index, original_slot in enumerate(task_related_objects):
        if index not in replacement_slots:
            rebuilt_task_related_objects.append(copy.deepcopy(original_slot))
            continue
        replacement = replacement_slots[index]
        if replacement is not None:
            rebuilt_task_related_objects.append(replacement)
    template["objects"]["task_related_objects"] = rebuilt_task_related_objects

    scene_pool = dedupe_records(role_records["pick"] + role_records["scene"])
    if scene_pool:
        if not template["objects"].get("scene_objects"):
            template["objects"]["scene_objects"] = [
                {
                    "sample": {"min_num": 1, "max_num": len(scene_pool), "max_repeat": 1},
                    "workspace_id": pick_slot.get("workspace_id", "work_table"),
                    "available_objects": [],
                }
            ]
        scene_group = template["objects"]["scene_objects"][0]
        scene_group["available_objects"] = [
            build_scene_entry(record, scene_mass)
            for record in scene_pool
        ]
        sample = scene_group.setdefault("sample", {})
        sample["min_num"] = min(sample.get("min_num", 1), len(scene_pool))
        sample["max_num"] = min(sample.get("max_num", len(scene_pool)), len(scene_pool))
        sample["max_repeat"] = sample.get("max_repeat", 1)

    template["task"] = task_id
    task_description = template.setdefault("task_description", {})
    task_description["task_name"] = task_name_cn
    task_description["english_task_name"] = task_name_en
    task_description["init_scene_text"] = init_scene_text

    place_stage = find_stage(template, "place")
    place_action = place_stage.setdefault("action_description", {})
    place_action["action_text"] = place_action_cn
    place_action["english_action_text"] = place_action_en

    return template


class TaskTemplateEditor:
    def __init__(
        self,
        root: tk.Tk,
        asset_root: Path,
        template_path: Path,
        output_path: Path,
    ) -> None:
        self.root = root
        self.asset_root = asset_root
        self.catalog, _, self.catalog_by_data_info_dir = (
            load_asset_catalog(asset_root)
        )
        self.reference_template: dict[str, Any] | None = None
        self.template_slots: TemplateSlots | None = None
        self.filtered_records: list[AssetRecord] = list(self.catalog)
        self.role_records: dict[str, list[AssetRecord]] = {
            role: [] for role in ROLE_ORDER
        }
        self.role_listboxes: dict[str, tk.Listbox] = {}
        self.tree_record_map: dict[str, AssetRecord] = {}
        self.current_preview_image: tk.PhotoImage | None = None
        self.ui_font_family = ""
        self.mono_font_family = ""
        self.selected_canvas: tk.Canvas | None = None
        self.pick_pool_canvas: tk.Canvas | None = None
        self.place_pool_canvas: tk.Canvas | None = None
        self.scene_pool_canvas: tk.Canvas | None = None
        self.selected_info_var = tk.StringVar()
        self.selected_snapshot_info_var = tk.StringVar(value="snapshot 0/0")
        self.selected_zoom_info_var = tk.StringVar(value="100%")
        self.selected_image_scale_var = tk.DoubleVar(value=1.0)
        self.current_selected_record: AssetRecord | None = None
        self.selected_snapshot_index = 0
        self.prev_snapshot_button: ttk.Button | None = None
        self.next_snapshot_button: ttk.Button | None = None
        self.pool_display_selection: dict[str, tuple[str, int] | None] = {
            "pick": None,
            "scene": None,
            "place": None,
        }
        self.pick_pool_images: list[tk.PhotoImage] = []
        self.place_pool_images: list[tk.PhotoImage] = []
        self.scene_pool_images: list[tk.PhotoImage] = []
        self.template_path_var = tk.StringVar(value=str(template_path))
        self.output_path_var = tk.StringVar(value=str(output_path))
        self.search_var = tk.StringVar()
        self.category_var = tk.StringVar(value="All")
        self.pose_filter_var = tk.StringVar(value="All")
        self.status_var = tk.StringVar(
            value=f"Loaded {len(self.catalog)} benchmark objects"
        )
        self.library_count_var = tk.StringVar()
        self.pick_mass_var = tk.DoubleVar(value=0.05)
        self.place_mass_var = tk.DoubleVar(value=1000.0)
        self.scene_mass_var = tk.DoubleVar(value=0.05)
        self.task_id_var = tk.StringVar()
        self.task_name_cn_var = tk.StringVar()
        self.task_name_en_var = tk.StringVar()
        self.init_scene_var = tk.StringVar()
        self.place_action_cn_var = tk.StringVar()
        self.place_action_en_var = tk.StringVar()

        self._build_ui()
        self._bind_events()
        self.refresh_library()
        self.load_template_from_path(initial=True)

    def _build_ui(self) -> None:
        self.root.title("GenieSim Pick-Place Task Template Editor")
        self.root.geometry("1680x980")
        self.root.minsize(1440, 900)

        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        self._configure_fonts(style)

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        path_frame = ttk.LabelFrame(outer, text="Template And Output", padding=10)
        path_frame.pack(fill=tk.X)
        path_frame.columnconfigure(1, weight=1)
        path_frame.columnconfigure(4, weight=1)

        ttk.Label(path_frame, text="Reference template").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(path_frame, textvariable=self.template_path_var).grid(
            row=0, column=1, columnspan=2, sticky="ew", pady=4
        )
        ttk.Button(
            path_frame,
            text="Browse",
            command=self.browse_template_path,
        ).grid(row=0, column=3, padx=6, pady=4)
        ttk.Button(
            path_frame,
            text="Load Template",
            command=self.load_template_from_path,
        ).grid(row=0, column=4, sticky="w", pady=4)

        ttk.Label(path_frame, text="Output task file").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(path_frame, textvariable=self.output_path_var).grid(
            row=1, column=1, columnspan=2, sticky="ew", pady=4
        )
        ttk.Button(
            path_frame,
            text="Browse",
            command=self.browse_output_path,
        ).grid(row=1, column=3, padx=6, pady=4)
        ttk.Button(
            path_frame,
            text="Write Task File",
            command=self.write_task_file,
        ).grid(row=1, column=4, sticky="w", pady=4)

        main_pane = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_pane, padding=(0, 0, 8, 0))
        right_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=2)
        main_pane.add(right_frame, weight=5)

        library_controls = ttk.LabelFrame(left_frame, text="Object Library", padding=10)
        library_controls.pack(fill=tk.X)
        library_controls.columnconfigure(1, weight=1)
        library_controls.columnconfigure(3, weight=1)
        library_controls.columnconfigure(5, weight=1)

        ttk.Label(library_controls, text="Search").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(library_controls, textvariable=self.search_var).grid(
            row=0, column=1, sticky="ew", pady=4
        )
        ttk.Label(library_controls, text="Category").grid(
            row=0, column=2, sticky="w", padx=(16, 8), pady=4
        )
        categories = ["All"] + sorted({record.category for record in self.catalog})
        ttk.Combobox(
            library_controls,
            textvariable=self.category_var,
            values=categories,
            state="readonly",
        ).grid(row=0, column=3, sticky="ew", pady=4)

        ttk.Label(library_controls, text="Pose labels").grid(
            row=0, column=4, sticky="w", padx=(16, 8), pady=4
        )
        ttk.Combobox(
            library_controls,
            textvariable=self.pose_filter_var,
            values=["All", "Has labeled poses", "Missing labeled poses"],
            state="readonly",
        ).grid(row=0, column=5, sticky="ew", pady=4)
        ttk.Label(
            library_controls,
            textvariable=self.library_count_var,
        ).grid(row=1, column=5, sticky="e", pady=(4, 0))

        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 8))

        columns = ("category", "grasp", "place")
        self.library_tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="tree headings",
            selectmode="extended",
        )
        self.library_tree.heading("#0", text="Asset ID")
        self.library_tree.heading("category", text="Category")
        self.library_tree.heading("grasp", text="Grasp")
        self.library_tree.heading("place", text="Place")
        self.library_tree.column("#0", width=240, stretch=True)
        self.library_tree.column("category", width=115, anchor="w")
        self.library_tree.column("grasp", width=70, anchor="center")
        self.library_tree.column("place", width=70, anchor="center")

        tree_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.library_tree.yview)
        tree_x = ttk.Scrollbar(
            tree_frame, orient=tk.HORIZONTAL, command=self.library_tree.xview
        )
        self.library_tree.configure(yscrollcommand=tree_y.set, xscrollcommand=tree_x.set)
        self.library_tree.grid(row=0, column=0, sticky="nsew")
        tree_y.grid(row=0, column=1, sticky="ns")
        tree_x.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        role_button_frame = ttk.Frame(left_frame)
        role_button_frame.pack(fill=tk.X)
        ttk.Button(
            role_button_frame,
            text="Add Selected To Pick Pool",
            command=lambda: self.add_selected_to_role("pick"),
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            role_button_frame,
            text="Add Selected To Scene Pool",
            command=lambda: self.add_selected_to_role("scene"),
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            role_button_frame,
            text="Add Selected To Target Container",
            command=lambda: self.add_selected_to_role("target_place"),
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            role_button_frame,
            text="Add Selected To Wrong Container",
            command=lambda: self.add_selected_to_role("wrong_place"),
        ).pack(side=tk.LEFT, padx=6)

        task_text_frame = ttk.LabelFrame(left_frame, text="Task Description And Action Text", padding=10)
        task_text_frame.pack(fill=tk.X, pady=(10, 0))
        task_text_frame.columnconfigure(1, weight=1)

        task_fields = [
            ("task", self.task_id_var),
            ("task_name", self.task_name_cn_var),
            ("english_task_name", self.task_name_en_var),
            ("init_scene_text", self.init_scene_var),
            ("place action_text", self.place_action_cn_var),
            ("place english_action_text", self.place_action_en_var),
        ]
        for row_index, (label, variable) in enumerate(task_fields):
            ttk.Label(task_text_frame, text=label).grid(
                row=row_index,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=4,
            )
            ttk.Entry(task_text_frame, textvariable=variable).grid(
                row=row_index,
                column=1,
                sticky="ew",
                pady=4,
            )

        visual_frame = ttk.Frame(right_frame)
        visual_frame.pack(fill=tk.BOTH, expand=True)
        visual_frame.rowconfigure(0, weight=1)
        visual_frame.rowconfigure(1, weight=1)
        visual_frame.columnconfigure(0, weight=3)
        visual_frame.columnconfigure(1, weight=4)
        visual_frame.columnconfigure(2, weight=4)

        detail_frame = ttk.LabelFrame(visual_frame, text="Current Selection", padding=10)
        detail_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        detail_frame.rowconfigure(0, weight=1)
        detail_frame.columnconfigure(0, weight=1)

        selected_canvas_frame = ttk.Frame(detail_frame)
        selected_canvas_frame.grid(row=0, column=0, sticky="nsew")
        selected_canvas_frame.rowconfigure(0, weight=1)
        selected_canvas_frame.columnconfigure(0, weight=1)

        self.selected_canvas = tk.Canvas(
            selected_canvas_frame,
            background="#faf7f2",
            highlightthickness=0,
            width=320,
            height=420,
        )
        self.selected_canvas.grid(row=0, column=0, sticky="nsew")
        selected_canvas_y = ttk.Scrollbar(
            selected_canvas_frame,
            orient=tk.VERTICAL,
            command=self.selected_canvas.yview,
        )
        selected_canvas_y.grid(row=0, column=1, sticky="ns")
        selected_canvas_x = ttk.Scrollbar(
            selected_canvas_frame,
            orient=tk.HORIZONTAL,
            command=self.selected_canvas.xview,
        )
        selected_canvas_x.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.selected_canvas.configure(
            xscrollcommand=selected_canvas_x.set,
            yscrollcommand=selected_canvas_y.set,
        )

        selected_controls = ttk.Frame(detail_frame)
        selected_controls.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        selected_controls.columnconfigure(2, weight=1)

        self.prev_snapshot_button = ttk.Button(
            selected_controls,
            text="<",
            width=3,
            command=lambda: self.change_selected_snapshot(-1),
        )
        self.prev_snapshot_button.grid(row=0, column=0, padx=(0, 6))
        ttk.Label(
            selected_controls,
            textvariable=self.selected_snapshot_info_var,
        ).grid(row=0, column=1, sticky="w")
        ttk.Scale(
            selected_controls,
            from_=1.0,
            to=4.0,
            variable=self.selected_image_scale_var,
            command=self.on_selected_scale_change,
        ).grid(row=0, column=2, sticky="ew", padx=8)
        ttk.Label(
            selected_controls,
            textvariable=self.selected_zoom_info_var,
            width=6,
        ).grid(row=0, column=3, padx=(0, 6))
        self.next_snapshot_button = ttk.Button(
            selected_controls,
            text=">",
            width=3,
            command=lambda: self.change_selected_snapshot(1),
        )
        self.next_snapshot_button.grid(row=0, column=4)

        ttk.Label(
            detail_frame,
            textvariable=self.selected_info_var,
            justify=tk.CENTER,
            anchor="center",
            wraplength=280,
        ).grid(row=2, column=0, sticky="ew", pady=(10, 0))

        pick_pool_frame = ttk.LabelFrame(visual_frame, text="Pick Pool Images", padding=10)
        pick_pool_frame.grid(row=0, column=1, sticky="nsew", padx=4)
        pick_pool_frame.rowconfigure(0, weight=1)
        pick_pool_frame.columnconfigure(0, weight=1)

        self.pick_pool_canvas = tk.Canvas(
            pick_pool_frame,
            background="#f8fafc",
            highlightthickness=0,
            height=420,
        )
        self.pick_pool_canvas.grid(row=0, column=0, sticky="nsew")
        pick_scroll = ttk.Scrollbar(
            pick_pool_frame,
            orient=tk.HORIZONTAL,
            command=self.pick_pool_canvas.xview,
        )
        pick_scroll.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.pick_pool_canvas.configure(xscrollcommand=pick_scroll.set)
        self.pick_pool_canvas.bind(
            "<Button-1>",
            lambda event: self.on_pool_canvas_click(event, "pick"),
        )
        pick_actions = ttk.Frame(pick_pool_frame)
        pick_actions.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        pick_actions.columnconfigure(0, weight=1)
        pick_actions.columnconfigure(1, weight=1)
        ttk.Button(
            pick_actions,
            text="Remove Selected Pick",
            command=lambda: self.remove_selected_from_display_pool("pick"),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(
            pick_actions,
            text="Clear Pick Pool",
            command=lambda: self.clear_role("pick"),
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        place_pool_frame = ttk.LabelFrame(
            visual_frame,
            text="Place Pool Images",
            padding=10,
        )
        place_pool_frame.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        place_pool_frame.rowconfigure(0, weight=1)
        place_pool_frame.columnconfigure(0, weight=1)

        self.place_pool_canvas = tk.Canvas(
            place_pool_frame,
            background="#f8fafc",
            highlightthickness=0,
            height=420,
        )
        self.place_pool_canvas.grid(row=0, column=0, sticky="nsew")
        place_scroll = ttk.Scrollbar(
            place_pool_frame,
            orient=tk.HORIZONTAL,
            command=self.place_pool_canvas.xview,
        )
        place_scroll.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.place_pool_canvas.configure(xscrollcommand=place_scroll.set)
        self.place_pool_canvas.bind(
            "<Button-1>",
            lambda event: self.on_pool_canvas_click(event, "place"),
        )
        place_actions = ttk.Frame(place_pool_frame)
        place_actions.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        for col_index in range(3):
            place_actions.columnconfigure(col_index, weight=1)
        ttk.Button(
            place_actions,
            text="Remove Selected",
            command=lambda: self.remove_selected_from_display_pool("place"),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(
            place_actions,
            text="Clear Target",
            command=lambda: self.clear_role("target_place"),
        ).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(
            place_actions,
            text="Clear Wrong",
            command=lambda: self.clear_role("wrong_place"),
        ).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        scene_pool_frame = ttk.LabelFrame(
            visual_frame,
            text="Scene Objects Pool Images",
            padding=10,
        )
        scene_pool_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        scene_pool_frame.rowconfigure(0, weight=1)
        scene_pool_frame.columnconfigure(0, weight=1)

        self.scene_pool_canvas = tk.Canvas(
            scene_pool_frame,
            background="#f8fafc",
            highlightthickness=0,
            height=260,
        )
        self.scene_pool_canvas.grid(row=0, column=0, sticky="nsew")
        scene_scroll = ttk.Scrollbar(
            scene_pool_frame,
            orient=tk.HORIZONTAL,
            command=self.scene_pool_canvas.xview,
        )
        scene_scroll.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.scene_pool_canvas.configure(xscrollcommand=scene_scroll.set)
        self.scene_pool_canvas.bind(
            "<Button-1>",
            lambda event: self.on_pool_canvas_click(event, "scene"),
        )
        scene_actions = ttk.Frame(scene_pool_frame)
        scene_actions.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        scene_actions.columnconfigure(0, weight=1)
        scene_actions.columnconfigure(1, weight=1)
        ttk.Button(
            scene_actions,
            text="Remove Selected Scene",
            command=lambda: self.remove_selected_from_display_pool("scene"),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(
            scene_actions,
            text="Clear Scene Pool",
            command=lambda: self.clear_role("scene"),
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        role_frame = ttk.LabelFrame(right_frame, text="Selection Lists", padding=10)
        role_frame.pack(fill=tk.X, pady=(10, 0))
        role_frame.rowconfigure(0, weight=1)
        for col_index in range(4):
            role_frame.columnconfigure(col_index, weight=1)

        positions = {
            "pick": (0, 0),
            "scene": (0, 1),
            "target_place": (0, 2),
            "wrong_place": (0, 3),
        }
        for role, (row_index, col_index) in positions.items():
            frame = ttk.LabelFrame(role_frame, text=ROLE_LABELS[role], padding=8)
            frame.grid(row=row_index, column=col_index, sticky="nsew", padx=5, pady=5)
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            listbox = tk.Listbox(
                frame,
                exportselection=False,
                font=(self.ui_font_family, 10),
                height=6,
            )
            listbox.grid(row=0, column=0, sticky="nsew")
            scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=listbox.yview)
            scrollbar.grid(row=0, column=1, sticky="ns")
            listbox.configure(yscrollcommand=scrollbar.set)
            self.role_listboxes[role] = listbox
            buttons = ttk.Frame(frame)
            buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
            ttk.Button(
                buttons,
                text="Remove Selected",
                command=lambda role=role: self.remove_selected_from_role(role),
            ).pack(side=tk.LEFT)
            ttk.Button(
                buttons,
                text="Clear",
                command=lambda role=role: self.clear_role(role),
            ).pack(side=tk.LEFT, padx=6)

        ttk.Label(
            outer,
            textvariable=self.status_var,
            anchor="w",
            relief=tk.SUNKEN,
            padding=6,
        ).pack(fill=tk.X, pady=(10, 0))

        self._update_selected_snapshot_controls()
        self._render_selected_canvas(None)
        self.refresh_pool_views()

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

    def _bind_events(self) -> None:
        self.search_var.trace_add("write", lambda *_: self.refresh_library())
        self.category_var.trace_add("write", lambda *_: self.refresh_library())
        self.pose_filter_var.trace_add("write", lambda *_: self.refresh_library())
        self.library_tree.bind("<<TreeviewSelect>>", self.on_library_select)

    def on_selected_scale_change(self, _value: str) -> None:
        zoom_percent = int(round(self.selected_image_scale_var.get() * 100))
        self.selected_zoom_info_var.set(f"{zoom_percent}%")
        self._render_selected_canvas(self.current_selected_record)

    def change_selected_snapshot(self, delta: int) -> None:
        if self.current_selected_record is None or not self.current_selected_record.snapshot_paths:
            return
        total = len(self.current_selected_record.snapshot_paths)
        self.selected_snapshot_index = (self.selected_snapshot_index + delta) % total
        self._update_selected_snapshot_controls()
        self._render_selected_canvas(self.current_selected_record)

    def _update_selected_snapshot_controls(self) -> None:
        total = len(self.current_selected_record.snapshot_paths) if self.current_selected_record else 0
        if total:
            self.selected_snapshot_info_var.set(
                f"snapshot {self.selected_snapshot_index + 1}/{total}"
            )
        else:
            self.selected_snapshot_info_var.set("snapshot 0/0")

        state = tk.NORMAL if total > 1 else tk.DISABLED
        if self.prev_snapshot_button is not None:
            self.prev_snapshot_button.configure(state=state)
        if self.next_snapshot_button is not None:
            self.next_snapshot_button.configure(state=state)

    def browse_template_path(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select reference task template",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(DEFAULT_TEMPLATE_PATH.parent),
        )
        if selected:
            self.template_path_var.set(selected)

    def browse_output_path(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Choose output task file",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(Path(self.output_path_var.get()).parent),
            initialfile=Path(self.output_path_var.get()).name,
        )
        if selected:
            self.output_path_var.set(selected)

    def refresh_library(self) -> None:
        query = self.search_var.get().strip().lower()
        query_terms = [term for term in query.split() if term]
        category = self.category_var.get()
        pose_filter = self.pose_filter_var.get()
        self.filtered_records = []

        for record in self.catalog:
            haystack = " ".join(
                [
                    record.asset_id,
                    record.category,
                    record.semantic_name,
                    record.color,
                    record.description,
                    record.data_info_dir,
                ]
            ).lower()
            if query_terms and not all(term in haystack for term in query_terms):
                continue
            if category != "All" and record.category != category:
                continue
            if pose_filter == "Has labeled poses" and not record.interaction.has_any:
                continue
            if pose_filter == "Missing labeled poses" and record.interaction.has_any:
                continue
            self.filtered_records.append(record)

        for item_id in self.library_tree.get_children():
            self.library_tree.delete(item_id)
        self.tree_record_map.clear()

        for record in self.filtered_records:
            item_id = record.asset_id
            self.tree_record_map[item_id] = record
            self.library_tree.insert(
                "",
                tk.END,
                iid=item_id,
                text=record.asset_id,
                values=(
                    record.category,
                    "Y" if record.interaction.has_grasp else "N",
                    "Y" if record.interaction.has_place else "N",
                ),
            )

        self.library_count_var.set(
            f"{len(self.filtered_records)} shown / {len(self.catalog)} total"
        )

    def on_library_select(self, _event: tk.Event[Any] | None = None) -> None:
        selected_records = self.get_selected_library_records()
        if not selected_records:
            self.set_detail(None)
            self.status_var.set("No object selected")
            return

        record = selected_records[0]
        self.set_detail(record)
        selected_count = len(selected_records)
        suffix = "" if selected_count == 1 else f" | multi-select: {selected_count}"
        self.status_var.set(
            f"{record.asset_id}: {record.interaction.status_text()}{suffix}"
        )

    def set_detail(self, record: AssetRecord | None) -> None:
        if record is None:
            self.current_selected_record = None
            self.selected_snapshot_index = 0
            self.current_preview_image = None
            self.selected_info_var.set("")
            self._update_selected_snapshot_controls()
            self._render_selected_canvas(None)
            return

        if self.current_selected_record is None or self.current_selected_record.asset_id != record.asset_id:
            self.selected_snapshot_index = 0
        self.current_selected_record = record
        self._update_selected_snapshot_controls()
        self.selected_info_var.set(
            "\n".join(
                [
                    record.asset_id,
                    f"{record.category} | {record.semantic_name or '-'} | {record.color or '-'}",
                    f"operation poses: {record.interaction.status_text()}",
                ]
            )
        )
        self._render_selected_canvas(record)

    def _render_selected_canvas(self, record: AssetRecord | None) -> None:
        if self.selected_canvas is None:
            return
        canvas = self.selected_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), int(canvas.cget("width")))
        height = max(canvas.winfo_height(), int(canvas.cget("height")))

        if record is None:
            canvas.configure(scrollregion=(0, 0, width, height))
            canvas.create_text(
                width / 2,
                height / 2,
                text="Select an object",
                font=(self.ui_font_family, 16, "bold"),
                fill="#6b7280",
            )
            return

        snapshot_path = None
        if record.snapshot_paths:
            snapshot_path = record.snapshot_paths[
                min(self.selected_snapshot_index, len(record.snapshot_paths) - 1)
            ]
        viewport_w = max(width - 40, 220)
        viewport_h = max(height - 40, 260)
        zoom = max(self.selected_image_scale_var.get(), 1.0)
        image = self._load_scaled_photo_image(
            snapshot_path,
            int(viewport_w * zoom),
            int(viewport_h * zoom),
        )
        self.current_preview_image = image
        if image is None:
            canvas.configure(scrollregion=(0, 0, width, height))
            canvas.create_rectangle(30, 30, width - 30, height - 30, outline="#cbd5e1", width=2)
            canvas.create_text(
                width / 2,
                height / 2,
                text=f"No preview\n{record.asset_id}",
                font=(self.ui_font_family, 14, "bold"),
                fill="#6b7280",
                justify=tk.CENTER,
            )
            return

        image_x = max((width - image.width()) // 2, 18)
        image_y = max((height - image.height()) // 2, 18)
        canvas.create_rectangle(
            image_x - 12,
            image_y - 12,
            image_x + image.width() + 12,
            image_y + image.height() + 12,
            outline="#d6d3d1",
            width=2,
        )
        canvas.create_image(image_x, image_y, image=image, anchor=tk.NW)
        canvas.configure(
            scrollregion=(
                0,
                0,
                max(width, image_x + image.width() + 24),
                max(height, image_y + image.height() + 24),
            )
        )

    def _load_scaled_photo_image(
        self,
        image_path: Path | None,
        max_width: int,
        max_height: int,
    ) -> tk.PhotoImage | None:
        if not image_path or not image_path.exists():
            return None
        try:
            image = tk.PhotoImage(file=str(image_path))
        except tk.TclError:
            return None

        width_ratio = max_width / max(image.width(), 1)
        height_ratio = max_height / max(image.height(), 1)
        scale_ratio = max(min(width_ratio, height_ratio), 0.1)
        numerator = max(1, int(round(scale_ratio * 10)))
        denominator = 10
        image = image.zoom(numerator, numerator).subsample(denominator, denominator)
        return image

    def get_selected_library_records(self) -> list[AssetRecord]:
        item_ids = self.library_tree.selection()
        return [
            self.tree_record_map[item_id]
            for item_id in item_ids
            if item_id in self.tree_record_map
        ]

    def add_selected_to_role(self, role: str) -> None:
        selected_records = self.get_selected_library_records()
        if not selected_records:
            messagebox.showinfo("No selection", "Select one or more objects from the library first.")
            return
        current = self.role_records[role]
        self.role_records[role] = dedupe_records(current + selected_records)
        self.refresh_role_listbox(role)
        self.status_var.set(
            f"Added {len(selected_records)} object(s) to {ROLE_LABELS[role]}"
        )

    def on_pool_canvas_click(self, event: tk.Event[Any], display_role: str) -> None:
        canvas = {
            "pick": self.pick_pool_canvas,
            "scene": self.scene_pool_canvas,
            "place": self.place_pool_canvas,
        }.get(display_role)
        if canvas is None:
            return

        current_item = canvas.find_withtag("current")
        if not current_item:
            return
        tags = canvas.gettags(current_item[0])

        actual_role = None
        item_index = None
        for tag in tags:
            if tag.startswith("pool_"):
                actual_role = tag[len("pool_") :]
            elif tag.startswith("idx_"):
                item_index = int(tag[len("idx_") :])

        if actual_role is None or item_index is None:
            return

        self.pool_display_selection[display_role] = (actual_role, item_index)
        self._sync_listbox_selection(actual_role, item_index)
        if 0 <= item_index < len(self.role_records[actual_role]):
            self.set_detail(self.role_records[actual_role][item_index])
            self.status_var.set(
                f"Selected {self.role_records[actual_role][item_index].asset_id} from {ROLE_LABELS[actual_role]}"
            )
        self.refresh_pool_views()

    def _sync_listbox_selection(self, role: str, item_index: int) -> None:
        if role not in self.role_listboxes:
            return
        listbox = self.role_listboxes[role]
        listbox.selection_clear(0, tk.END)
        if 0 <= item_index < listbox.size():
            listbox.selection_set(item_index)
            listbox.activate(item_index)
            listbox.see(item_index)

    def remove_selected_from_display_pool(self, display_role: str) -> None:
        selection = self.pool_display_selection.get(display_role)
        if selection is None:
            self.status_var.set(f"No selected item in the {display_role} image panel")
            return

        actual_role, item_index = selection
        if not (0 <= item_index < len(self.role_records[actual_role])):
            self.pool_display_selection[display_role] = None
            self.refresh_pool_views()
            return

        removed_record = self.role_records[actual_role][item_index]
        self.role_records[actual_role] = [
            record
            for index, record in enumerate(self.role_records[actual_role])
            if index != item_index
        ]
        self.pool_display_selection[display_role] = None
        if actual_role in self.role_listboxes:
            self.role_listboxes[actual_role].selection_clear(0, tk.END)
        self.refresh_role_listbox(actual_role)
        self.status_var.set(
            f"Removed {removed_record.asset_id} from {ROLE_LABELS[actual_role]}"
        )

    def remove_selected_from_role(self, role: str) -> None:
        listbox = self.role_listboxes[role]
        selected_indices = listbox.curselection()
        if not selected_indices:
            display_role = role
            if role in {"target_place", "wrong_place"}:
                display_role = "place"
            selection = self.pool_display_selection.get(display_role)
            if selection is not None and selection[0] == role:
                self.remove_selected_from_display_pool(display_role)
            return
        remaining = [
            record
            for index, record in enumerate(self.role_records[role])
            if index not in selected_indices
        ]
        self.role_records[role] = remaining
        if role == "pick":
            self.pool_display_selection["pick"] = None
        elif role == "scene":
            self.pool_display_selection["scene"] = None
        elif role in {"target_place", "wrong_place"}:
            self.pool_display_selection["place"] = None
        self.refresh_role_listbox(role)

    def clear_role(self, role: str) -> None:
        self.role_records[role] = []
        if role == "pick":
            self.pool_display_selection["pick"] = None
        elif role == "scene":
            self.pool_display_selection["scene"] = None
        elif role in {"target_place", "wrong_place"}:
            self.pool_display_selection["place"] = None
        self.refresh_role_listbox(role)

    def refresh_role_listbox(self, role: str) -> None:
        if role not in self.role_listboxes:
            self.refresh_pool_views()
            return
        listbox = self.role_listboxes[role]
        listbox.delete(0, tk.END)
        for record in self.role_records[role]:
            listbox.insert(
                tk.END,
                f"{record.asset_id} | G:{'Y' if record.interaction.has_grasp else 'N'} "
                f"P:{'Y' if record.interaction.has_place else 'N'}",
            )
        self.refresh_pool_views()

    def refresh_pool_views(self) -> None:
        self._render_pool_canvas(
            display_role="pick",
            canvas=self.pick_pool_canvas,
            records=self.role_records["pick"],
            image_refs=self.pick_pool_images,
            label_fn=lambda index, _record: f"PICK {index}",
            border_fn=lambda _record: "#2563eb",
            empty_text="No pick candidates selected",
        )

        self._render_pool_canvas(
            display_role="scene",
            canvas=self.scene_pool_canvas,
            records=self.role_records["scene"],
            image_refs=self.scene_pool_images,
            label_fn=lambda index, _record: f"SCENE {index}",
            border_fn=lambda _record: "#475569",
            empty_text="No scene objects selected",
        )

        place_records: list[tuple[AssetRecord, str]] = []
        for record in self.role_records["target_place"]:
            place_records.append((record, "TARGET"))
        for record in self.role_records["wrong_place"]:
            place_records.append((record, "WRONG"))

        self._render_pool_canvas(
            display_role="place",
            canvas=self.place_pool_canvas,
            records=place_records,
            image_refs=self.place_pool_images,
            label_fn=lambda _index, item: item[1],
            border_fn=lambda item: "#15803d" if item[1] == "TARGET" else "#b91c1c",
            empty_text="No place containers selected",
        )

    def _render_pool_canvas(
        self,
        display_role: str,
        canvas: tk.Canvas | None,
        records: list[Any],
        image_refs: list[tk.PhotoImage],
        label_fn,
        border_fn,
        empty_text: str,
    ) -> None:
        if canvas is None:
            return

        canvas.delete("all")
        image_refs.clear()
        height = max(canvas.winfo_height(), int(canvas.cget("height")))
        if not records:
            self.pool_display_selection[display_role] = None
            canvas.create_text(
                220,
                height / 2,
                text=empty_text,
                font=(self.ui_font_family, 16, "bold"),
                fill="#6b7280",
            )
            canvas.configure(scrollregion=(0, 0, 440, height))
            return

        x = 16
        gap = 18
        thumb_max_w = 170
        thumb_max_h = 170
        label_y = 22
        image_y = 140
        name_y = 260
        bottom_y = 292

        for index, item in enumerate(records, start=1):
            record = item[0] if isinstance(item, tuple) else item
            actual_role = item[1].lower() if display_role == "place" and isinstance(item, tuple) else display_role
            if actual_role == "target":
                actual_role = "target_place"
            elif actual_role == "wrong":
                actual_role = "wrong_place"
            label_text = label_fn(index, item)
            border_color = border_fn(item)
            image = self._load_scaled_photo_image(record.snapshot_path, thumb_max_w, thumb_max_h)
            tile_w = thumb_max_w + 24
            tile_h = 280
            tile_tag = f"tile_{display_role}_{index - 1}"
            tags = (tile_tag, f"display_{display_role}", f"pool_{actual_role}", f"idx_{index - 1}")
            is_selected = self.pool_display_selection.get(display_role) == (actual_role, index - 1)
            outline_color = "#f59e0b" if is_selected else border_color
            outline_width = 5 if is_selected else 3

            canvas.create_rectangle(
                x,
                12,
                x + tile_w,
                12 + tile_h,
                outline=outline_color,
                width=outline_width,
                tags=tags,
            )
            canvas.create_text(
                x + tile_w / 2,
                label_y,
                text=label_text,
                font=(self.ui_font_family, 10, "bold"),
                fill=outline_color,
                tags=tags,
            )

            if image is None:
                canvas.create_rectangle(
                    x + 12,
                    52,
                    x + tile_w - 12,
                    228,
                    outline="#cbd5e1",
                    width=2,
                    tags=tags,
                )
                canvas.create_text(
                    x + tile_w / 2,
                    image_y,
                    text="No preview",
                    font=(self.ui_font_family, 12, "bold"),
                    fill="#6b7280",
                    tags=tags,
                )
            else:
                image_refs.append(image)
                canvas.create_image(
                    x + tile_w / 2,
                    image_y,
                    image=image,
                    tags=tags,
                )

            canvas.create_text(
                x + tile_w / 2,
                name_y,
                text=record.asset_id,
                width=tile_w - 16,
                font=(self.ui_font_family, 10, "bold"),
                justify=tk.CENTER,
                tags=tags,
            )
            canvas.create_text(
                x + tile_w / 2,
                bottom_y,
                text=f"{record.semantic_name or '-'} | {record.interaction.status_text()}",
                width=tile_w - 16,
                font=(self.ui_font_family, 9),
                fill="#334155",
                justify=tk.CENTER,
                tags=tags,
            )
            x += tile_w + gap

        canvas.configure(scrollregion=(0, 0, x, height))

    def load_template_from_path(self, initial: bool = False) -> None:
        template_path = Path(self.template_path_var.get()).expanduser()
        if not template_path.exists():
            if initial:
                self.status_var.set(f"Template not found: {template_path}")
                return
            messagebox.showerror("Template missing", f"Template not found:\n{template_path}")
            return

        try:
            reference_template = load_json(template_path)
            template_slots = detect_template_slots(reference_template)
        except Exception as exc:  # pragma: no cover - UI error path
            messagebox.showerror("Load failed", str(exc))
            return

        self.reference_template = reference_template
        self.template_slots = template_slots

        missing_assets: list[str] = []

        pick_entries = expand_slot_entries(
            reference_template["objects"]["task_related_objects"][template_slots.pick_index]
        )
        target_place_entries = expand_slot_entries(
            reference_template["objects"]["task_related_objects"][template_slots.target_place_index]
        )
        wrong_place_entries: list[dict[str, Any]] = []
        for index in template_slots.wrong_place_indices:
            wrong_place_entries.extend(
                expand_slot_entries(reference_template["objects"]["task_related_objects"][index])
            )

        scene_entries: list[dict[str, Any]] = []
        for group in reference_template.get("objects", {}).get("scene_objects", []):
            scene_entries.extend(group.get("available_objects", []))

        self.role_records["pick"], missing = resolve_records_from_entries(
            pick_entries, self.catalog_by_data_info_dir
        )
        missing_assets.extend(missing)
        self.role_records["target_place"], missing = resolve_records_from_entries(
            target_place_entries, self.catalog_by_data_info_dir
        )
        missing_assets.extend(missing)
        self.role_records["wrong_place"], missing = resolve_records_from_entries(
            wrong_place_entries, self.catalog_by_data_info_dir
        )
        missing_assets.extend(missing)
        self.role_records["scene"], missing = resolve_records_from_entries(
            scene_entries, self.catalog_by_data_info_dir
        )
        missing_assets.extend(missing)

        for role in ROLE_ORDER:
            self.refresh_role_listbox(role)

        self._populate_text_fields(reference_template, template_path)
        self.refresh_preview()

        if missing_assets:
            self.status_var.set(
                f"Loaded template with {len(missing_assets)} missing asset entries"
            )
        else:
            self.status_var.set(
                f"Loaded template '{template_path.name}'"
            )

    def _populate_text_fields(
        self,
        template: dict[str, Any],
        template_path: Path,
    ) -> None:
        task_description = template.get("task_description", {})
        place_stage = find_stage(template, "place")
        place_action = place_stage.get("action_description", {})
        self.task_id_var.set(template.get("task", template_path.stem))
        self.task_name_cn_var.set(task_description.get("task_name", ""))
        self.task_name_en_var.set(task_description.get("english_task_name", ""))
        self.init_scene_var.set(task_description.get("init_scene_text", ""))
        self.place_action_cn_var.set(place_action.get("action_text", ""))
        self.place_action_en_var.set(place_action.get("english_action_text", ""))

        pick_slot = template["objects"]["task_related_objects"][self.template_slots.pick_index]
        target_place_slot = template["objects"]["task_related_objects"][self.template_slots.target_place_index]
        self.pick_mass_var.set(float(pick_slot.get("mass", 0.05)))
        self.place_mass_var.set(float(target_place_slot.get("mass", 1000.0)))

        scene_objects = template.get("objects", {}).get("scene_objects", [])
        scene_mass = self.pick_mass_var.get()
        if scene_objects:
            available_objects = scene_objects[0].get("available_objects", [])
            if available_objects:
                scene_mass = float(available_objects[0].get("mass", scene_mass))
        self.scene_mass_var.set(scene_mass)

        output_path = build_default_output_path(template_path)
        self.output_path_var.set(str(output_path))

    def collect_task_text(self) -> tuple[str, str, str, str, str, str]:
        output_path = Path(self.output_path_var.get()).expanduser()
        task_id = self.task_id_var.get().strip() or output_path.stem
        return (
            task_id,
            self.task_name_cn_var.get().strip(),
            self.task_name_en_var.get().strip(),
            self.init_scene_var.get().strip(),
            self.place_action_cn_var.get().strip(),
            self.place_action_en_var.get().strip(),
        )

    def generate_template(self) -> dict[str, Any]:
        if self.reference_template is None or self.template_slots is None:
            raise ValueError("No reference template is loaded")

        task_text = self.collect_task_text()
        return build_task_template(
            reference_template=self.reference_template,
            slots=self.template_slots,
            role_records=self.role_records,
            task_id=task_text[0],
            task_name_cn=task_text[1],
            task_name_en=task_text[2],
            init_scene_text=task_text[3],
            place_action_cn=task_text[4],
            place_action_en=task_text[5],
            pick_mass=float(self.pick_mass_var.get()),
            place_mass=float(self.place_mass_var.get()),
            scene_mass=float(self.scene_mass_var.get()),
        )

    def refresh_preview(self) -> None:
        try:
            self.generate_template()
        except Exception as exc:
            self.status_var.set(f"Preview error: {exc}")
            return

        self.status_var.set("Task configuration ready")

    def write_task_file(self) -> None:
        output_path = Path(self.output_path_var.get()).expanduser()
        if output_path.suffix.lower() != ".json":
            messagebox.showerror(
                "Invalid output path",
                "Output path must end with .json",
            )
            return

        try:
            generated = self.generate_template()
            dump_json(output_path, generated)
            self.refresh_preview()
        except Exception as exc:
            messagebox.showerror("Write failed", str(exc))
            return

        self.status_var.set(f"Wrote task file: {output_path}")
        messagebox.showinfo("Task file written", f"Saved:\n{output_path}")


def main() -> None:
    args = parse_args()
    template_path = args.template.expanduser()
    output_path = args.output.expanduser() if args.output else build_default_output_path(template_path)
    root = tk.Tk()
    editor = TaskTemplateEditor(
        root=root,
        asset_root=args.asset_root.expanduser(),
        template_path=template_path,
        output_path=output_path,
    )
    editor.root.mainloop()


if __name__ == "__main__":
    main()
