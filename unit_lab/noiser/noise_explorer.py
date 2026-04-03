from __future__ import annotations

import argparse
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk

from noise_models import apply_noise, default_parameters, get_mode_spec, list_mode_specs, sample_current_parameters


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = SCRIPT_DIR / "head_color_first_frame.png"
SNAPSHOT_DIR = SCRIPT_DIR / "snapshots"
CARD_SIZE = (250, 180)
PREVIEW_PANEL_SIZE = (620, 480)
WINDOW_SIZE = "1720x980"
BACKGROUND_RGB = (245, 244, 239)
CARD_RGB = (255, 255, 255)
ACTIVE_BORDER_RGB = (41, 96, 104)
INACTIVE_BORDER_RGB = (193, 199, 202)

if hasattr(Image, "Resampling"):
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:
    RESAMPLE_LANCZOS = Image.LANCZOS


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def resize_longest_edge(image: np.ndarray, longest_edge: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    width, height = pil_image.size
    scale = min(1.0, float(longest_edge) / float(max(width, height)))
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    if new_size == pil_image.size:
        return image
    return np.asarray(pil_image.resize(new_size, RESAMPLE_LANCZOS), dtype=np.uint8)


def format_param_summary(mode_key: str, params: dict[str, float | int]) -> str:
    mode = get_mode_spec(mode_key)
    return ", ".join(
        f"{spec.label}={spec.format_value(params.get(spec.key, spec.default))}"
        for spec in mode.parameters
    )


def make_card(
    image: np.ndarray,
    title: str,
    subtitle: str = "",
    active: bool = False,
    size: tuple[int, int] = CARD_SIZE,
) -> Image.Image:
    canvas = Image.new("RGB", size, CARD_RGB)
    border_rgb = ACTIVE_BORDER_RGB if active else INACTIVE_BORDER_RGB
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.rounded_rectangle(
        (0, 0, size[0] - 1, size[1] - 1),
        radius=12,
        outline=border_rgb,
        width=3 if active else 2,
        fill=CARD_RGB,
    )
    draw.text((12, 10), title, fill=(30, 34, 35), font=font)
    if subtitle:
        draw.text((12, 28), subtitle, fill=(96, 99, 101), font=font)

    thumb = Image.fromarray(image)
    preview = ImageOps.contain(thumb, (size[0] - 24, size[1] - 56), method=RESAMPLE_LANCZOS)
    left = (size[0] - preview.width) // 2
    top = 48 + max(0, (size[1] - 56 - preview.height) // 2)
    canvas.paste(preview, (left, top))
    return canvas


def build_comparison_panel(
    original: np.ndarray,
    noised: np.ndarray,
    mode_key: str,
    params: dict[str, float | int],
    seed: int,
) -> Image.Image:
    mode = get_mode_spec(mode_key)
    original_panel = ImageOps.contain(Image.fromarray(original), PREVIEW_PANEL_SIZE, method=RESAMPLE_LANCZOS)
    noised_panel = ImageOps.contain(Image.fromarray(noised), PREVIEW_PANEL_SIZE, method=RESAMPLE_LANCZOS)

    header_h = 56
    footer_h = 36
    gap = 24
    width = original_panel.width + noised_panel.width + gap * 3
    height = max(original_panel.height, noised_panel.height) + header_h + footer_h + 24
    canvas = Image.new("RGB", (width, height), BACKGROUND_RGB)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((18, 16), "Original", fill=(29, 34, 35), font=font)
    draw.text((original_panel.width + gap * 2 + 18, 16), mode.title, fill=(29, 34, 35), font=font)
    draw.text((18, height - footer_h), f"seed={seed}", fill=(70, 75, 78), font=font)
    draw.text(
        (120, height - footer_h),
        format_param_summary(mode_key, params),
        fill=(70, 75, 78),
        font=font,
    )

    canvas.paste(original_panel, (gap, header_h))
    canvas.paste(noised_panel, (original_panel.width + gap * 2, header_h))
    return canvas


def build_export_grid(image: np.ndarray, seed: int) -> Image.Image:
    cards = [make_card(image, title="Original", subtitle="reference", active=False)]
    preview_source = resize_longest_edge(image, 480)
    for mode in list_mode_specs():
        params = default_parameters(mode.key)
        noised = apply_noise(mode.key, preview_source, seed=seed, params=params)
        subtitle = format_param_summary(mode.key, params)
        cards.append(make_card(noised, title=mode.title, subtitle=subtitle, active=False))

    columns = 3
    rows = int(np.ceil(len(cards) / columns))
    padding = 18
    width = columns * CARD_SIZE[0] + (columns + 1) * padding
    height = rows * CARD_SIZE[1] + (rows + 1) * padding
    grid = Image.new("RGB", (width, height), BACKGROUND_RGB)

    for index, card in enumerate(cards):
        row = index // columns
        col = index % columns
        x = padding + col * (CARD_SIZE[0] + padding)
        y = padding + row * (CARD_SIZE[1] + padding)
        grid.paste(card, (x, y))
    return grid


class NoiseExplorerApp:
    def __init__(self, root: tk.Tk, image_path: Path, seed: int) -> None:
        self.root = root
        self.root.title("Camera Noise Explorer")
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(1480, 860)

        self.mode_specs = list_mode_specs()
        self.mode_states = {mode.key: default_parameters(mode.key) for mode in self.mode_specs}
        self.mode_var = tk.StringVar(value=self.mode_specs[0].key)
        self.seed_var = tk.IntVar(value=int(seed))
        self.image_path_var = tk.StringVar(value=str(image_path))
        self.status_var = tk.StringVar()
        self.info_var = tk.StringVar()

        self._refresh_after_id: str | None = None
        self.active_param_vars: dict[str, tk.Variable] = {}
        self.active_value_labels: dict[str, tk.StringVar] = {}
        self.thumbnail_photo_refs: dict[str, ImageTk.PhotoImage] = {}
        self.preview_photo_ref: ImageTk.PhotoImage | None = None
        self.last_output: np.ndarray | None = None

        self.current_image_path = Path(image_path)
        self.original_image = load_rgb_image(self.current_image_path)
        self.thumbnail_source = resize_longest_edge(self.original_image, 360)

        self._build_layout()
        self._rebuild_parameter_panel()
        self.refresh_all()

    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.root, padding=14)
        preview = ttk.Frame(self.root, padding=(0, 14, 14, 14))
        controls.grid(row=0, column=0, sticky="nsw")
        preview.grid(row=0, column=1, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)

        mode_frame = ttk.LabelFrame(controls, text="Noise Modes", padding=10)
        mode_frame.grid(row=0, column=0, sticky="ew")
        for row, mode in enumerate(self.mode_specs):
            suffix = "used" if "Used by publish_noised_rgb" in mode.pipeline_note else "not used"
            button = ttk.Radiobutton(
                mode_frame,
                text=f"{mode.title} ({suffix})",
                value=mode.key,
                variable=self.mode_var,
                command=self._on_mode_changed,
            )
            button.grid(row=row, column=0, sticky="w", pady=1)

        control_frame = ttk.LabelFrame(controls, text="Inputs", padding=10)
        control_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(control_frame, text="Image").grid(row=0, column=0, sticky="w")
        ttk.Button(control_frame, text="Load Image", command=self._load_image).grid(row=0, column=1, sticky="e")
        ttk.Label(
            control_frame,
            textvariable=self.image_path_var,
            wraplength=340,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 10))

        ttk.Label(control_frame, text="Seed").grid(row=2, column=0, sticky="w")
        seed_spin = ttk.Spinbox(control_frame, from_=0, to=999999, textvariable=self.seed_var, width=12)
        seed_spin.grid(row=2, column=1, sticky="e")
        self.seed_var.trace_add("write", lambda *_args: self.schedule_refresh())

        button_row = ttk.Frame(control_frame)
        button_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(button_row, text="Randomize Seed", command=self._randomize_seed).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(button_row, text="Sample Current Range", command=self._sample_current_range).grid(
            row=0,
            column=1,
            padx=6,
        )
        ttk.Button(button_row, text="Reset Mode", command=self._reset_mode).grid(row=0, column=2, padx=6)
        ttk.Button(button_row, text="Save Snapshot", command=self._save_snapshot).grid(row=0, column=3, padx=(6, 0))

        self.parameter_frame = ttk.LabelFrame(controls, text="Parameters", padding=10)
        self.parameter_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))

        info_frame = ttk.LabelFrame(controls, text="Mode Notes", padding=10)
        info_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(info_frame, textvariable=self.status_var, foreground="#255d5d", wraplength=360, justify="left").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(info_frame, textvariable=self.info_var, wraplength=360, justify="left").grid(
            row=1,
            column=0,
            sticky="w",
            pady=(8, 0),
        )

        preview_frame = ttk.LabelFrame(preview, text="Preview", padding=10)
        preview_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        overview_frame = ttk.LabelFrame(preview, text="All Modes Overview", padding=10)
        overview_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self.card_labels: dict[str, ttk.Label] = {}
        card_keys = ["__original__"] + [mode.key for mode in self.mode_specs]
        for index, key in enumerate(card_keys):
            row = index // 4
            col = index % 4
            label = ttk.Label(overview_frame)
            label.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            self.card_labels[key] = label
            if key != "__original__":
                label.bind("<Button-1>", lambda _event, mode_key=key: self._select_mode(mode_key))

    def _load_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Image",
            initialdir=str(self.current_image_path.parent),
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        if not selected:
            return
        self.current_image_path = Path(selected)
        self.original_image = load_rgb_image(self.current_image_path)
        self.thumbnail_source = resize_longest_edge(self.original_image, 360)
        self.image_path_var.set(str(self.current_image_path))
        self.schedule_refresh()

    def _randomize_seed(self) -> None:
        self.seed_var.set(int(np.random.default_rng().integers(0, 1_000_000)))

    def _sample_current_range(self) -> None:
        mode_key = self.mode_var.get()
        sampled = sample_current_parameters(mode_key)
        self.mode_states[mode_key].update(sampled)
        self._rebuild_parameter_panel()
        self.schedule_refresh()

    def _reset_mode(self) -> None:
        mode_key = self.mode_var.get()
        self.mode_states[mode_key] = default_parameters(mode_key)
        self._rebuild_parameter_panel()
        self.schedule_refresh()

    def _select_mode(self, mode_key: str) -> None:
        self.mode_var.set(mode_key)
        self._on_mode_changed()

    def _on_mode_changed(self) -> None:
        self._rebuild_parameter_panel()
        self.schedule_refresh()

    def _rebuild_parameter_panel(self) -> None:
        for child in self.parameter_frame.winfo_children():
            child.destroy()

        self.active_param_vars.clear()
        self.active_value_labels.clear()
        mode = get_mode_spec(self.mode_var.get())
        state = self.mode_states[mode.key]

        for row, spec in enumerate(mode.parameters):
            base_row = row * 3
            ttk.Label(self.parameter_frame, text=spec.label).grid(row=base_row, column=0, sticky="w")
            value_label_var = tk.StringVar(value=spec.format_value(state[spec.key]))
            ttk.Label(self.parameter_frame, textvariable=value_label_var, width=10).grid(
                row=base_row,
                column=1,
                sticky="e",
            )

            variable: tk.Variable
            if spec.value_type == "int":
                variable = tk.IntVar(value=int(state[spec.key]))
            else:
                variable = tk.DoubleVar(value=float(state[spec.key]))

            scale = tk.Scale(
                self.parameter_frame,
                from_=spec.minimum,
                to=spec.maximum,
                resolution=spec.resolution,
                orient=tk.HORIZONTAL,
                length=240,
                variable=variable,
                showvalue=False,
                highlightthickness=0,
            )
            scale.grid(row=base_row + 1, column=0, sticky="ew", padx=(0, 10))

            spin = ttk.Spinbox(
                self.parameter_frame,
                from_=spec.minimum,
                to=spec.maximum,
                increment=spec.resolution,
                textvariable=variable,
                width=10,
            )
            spin.grid(row=base_row + 1, column=1, sticky="e")

            info = f"doc: {spec.documented_range} | current: {spec.current_range}"
            ttk.Label(self.parameter_frame, text=info, foreground="#667075", wraplength=340, justify="left").grid(
                row=base_row + 2,
                column=0,
                columnspan=2,
                sticky="w",
                pady=(2, 8),
            )

            self.active_param_vars[spec.key] = variable
            self.active_value_labels[spec.key] = value_label_var
            variable.trace_add("write", self._make_parameter_trace(mode.key, spec, variable, value_label_var))

        self.parameter_frame.columnconfigure(0, weight=1)
        self._update_mode_notes()

    def _make_parameter_trace(
        self,
        mode_key: str,
        spec,
        variable: tk.Variable,
        value_label_var: tk.StringVar,
    ):
        def _callback(*_args) -> None:
            try:
                raw_value = variable.get()
            except tk.TclError:
                return
            clamped = spec.clamp(raw_value)
            self.mode_states[mode_key][spec.key] = clamped
            value_label_var.set(spec.format_value(clamped))
            self.schedule_refresh()

        return _callback

    def _current_params(self, mode_key: str) -> dict[str, float | int]:
        mode = get_mode_spec(mode_key)
        return {
            spec.key: spec.clamp(self.mode_states[mode_key].get(spec.key, spec.default))
            for spec in mode.parameters
        }

    def _update_mode_notes(self) -> None:
        mode = get_mode_spec(self.mode_var.get())
        params = self._current_params(mode.key)
        self.status_var.set(mode.pipeline_note)
        lines = [mode.description, ""]
        for spec in mode.parameters:
            lines.append(f"{spec.label}: {spec.description}")
            lines.append(f"current value: {spec.format_value(params[spec.key])}")
            lines.append(f"documented range: {spec.documented_range}")
            lines.append(f"current sampler range: {spec.current_range}")
            lines.append("")
        self.info_var.set("\n".join(lines).strip())

    def schedule_refresh(self) -> None:
        if self._refresh_after_id is not None:
            self.root.after_cancel(self._refresh_after_id)
        self._refresh_after_id = self.root.after(60, self.refresh_all)

    def refresh_all(self) -> None:
        self._refresh_after_id = None
        mode_key = self.mode_var.get()
        params = self._current_params(mode_key)
        try:
            seed = int(self.seed_var.get())
        except tk.TclError:
            seed = 0
            self.seed_var.set(seed)

        self.last_output = apply_noise(mode_key, self.original_image, seed=seed, params=params)
        comparison = build_comparison_panel(self.original_image, self.last_output, mode_key, params, seed)
        self.preview_photo_ref = ImageTk.PhotoImage(comparison)
        self.preview_label.configure(image=self.preview_photo_ref)

        self._update_mode_notes()
        self._refresh_cards(seed)

    def _refresh_cards(self, seed: int) -> None:
        original_card = make_card(self.thumbnail_source, title="Original", subtitle="reference", active=False)
        self.thumbnail_photo_refs["__original__"] = ImageTk.PhotoImage(original_card)
        self.card_labels["__original__"].configure(image=self.thumbnail_photo_refs["__original__"])

        selected = self.mode_var.get()
        for mode in self.mode_specs:
            params = self._current_params(mode.key)
            preview = apply_noise(mode.key, self.thumbnail_source, seed=seed, params=params)
            subtitle = "used" if "Used by publish_noised_rgb" in mode.pipeline_note else "not used"
            card = make_card(preview, title=mode.title, subtitle=subtitle, active=mode.key == selected)
            self.thumbnail_photo_refs[mode.key] = ImageTk.PhotoImage(card)
            self.card_labels[mode.key].configure(image=self.thumbnail_photo_refs[mode.key])

    def _save_snapshot(self) -> None:
        if self.last_output is None:
            return
        mode_key = self.mode_var.get()
        params = self._current_params(mode_key)
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{timestamp}_{mode_key}.png"
        output_path = SNAPSHOT_DIR / file_name
        Image.fromarray(self.last_output).save(output_path)
        messagebox.showinfo(
            title="Snapshot Saved",
            message=f"Saved to:\n{output_path}\n\n{format_param_summary(mode_key, params)}",
        )


def run_smoke_test(image_path: Path, seed: int) -> None:
    image = load_rgb_image(image_path)
    for mode in list_mode_specs():
        params = default_parameters(mode.key)
        output = apply_noise(mode.key, image, seed=seed, params=params)
        print(
            f"{mode.key}: shape={output.shape} dtype={output.dtype} "
            f"min={int(output.min())} max={int(output.max())} params={format_param_summary(mode.key, params)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive explorer for camera_noiser.py noise modes.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="Reference image for the preview UI.")
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed.")
    parser.add_argument(
        "--export-grid",
        type=Path,
        default=None,
        help="Save a default-parameter contact sheet and exit.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run all modes once on the input image and print summary stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if args.smoke_test:
        run_smoke_test(image_path, seed=args.seed)
        return

    if args.export_grid is not None:
        image = load_rgb_image(image_path)
        grid = build_export_grid(image, seed=args.seed)
        output_path = args.export_grid.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(output_path)
        print(output_path)
        return

    root = tk.Tk()
    app = NoiseExplorerApp(root, image_path=image_path, seed=args.seed)
    root.noise_explorer_app = app
    root.mainloop()


if __name__ == "__main__":
    main()
