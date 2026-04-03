from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


ImageArray = np.ndarray
ParameterValues = dict[str, float | int]
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_COLLECTION_ROOT = REPO_ROOT / "source" / "data_collection"
DEFAULT_WARP_CACHE_PATH = "/tmp/genie_sim_warp_cache"

os.environ.setdefault("WARP_CACHE_PATH", DEFAULT_WARP_CACHE_PATH)

if str(DATA_COLLECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_ROOT))

try:
    from server.ros_publisher.camera_noiser import apply_noise_to_image as official_apply_noise_to_image
    from server.ros_publisher.camera_noiser import get_random_parameters as official_get_random_parameters

    OFFICIAL_CAMERA_NOISER_AVAILABLE = True
except Exception:
    official_apply_noise_to_image = None
    official_get_random_parameters = None
    OFFICIAL_CAMERA_NOISER_AVAILABLE = False


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    minimum: float
    maximum: float
    resolution: float
    default: float | int
    value_type: str = "float"
    display_format: str = ".3f"
    documented_range: str = ""
    current_range: str = ""
    description: str = ""

    def clamp(self, value: float | int) -> float | int:
        if self.value_type == "int":
            return int(min(max(round(float(value)), self.minimum), self.maximum))
        return float(min(max(float(value), self.minimum), self.maximum))

    def format_value(self, value: float | int) -> str:
        if self.value_type == "int":
            return str(int(round(float(value))))
        return format(float(value), self.display_format)


@dataclass(frozen=True)
class NoiseModeSpec:
    key: str
    title: str
    description: str
    pipeline_note: str
    parameters: tuple[ParameterSpec, ...]
    apply_fn: Callable[..., ImageArray]
    current_sampler: Callable[[np.random.Generator], ParameterValues] | None = None
    supports_official_apply: bool = False

    def default_parameters(self) -> ParameterValues:
        return {spec.key: spec.clamp(spec.default) for spec in self.parameters}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def _to_uint8_image(image: ImageArray) -> ImageArray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def _truncated_absolute_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    lower: float,
    upper: float,
    max_attempts: int = 10000,
) -> float:
    for _ in range(max_attempts):
        sample = abs(float(rng.normal(loc=mean, scale=std)))
        if lower <= sample <= upper:
            return sample
    return float(abs(rng.uniform(lower, upper)))


def _with_numpy_seed(seed: int | None, fn: Callable[..., ParameterValues], *args) -> ParameterValues:
    if seed is None:
        return fn(*args)

    state = np.random.get_state()
    np.random.seed(int(seed) & 0xFFFFFFFF)
    try:
        return fn(*args)
    finally:
        np.random.set_state(state)


def gaussian_noise(image: ImageArray, seed: int, sigma: float = 0.1) -> ImageArray:
    rng = _rng(seed)
    result = image.astype(np.float32) + (255.0 * float(sigma) * rng.normal(size=image.shape))
    return result.astype(np.uint8)


def salt_pepper_noise(
    image: ImageArray,
    seed: int,
    salt_prob: float = 0.01,
    pepper_prob: float = 0.01,
) -> ImageArray:
    rng = _rng(seed)
    rand_map = rng.random(size=image.shape[:2])
    result = image.copy()
    salt_mask = rand_map < float(salt_prob)
    pepper_mask = (rand_map >= float(salt_prob)) & (rand_map < float(salt_prob) + float(pepper_prob))
    result[salt_mask] = 255
    result[pepper_mask] = 0
    return result


def poisson_noise(image: ImageArray, seed: int, scale: float = 1.0) -> ImageArray:
    rng = _rng(seed)
    normalized = image.astype(np.float32) / 255.0
    variance = np.sqrt(normalized).astype(np.float32)
    noise = rng.normal(size=image.shape).astype(np.float32) * variance * float(scale)
    result = np.clip((normalized + noise) * 255.0, 0.0, 255.0)
    return result.astype(np.uint8)


def speckle_noise(image: ImageArray, seed: int, sigma: float = 0.1) -> ImageArray:
    rng = _rng(seed)
    factors = 1.0 + float(sigma) * rng.normal(size=image.shape).astype(np.float32)
    result = np.clip(image.astype(np.float32) * factors, 0.0, 255.0)
    return result.astype(np.uint8)


def quantization_noise(image: ImageArray, seed: int, bits: int = 4) -> ImageArray:
    del seed
    levels = float(2 ** int(bits))
    quant_step = 255.0 / (levels - 1.0)
    result = np.round(image.astype(np.float32) / quant_step) * quant_step
    return result.astype(np.uint8)


def sensor_noise(
    image: ImageArray,
    seed: int,
    shot_noise: float = 0.01,
    read_noise: float = 0.02,
) -> ImageArray:
    rng = _rng(seed)
    normalized = image.astype(np.float32) / 255.0
    signal_term = np.sqrt(normalized).astype(np.float32)
    shot = rng.normal(size=image.shape).astype(np.float32) * signal_term * float(shot_noise)
    read = rng.normal(size=image.shape).astype(np.float32) * float(read_noise)
    result = np.clip((normalized + shot + read) * 255.0, 0.0, 255.0)
    return result.astype(np.uint8)


def brownian_noise(image: ImageArray, seed: int, intensity: float = 0.1) -> ImageArray:
    rng = _rng(seed)
    brownian = np.zeros_like(image, dtype=np.float32)
    for octave in range(4):
        weight = 1.0 / float(2**octave)
        brownian += weight * rng.normal(size=image.shape).astype(np.float32)
    brownian_scale = 1.0 / 1.875
    result = np.clip(
        image.astype(np.float32) + 255.0 * float(intensity) * brownian * brownian_scale,
        0.0,
        255.0,
    )
    return result.astype(np.uint8)


def _sample_gaussian_current(rng: np.random.Generator) -> ParameterValues:
    if OFFICIAL_CAMERA_NOISER_AVAILABLE and official_get_random_parameters is not None:
        return _with_numpy_seed(int(rng.integers(0, 2**32 - 1)), official_get_random_parameters, "gaussian")
    return {
        "sigma": _truncated_absolute_normal(
            rng=rng,
            mean=0.1,
            std=0.08,
            lower=0.1,
            upper=0.15,
        )
    }


def _sample_salt_pepper_current(rng: np.random.Generator) -> ParameterValues:
    if OFFICIAL_CAMERA_NOISER_AVAILABLE and official_get_random_parameters is not None:
        return _with_numpy_seed(int(rng.integers(0, 2**32 - 1)), official_get_random_parameters, "salt_pepper")
    prob = float(rng.uniform(0.002, 0.005))
    return {"salt_prob": prob, "pepper_prob": prob}


def _sample_poisson_current(rng: np.random.Generator) -> ParameterValues:
    if OFFICIAL_CAMERA_NOISER_AVAILABLE and official_get_random_parameters is not None:
        return _with_numpy_seed(int(rng.integers(0, 2**32 - 1)), official_get_random_parameters, "poisson")
    return {"scale": float(rng.uniform(0.05, 0.1))}


def _sample_speckle_current(rng: np.random.Generator) -> ParameterValues:
    if OFFICIAL_CAMERA_NOISER_AVAILABLE and official_get_random_parameters is not None:
        return _with_numpy_seed(int(rng.integers(0, 2**32 - 1)), official_get_random_parameters, "speckle")
    return {"sigma": float(rng.uniform(0.05, 0.1))}


def _sample_quantization_current(rng: np.random.Generator) -> ParameterValues:
    if OFFICIAL_CAMERA_NOISER_AVAILABLE and official_get_random_parameters is not None:
        return _with_numpy_seed(int(rng.integers(0, 2**32 - 1)), official_get_random_parameters, "quantization")
    return {"bits": int(rng.integers(3, 5))}


PIPELINE_ACTIVE = "Used by publish_noised_rgb in the current pipeline."
PIPELINE_INACTIVE = "Defined in camera_noiser.py, but not wired into publish_noised_rgb."


MODE_SPECS: dict[str, NoiseModeSpec] = {
    "gaussian": NoiseModeSpec(
        key="gaussian",
        title="Gaussian",
        description=(
            "Additive white noise. The current Warp kernel casts directly to uint8 without clipping, "
            "so large values wrap around in the same way as the online path."
        ),
        pipeline_note=PIPELINE_ACTIVE,
        parameters=(
            ParameterSpec(
                key="sigma",
                label="sigma",
                minimum=0.0,
                maximum=0.4,
                resolution=0.005,
                default=0.125,
                display_format=".3f",
                documented_range="0.01 - 0.30",
                current_range="0.10 - 0.15 (truncated abs normal around 0.10)",
                description="Standard deviation of additive noise, normalized by 255.",
            ),
        ),
        apply_fn=gaussian_noise,
        current_sampler=_sample_gaussian_current,
        supports_official_apply=True,
    ),
    "salt_pepper": NoiseModeSpec(
        key="salt_pepper",
        title="Salt Pepper",
        description="Random white and black pixels simulating dead pixels or transmission corruption.",
        pipeline_note=PIPELINE_ACTIVE,
        parameters=(
            ParameterSpec(
                key="salt_prob",
                label="salt_prob",
                minimum=0.0,
                maximum=0.05,
                resolution=0.0005,
                default=0.0035,
                display_format=".4f",
                documented_range="0.001 - 0.050",
                current_range="0.002 - 0.005",
                description="Probability of replacing a pixel with white.",
            ),
            ParameterSpec(
                key="pepper_prob",
                label="pepper_prob",
                minimum=0.0,
                maximum=0.05,
                resolution=0.0005,
                default=0.0035,
                display_format=".4f",
                documented_range="0.001 - 0.050",
                current_range="0.002 - 0.005",
                description="Probability of replacing a pixel with black.",
            ),
        ),
        apply_fn=salt_pepper_noise,
        current_sampler=_sample_salt_pepper_current,
        supports_official_apply=True,
    ),
    "poisson": NoiseModeSpec(
        key="poisson",
        title="Poisson",
        description="Signal-dependent shot noise approximation. Darker regions are less affected than bright ones.",
        pipeline_note=PIPELINE_ACTIVE,
        parameters=(
            ParameterSpec(
                key="scale",
                label="scale",
                minimum=0.0,
                maximum=3.0,
                resolution=0.01,
                default=0.075,
                display_format=".3f",
                documented_range="0.50 - 3.00",
                current_range="0.05 - 0.10",
                description="Multiplier on the sqrt(signal) noise term.",
            ),
        ),
        apply_fn=poisson_noise,
        current_sampler=_sample_poisson_current,
        supports_official_apply=True,
    ),
    "speckle": NoiseModeSpec(
        key="speckle",
        title="Speckle",
        description="Multiplicative noise. Bright regions expand and contract with the signal.",
        pipeline_note=PIPELINE_ACTIVE,
        parameters=(
            ParameterSpec(
                key="sigma",
                label="sigma",
                minimum=0.0,
                maximum=0.3,
                resolution=0.005,
                default=0.075,
                display_format=".3f",
                documented_range="0.05 - 0.30",
                current_range="0.05 - 0.10",
                description="Standard deviation of the multiplicative factor.",
            ),
        ),
        apply_fn=speckle_noise,
        current_sampler=_sample_speckle_current,
        supports_official_apply=True,
    ),
    "quantization": NoiseModeSpec(
        key="quantization",
        title="Quantization",
        description="Reduced bit depth causing banding and stepped color transitions.",
        pipeline_note=PIPELINE_ACTIVE,
        parameters=(
            ParameterSpec(
                key="bits",
                label="bits",
                minimum=2,
                maximum=8,
                resolution=1,
                default=4,
                value_type="int",
                documented_range="2 - 7",
                current_range="3 - 4",
                description="Effective output bit depth after quantization.",
            ),
        ),
        apply_fn=quantization_noise,
        current_sampler=_sample_quantization_current,
        supports_official_apply=True,
    ),
    "sensor_noise": NoiseModeSpec(
        key="sensor_noise",
        title="Sensor Noise",
        description="Combined shot noise plus read noise physical model.",
        pipeline_note=PIPELINE_INACTIVE,
        parameters=(
            ParameterSpec(
                key="shot_noise",
                label="shot_noise",
                minimum=0.0,
                maximum=0.1,
                resolution=0.001,
                default=0.03,
                display_format=".3f",
                documented_range="0.01 - 0.05",
                current_range="not sampled in current pipeline",
                description="Signal-dependent shot noise term.",
            ),
            ParameterSpec(
                key="read_noise",
                label="read_noise",
                minimum=0.0,
                maximum=0.1,
                resolution=0.001,
                default=0.0175,
                display_format=".3f",
                documented_range="0.005 - 0.03",
                current_range="not sampled in current pipeline",
                description="Signal-independent sensor readout term.",
            ),
        ),
        apply_fn=sensor_noise,
    ),
    "brownian": NoiseModeSpec(
        key="brownian",
        title="Brownian",
        description="Fractal-style multi-octave noise as implemented in camera_noiser.py.",
        pipeline_note=PIPELINE_INACTIVE,
        parameters=(
            ParameterSpec(
                key="intensity",
                label="intensity",
                minimum=0.0,
                maximum=0.3,
                resolution=0.005,
                default=0.125,
                display_format=".3f",
                documented_range="0.05 - 0.20",
                current_range="not sampled in current pipeline",
                description="Overall Brownian noise amplitude.",
            ),
        ),
        apply_fn=brownian_noise,
    ),
}


def list_mode_specs() -> list[NoiseModeSpec]:
    return list(MODE_SPECS.values())


def get_mode_spec(mode_key: str) -> NoiseModeSpec:
    return MODE_SPECS[mode_key]


def default_parameters(mode_key: str) -> ParameterValues:
    return get_mode_spec(mode_key).default_parameters()


def sample_current_parameters(mode_key: str, seed: int | None = None) -> ParameterValues:
    mode = get_mode_spec(mode_key)
    if mode.current_sampler is None:
        return mode.default_parameters()
    rng = np.random.default_rng(seed)
    sampled = mode.current_sampler(rng)
    params = mode.default_parameters()
    for key, value in sampled.items():
        params[key] = value
    return params


def apply_noise(mode_key: str, image: ImageArray, seed: int, params: ParameterValues | None = None) -> ImageArray:
    mode = get_mode_spec(mode_key)
    values = mode.default_parameters()
    if params:
        for spec in mode.parameters:
            if spec.key in params:
                values[spec.key] = spec.clamp(params[spec.key])
    image_uint8 = _to_uint8_image(image)
    if mode.supports_official_apply and OFFICIAL_CAMERA_NOISER_AVAILABLE and official_apply_noise_to_image is not None:
        return _to_uint8_image(official_apply_noise_to_image(image_uint8, noise_type=mode_key, seed=int(seed), **values))
    return _to_uint8_image(mode.apply_fn(image_uint8, int(seed), **values))
