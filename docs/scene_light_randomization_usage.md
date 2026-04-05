# Scene Light Randomization Usage

## Goal

`scene_light_randomization` is used to randomize existing light prims that already live inside the scene USD.

This flow:

- reads the configured `light_prim` from the task json
- samples one `intensity` and one `temperature` value per episode
- reuses the existing `set_light` RPC
- updates the existing light prim on the Isaac stage instead of creating a new light by default

## Task Json Schema

Add `scene_light_randomization` at the top level of the task json:

```json
"scene_light_randomization": [
  {
    "light_prim": "/World/Light/Light_00/DiskLight_02/DiskLight_02",
    "light_type": "Disk",
    "intensity": {
      "min": 40000.0,
      "max": 80000.0
    },
    "temperature": {
      "min": 6500.0,
      "max": 6500.0
    }
  }
]
```

## Required Fields

- `light_prim`
  - Absolute prim path of the existing scene light.
- `light_type`
  - Light type used for validation.
- `intensity`
  - Must be a dict with `min` and `max`.
- `temperature`
  - Must be a dict with `min` and `max`.

If you want a fixed value, set `min` and `max` to the same number.

## light_type Values

Use one of the following values:

- `Disk`
- `Rect`
- `Sphere`
- `Dome`
- `Distant`

The implementation also accepts these aliases:

- `DiskLight`
- `RectLight`
- `SphereLight`
- `DomeLight`
- `DistantLight`

## When Sampling Happens

Sampling happens once per episode when the concrete saved task is loaded.

This means:

- the task json keeps the range configuration
- the actual sampled value is generated at runtime per episode
- a new episode can use a different sampled value

## Example Prim Paths

Example from `home_b`:

```json
{
  "light_prim": "/World/Light/Light_00/DiskLight_02/DiskLight_02",
  "light_type": "Disk",
  "intensity": {
    "min": 40000.0,
    "max": 80000.0
  },
  "temperature": {
    "min": 6500.0,
    "max": 6500.0
  }
}
```

Example from `restaurant_00`:

```json
{
  "light_prim": "/World/Light/RectLight_04",
  "light_type": "Rect",
  "intensity": {
    "min": 8000.0,
    "max": 20000.0
  },
  "temperature": {
    "min": 6500.0,
    "max": 6500.0
  }
}
```

## Notes

- This feature is intended for fixed-scene tasks where the correct `light_prim` is known ahead of time.
- The current implementation updates `inputs:intensity` and `inputs:colorTemperature` on the existing prim.
- If the configured prim path does not exist on the stage, the code falls back to the older light creation behavior used by the existing `set_light` RPC.
