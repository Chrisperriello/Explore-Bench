# Level-0 Sensors

Level-0 supports an interchangeable sensor for each agent. Existing callers use
the `omnidirectional` sensor by default, so their occupancy observations remain
unchanged.

## Sensor Types

- `omnidirectional` preserves the original 360-degree grid scanner.
- `four_beam` casts one-cell-wide front, back, left, and right rays relative to
  the robot's current heading.

Both sensors project observations into the existing occupancy encoding:

- `0`: occupied
- `205`: unknown
- `254`: free

The four-beam sensor also publishes its latest named distances and hit flags in
`info["sensor_readings"]`. A finite miss reports the configured maximum range;
an infinite-range miss reports `inf`.

`info["agent_yaw"]` contains the exact heading in radians used by directional
sensors. The existing four-way integer `info["agent_direction"]` remains
available for compatibility with trained policies.

## Python Configuration

Pass one `SensorConfig` to use it for every agent:

```python
from onpolicy.envs.GridEnv.sensors import SensorConfig

env = GridEnv(
    0.1,
    3.0,
    2,
    100,
    sensor_configs=SensorConfig("four_beam", 4.0),
)
```

Pass one configuration per agent to mix sensor types or ranges:

```python
sensor_configs = [
    SensorConfig("omnidirectional", 3.0),
    SensorConfig("four_beam", float("inf")),
]
```

## Command-Line Configuration

Training accepts one value for every agent or one value per agent:

```bash
python train/train_grid.py \
  --env_name GridEnv \
  --num_agents 2 \
  --sensor_types four_beam \
  --sensor_ranges 4.0
```

Standalone evaluation uses the same options:

```bash
python GridEnv.py cost 2 ../onpolicy/onpolicy/envs/GridEnv/datasets/corner.pgm \
  --sensor_types omnidirectional four_beam \
  --sensor_ranges 3.5 inf
```

The four-beam distances remain metadata rather than policy observation
channels. This preserves compatibility with existing trained policies while
leaving the measurement/projector boundary available for future Level-1
adapters.
