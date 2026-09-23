# Interchangeable Level-0 Sensors

## Implementation Order

1. Capture golden tests for the current 360 sensor before moving its code.
2. Extract the sensor interfaces and unchanged 360 implementation.
3. Implement the four-beam sensor and occupancy projector.
4. Integrate per-agent selection into both Level-0 environments.
5. Add configuration, documentation, and complete verification.

## Sensor Interfaces

- Add a shared Level-0 sensor module containing sensor configuration, measurement, projection, and result interfaces.
- Support `omnidirectional` and `four_beam` sensor selectors.
- Add optional per-agent sensor configurations to `GridEnv` while preserving the existing constructor order and `sensor_range` behavior.
- Broadcast a single configuration to all agents, or require exactly one configuration per agent.
- Accept positive finite ranges and positive infinity; reject unknown types, nonpositive values, NaN, and incorrect configuration counts.

## Sensor Behavior

- Move the existing optimized 360 implementation without changing its finite-range square boundary, occlusion, `3x3` reveal, or `0/205/254` encoding.
- Treat an infinite 360 range as the complete in-bounds map.
- Add one-cell-wide front, back, left, and right rays using the robot's exact yaw.
- Use deterministic grid traversal for cardinal, diagonal, and intermediate headings.
- Reveal the robot cell, free cells along each ray, and the first occupied cell; leave cells behind obstacles unknown.
- Report obstacle distance in meters from the robot-cell center to the occupied-cell boundary.
- Return the configured finite range for a finite miss and infinity for an infinite miss, with `hit=false` in either case.

## Environment Integration

- Give every agent its own sensor and route all Level-0 sensing through one environment helper.
- Store exact heading as normalized `agent_yaw`, use path yaw for scans, and retain the existing cardinal `agent_direction` policy input.
- Preserve existing map merging, frontier detection, rewards, exploration metrics, observation shapes, and path sampling frequency.
- Expose the latest per-agent sensor measurements in `info["sensor_readings"]`.
- Add per-agent sensor type and range options to training and standalone evaluation while retaining the existing 360 defaults.
- Keep one canonical implementation shared by both Level-0 environments.

## Level-1 Preparation

- Keep raw measurement generation separate from occupancy projection.
- Keep measurements independent of Gym, policies, rewards, and frontiers so Level-1 can later provide external readings.
- Leave ROS/Gazebo adapters, noise, latency, beam width, and policy observation changes out of this implementation.

## Verification

- Require byte-for-byte compatibility for representative finite-range 360 scans.
- Test four-beam rotation, occlusion, exact-range hits, misses, unknown cells, boundaries, and infinite range.
- Test per-agent selection, broadcasting, mixed configurations, validation, heading updates, map merging, and frontier detection.
- Smoke-test training and standalone `cost`/`mmpf` entry points with both sensor types.

## Acceptance Criteria

- Existing commands retain their 360 observation behavior by default.
- Each agent can independently select either supported sensor and range.
- Four-beam readings rotate with actual robot motion and expose correct distances and hit flags.
- Four-beam maps use the existing frontier pipeline without sensor-specific frontier logic.
- Infinite range remains bounded by the map during projection and does not overflow.
