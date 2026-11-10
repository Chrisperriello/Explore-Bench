# Research plan after the Level-0 results (2026-11-09)

## Where the results stand

With the uint8 fix (59a7235), Cost and MMPF beat MAPPO in all four settings
(360° and four-beam sensors, belief and legacy routing). Random and wait
never succeed. Four-beam legacy Cost/MMPF are a lower bound: with legacy
routing a robot can step diagonally through unseen corner cells that four
beams never sense, so Cost can shuttle between two diagonal frontiers for
the rest of an episode (seen on familiar__room_1__seed_20001, robot 1).

The weakest point of every result is the test set: "unseen" is 2 door-free
loop maps, 20 episodes per planner.

## Order of work

1. **Frozen map sets** (detailed plan: [map-sets-plan.md](map-sets-plan.md)).
   Saved generated and HouseExpo maps with their own train/validation/test
   splits and computed difficulty tags. Everything below trains or tests on
   these.
2. **Small-map training.** Generated small layouts padded into the
   250 x 250 grid (outside = 205), so the network is unchanged. Time one
   small and one full episode before relying on the speedup. Compare
   small-then-full training against full-only training.
3. **DQN.** Decide the action space first: DQN needs discrete actions, and
   MAPPO outputs coordinates (docs/mappo/decisions.md M28). Proposal: pick
   one of the K nearest belief frontiers, close to what Cost does. Plug in
   through the controller protocol (explore_bench/controllers.py) with its
   own learner and replay buffer; the evaluator and scoring are shared.
4. **Longer MAPPO training.** The best validation checkpoint for seed 4 was
   its last (1,024,000 steps) and seed 3's was 921,600, so learning may not
   have plateaued. One longer 360° belief run answers this; it can queue on
   Ada alongside steps 2-3.

5. **Communication.** Level-0 already supports the `perfect`, `none`,
   `parallel` and `shared_collision` modes
   (onpolicy/onpolicy/envs/GridEnv/communication.py). Once the map sets and
   DQN exist, run the same methods under at least `perfect` and `none`.
6. **Replay and presentation.** Full evaluation traces already store, per
   robot and step, its cell, heading, goal, executed path and sensor reading
   (explore_bench/recording.py), so replays can be drawn from existing
   results without rerunning anything.

## Deliverables (2026-11-10)

Every arm is run with both sensors, 360° and four-beam, under belief
routing. Legacy routing stays a reference only.

| Axis | Values |
|---|---|
| Method | Cost, MMPF, MAPPO, DQN, random, wait |
| Sensor | 360°, four-beam |
| Communication | perfect, none (more modes if time allows) |
| Test maps | original 2 unseen maps, first frozen set (~30 maps) |

The full grid is large; MMPF evaluation (about 10 steps/min) and DQN
training set the pace, so order runs by what the thesis needs first.

## Keeping Gazebo in mind

The original paper's Level-1 is a ROS/Gazebo simulator, and its packages
are already in this repo: six Gazebo worlds in `sim_env/worlds/` matching
the original maps, and launch files for Cost, MMPF, RRT and DRL with one or
two robots in `exploration_benchmark/launch/`. Whether they still run has
not been checked. To keep a port possible:

- Controllers keep one interface: belief maps and robot poses in, goal cells
  out (explore_bench/controllers.py). A ROS node can then feed the same
  controller a SLAM map (gmapping's `/map`) and send its goals to the
  navigation stack. No controller may read grid-only internals.
- Map sets record resolution and occupancy values so each `.pgm` can be
  extruded into a Gazebo `.world` (walls as boxes).
- MAPPO and DQN expect 250 x 250 belief maps at 0.1 m. A Gazebo map has a
  different size and origin, so a port needs a resampling adapter; keep
  observation building in one place so that adapter has one target.
- Replays and metrics are computed from recorded traces, not from simulator
  state, so the same tools can later read traces logged from Gazebo.

## Open questions

- Which setting is the main thesis setting (360° belief so far)?
- How many training seeds per new arm, given Ada time?
- Which ROS version the Level-1 packages target, and whether Ada or a local
  machine can run Gazebo.
