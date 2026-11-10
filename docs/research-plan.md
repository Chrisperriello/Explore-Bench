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

## Open questions

- Which setting is the main thesis setting (360° belief so far)?
- How many training seeds per new arm, given Ada time?
