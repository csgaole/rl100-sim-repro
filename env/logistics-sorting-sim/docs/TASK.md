# Task contract v1

## Dynamics and controller

- MuJoCo 3.13.0; dt=1/240s; implicitfast integrator; 100 solver iterations; gravity (0,0,-9.81).
- Two 6-DoF PiPER arms and two fixed-opening grippers; gain/bias for arm drives 1600/-1600.
- Absolute joint target -> joint range clipped with .005rad margin -> command speed clipped to ±3rad/s -> position actuator.
- Requested actions and effective targets differ when clipped. `info.effective_actions` records executed targets.
- No explicit acceleration cap. No force injection or parcel teleportation after reset.
- Parcel starts label down on the transfer table. Seed changes initial x/y uniformly by ±.005m. SKU09: .25×.16×.07m, .22kg. SKU02: .24×.14×.14m, .18kg.
- Infeed and outfeed moving-contact surfaces are in the scene; this benchmark starts with one parcel on the transfer table.

## Termination (checked after EVERY physical tick)

Drop takes precedence if parcel lowest point z < .74m.

Success requires all of these continuously for at least 48 ticks:
1. parcel local +Z alignment with world +Z > .98;
2. contact with outfeed with normal force > .01;
3. parcel bounding extents fully inside outfeed x[-.325,.325], y[-1.83,-.33];
4. maximum per-arm accumulated gripper contact normal force < .05;
5. world y velocity strictly between -.6 and -.1m/s.

Additionally: center y displacement since the qualified streak began >= .10m AND trailing edge y <= -.65m. A broken qualification resets the streak and start position.

If neither terminal outcome occurs by 10 simulated seconds, watchdog truncates. Partial action chunks stop immediately and report actual executed ticks. A policy must decide whether to bootstrap time-limit truncation according to its learning objective; do not silently treat benchmark watchdog failure as an unlimited continuing task.

## Observation and timing

RGBs share a simulation timestamp; forward kinematics is refreshed after integration. Policy observations contain no object truth. Rendering is optional; empty RGB dictionary in physics-only mode.

`policy_observation()` stores two observations only when called. Call once after reset and once per policy step, not several times per step. It resizes the original cameras with Pillow to96×96 and repeats the initial frame. Inputs remain uint8 CHW; divide by255 only in adapters that expect that scale.

For π0.5 use an explicit adapter for camera keys, state normalization, action ordering and padding. Do not apply ALOHA-specific transforms blindly. An action horizon of50 does not authorize50 ticks of open-loop execution when comparing to the8-tick benchmark. Measure inference wall time separately from simulated time.

## Evaluation

Use disjoint training/development/final source seeds. Reuse the same final seeds across frozen candidates. Report success/drop/watchdog including all failures, paired wins/losses and uncertainty. Repeated testing of identical weights is not extra independent evidence. Environment packaging does not supply a trained controller.
