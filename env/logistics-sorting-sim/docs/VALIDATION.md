# Packaging validation — 2026-10-05

- MuJoCo3.13.0, Python3.12, Linux; separate virtual environment.
- Five pytest cases passed: deterministic reset/chunk stepping, exact10s watchdog including partial chunk, success transport/continuity requirements, and both SKU scene/input checks.
- Two SKUs × three seeds replayed against the pre-packaging environment. Exact qpos, qvel and effective actions at every compared tick; exact three-camera RGB at each replay endpoint. Tick counts and scope in [compatibility.json](compatibility.json).
- Replays use historical diagnostic actions solely to check packaging equivalence. They are not a learned-policy evaluation or a claim of task success rate. SKU09 replay comparisons ended when the diagnostic action file ended, before episode termination.
- EGL three-camera render and10-second hold-position demo passed. Preview is actual default global camera, with no brightening.
- Wheel built with all11 robot meshes, both scenes and configuration files included. Installed-wheel smoke tested separately from the source tree.
- No checkpoint, demonstration dataset, private server address, credentials or hardware driver is included.

If a machine has ROS pytest plugins injected through PYTHONPATH, use a clean shell or:
`PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`.

Bitwise equivalence here is within the same machine/software stack; it is not a guarantee across GPU drivers or MuJoCo versions.
