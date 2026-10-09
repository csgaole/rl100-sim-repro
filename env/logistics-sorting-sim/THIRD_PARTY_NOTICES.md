# Third-party robot resources

The robot meshes are byte-for-byte verified against AgileX's
[agx_arm_urdf](https://github.com/agilexrobotics/agx_arm_urdf) revision
`f6642ce0d7872c686f29c99e9e10cd23d1d49313`.

The upstream MIT license is retained in `third_party/agx_arm_urdf-LICENSE`.
See `third_party/robot_asset_manifest.json` for source paths and SHA256 hashes.
The robot geometry/inertial/joint definitions in the bundled scenes are derived
from that robot description, converted to MuJoCo and adapted into a dual-arm cell.
Local changes include mounting, colors, collision setup, actuators, cameras,
parcel/table/conveyor geometry and task evaluation. Mesh bytes are unchanged.

MuJoCo, NumPy and Pillow are installed as dependencies, not vendored here.
