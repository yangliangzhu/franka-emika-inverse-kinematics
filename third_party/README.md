# third_party

Vendored robot description, so the 3D viewers draw a real Panda **offline**.

| | |
|---|---|
| package | `franka_description` |
| upstream | [`frankaemika/franka_ros`](https://github.com/frankaemika/franka_ros), branch `develop`, path `franka_description/` |
| licence | Apache-2.0 (see `franka_description/LICENSE`) |
| contents | the `panda` arm only: `robots/panda/panda.urdf.xacro`, the four `robots/common/*.xacro` it includes, `robots/panda/joint_limits.yaml`, and the eight visual meshes (`meshes/visual/link0..7.dae`) with their collision STLs |
| not included | the hand, the FR3 and other arms, the `_sc` collision-only variants, and `meshes/collision/finger.stl` (referenced by the hand we do not mount) |
| size | ~11 MB, 26 files |

## Why a vendored copy at all

A `franka_description` ships the arm as **xacro**, not URDF: it includes other files
by `$(find franka_description)`, loads YAML and substitutes arguments. `xacro`
itself will not run without a ROS environment, so `franka_ik.swift_viz.expand_xacro`
expands this copy with `xacrodoc` -- already a dependency of `roboticstoolbox` --
and caches the result under `~/.cache/franka_ik`. A checkout needs a `package.xml`
for the `$(find ...)` to resolve, which is why one is added here.

The alternative considered was downloading at run time. That is what
`robot_descriptions` does, and it is why `franka_ik` does not use it: a viewer that
needs the network to draw a robot is a viewer that fails in a lab.

`FRANKA_IK_URDF` overrides all of this with any URDF of your own. It is checked
against `franka_ik.model.fk_tool` before it is drawn, because `franka_description`
ships the Panda *and* the FR3, whose wrists differ by 57 mm, and driving the wrong
one with Panda joint vectors renders a plausible, wrong robot.

## Updating

```bash
cd third_party
BASE=https://raw.githubusercontent.com/frankaemika/franka_ros/develop/franka_description
curl -sSL -o franka_description/robots/panda/panda.urdf.xacro \
  $BASE/robots/panda/panda.urdf.xacro
# ... and the same for whatever else changed, then re-run the Swift tests, which
# pin the mesh count, the placement and the joint limits.
```
