# Panda joint limits used by the closed-loop fixture

Position and torque come from MuJoCo Menagerie `franka_emika_panda/panda_nohand.xml` at
`4d038b3feae26ec82b46a4d586379114012a8ac7` (2026-09-29). The compiler angle is radian.
`nq = nv = nu = 7` for the seven hinge joints. The class default joint range is `-2.8973 2.8973`
and the class default `forcerange` is `-87 87`. `joint2`, `joint4`, and `joint6` override the
range. `actuator5`, `actuator6`, and `actuator7` override `forcerange` to `-12 12`. The file has
no velocity attribute. Those numbers are copied into `tools/fixtures/panda_nohand_4d038b3f.txt`.
`tau` from `SpikeReadout` is in N·m and is clamped to that `forcerange`, because
`robot_contact::step` does not saturate.

The observation order for a later 14-channel encoder is `qpos[0..7]` in radians, then
`qvel[7..14]` in rad/s, joint1 through joint7. The position half of that clamp is the joint
range above.

Velocity is not in the MJCF. The figure 2.1750 rad/s for joints 1–4 and 2.6100 rad/s for joints
5–7 was checked against the Franka Emika Panda datasheet, May 2019,
`https://static.generation-robots.com/media/Panda_Datasheet_May_2019.pdf`
(sha256 `7f372237f576fecc3a95ca407f3eb78ddc7f7b213d0b055b3fde1cc0297181fb`).
That sheet's joint velocity limits are `A1, A2, A3, A4: 150` °/s and `A5, A6, A7: 180` °/s.
They are not 2.1750 rad/s and not 2.6100 rad/s. The fixture therefore does not clamp velocity.
No converted rad/s value was substituted.
