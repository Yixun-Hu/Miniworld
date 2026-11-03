# K3D generation

Run from the repository root. All commands use CPU Mesa under Xvfb. The EGL
headless path does not work on this machine. Set a seed `S` for each clip; the
same seed reproduces the block count, initial state, policy and trajectory.

```bash
unset MINIWORLD_HEADLESS PYGLET_HEADLESS
export LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=3.3 MESA_GLSL_VERSION_OVERRIDE=330 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S=1
mkdir -p out/k3d-int out/k3d-v out/k3d-a out/k3d-smoke
```

Generate all three splits from the same HEAD revision of this generator with
500 steps, 128×128 frames and the same encoder settings shown below. Regenerate
K3D-int too: the older base-generator split ignored `--seed` and cannot serve
as a reproducible seeded split. K3D-int keeps the original grid code path;
`--k3d-integer` is an additional law validation mode, not the K3D-int data split.

Every command explicitly selects `--ffmpeg-threads 1` to limit encoder/filter
threads, with one batch worker and one thread per numerical/Mesa pool. Without
this option the writer passes no `ffmpeg_params`, preserving the base writer's
default bytes on the same system. Thread settings can change encoded pixels,
so use the same option for K3D-int, K3D-v and K3D-a.

## K3D-int (HEAD, grid mode)

```bash
xvfb-run -a -s "-screen 0 640x480x24" /home/yixunhu/miniconda3/envs/k3d/bin/python -m scripts.generate_videos --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 --steps 500 --output-2d-map --room-size 16 --ffmpeg-threads 1 \
  --block-size-xy 0.7 --block-height 1.5 --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.92 --cam-fov-y 60 --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette \
  --store-block-info --out-prefix out/k3d-int/0000 --no-depth --seed "$S"
```

## K3D-v

```bash
xvfb-run -a -s "-screen 0 640x480x24" /home/yixunhu/miniconda3/envs/k3d/bin/python -m scripts.generate_videos --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero --k3d-mode --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 --steps 500 --output-2d-map --room-size 16 --ffmpeg-threads 1 \
  --block-size-xy 0.7 --block-height 1.5 --agent-box-allow-overlap --box-allow-overlap \
  --policy biased_walk_v2 --forward-prob 0.92 --cam-fov-y 60 --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette \
  --store-block-info --out-prefix out/k3d-v/0000 --no-depth --seed "$S"
```

## K3D-a

```bash
xvfb-run -a -s "-screen 0 640x480x24" /home/yixunhu/miniconda3/envs/k3d/bin/python -m scripts.generate_videos --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero --k3d-mode --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 --steps 500 --output-2d-map --room-size 16 --ffmpeg-threads 1 \
  --block-size-xy 0.7 --block-height 1.5 --agent-box-allow-overlap --box-allow-overlap \
  --policy biased_walk_v2 --forward-prob 0.92 --cam-fov-y 60 --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette \
  --store-block-info --out-prefix out/k3d-a/0000 --no-depth --seed "$S" \
  --k3d-accel-min 2e-3 --k3d-accel-max 1e-2
```

## One-clip smoke test

```bash
xvfb-run -a -s "-screen 0 640x480x24" /home/yixunhu/miniconda3/envs/k3d/bin/python -m scripts.generate_videos --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero --k3d-mode --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 --steps 20 --output-2d-map --room-size 16 --ffmpeg-threads 1 \
  --block-size-xy 0.7 --block-height 1.5 --agent-box-allow-overlap --box-allow-overlap \
  --policy biased_walk_v2 --forward-prob 0.92 --cam-fov-y 60 --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette \
  --store-block-info --out-prefix out/k3d-smoke/0000 --no-depth --seed 1
```

## Batch launcher

Place `--seed "$S"` on either side of the separator to assign seed `S + index`
to each clip. The launcher removes a forwarded seed and adds the per-item seed;
if seeds are supplied on both sides, they must agree or the launcher errors.
Place all generation options, including the k3d flags, `--no-depth` and
`--store-block-info`, after `--`; these pass through verbatim.

```bash
xvfb-run -a -s "-screen 0 640x480x24" /home/yixunhu/miniconda3/envs/k3d/bin/python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 --dataset-root out/k3d-a-batch \
  --num-videos 10 --block-size 100 --num-processes 1 --seed "$S" -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero --k3d-mode --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 --steps 500 --output-2d-map --room-size 16 --ffmpeg-threads 1 \
  --block-size-xy 0.7 --block-height 1.5 --agent-box-allow-overlap --box-allow-overlap \
  --policy biased_walk_v2 --forward-prob 0.92 --cam-fov-y 60 --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette \
  --store-block-info --no-depth --k3d-accel-min 2e-3 --k3d-accel-max 1e-2
```

## Metadata and law conventions

`_actions.pt` retains its original keys. All block arrays are in the separate
`_block_info.pt`, which also records `seed` and `num_blocks` for all modes.
`block_pos_world` keeps XYZ shape `(T,N,3)`; its XZ coordinates come directly
from the continuous k3d state. No rounding to grid cells occurs.

Frame `t` is the state after `t` steps. `block_vel_k3d[t]` is the outgoing
velocity **after the acceleration kick and before reflection** for the
transition `t → t+1`. `block_bounce[t]` describes that same transition, and
`block_speed[t]` is the norm of that velocity. `block_acc_world[t]` is the fixed
world-frame acceleration at frame `t`. All arrays have length `T`, including
the last transition whose destination frame is not saved. Float32 storage
means displacement equalities hold to float32 rounding tolerance.
`env.k3d_block_state()` instead returns the current post-reflection velocity
and the last completed step's bounce flags. Static blocks have fixed positions
and zero velocity and acceleration in both environment state and metadata.

Reset and step return the base call's observation, as in grid mode; the generator
captures the updated block state through `env.render()`. Existing agent details
in k3d mode remain: heading is also quantized at reset, blocks still update on a
terminating/truncating step, and carried blocks are included in the block law.
The split commands use `--heading-zero`, `--no-time-limit` and a policy without
pickup actions, so these differences from grid mode do not affect them.

Acceleration defaults to disabled. Set both acceleration bounds for K3D-a;
when only a positive maximum is supplied, a zero minimum selects
`min(0.002, accel_max)`, including maxima below `0.002`.
Integer mode always forces acceleration to zero.

The rejection envelope is `sqrt(|v|² + 2|a|·(hi−lo)) ≤ vmax`, using
`|a|·(hi−lo)` as the potential drop. The exact worst-case potential drop over
the square is `(|a_x|+|a_z|)·(hi−lo)`, giving a speed envelope of about **1.18**
at the defaults (before discrete bounce errors). Neither envelope rejects
pairs at the K3D-a defaults. `vmax` is **not a hard bound** and velocities are
never clamped: the review measured maximum outgoing speeds over 4000 blocks
of **1.148 / 1.209 / 1.249** at **500 / 5000 / 50000** steps, respectively.

The conserved quantity of the discrete law on free steps is
`|v + a/2|²/2 − a·p`; reflections can change it. For `E = |v|²/2 − a·p`,
the correct per-step bound is `|ΔE| ≤ |a|²/2 + 2|a||v + a|`, using pre-kick
`v`. Tests check this bound, free-step conservation, and the exact identity
`ΔE = -|a|²/2 + a·(candidate_position - reflected_position)` over 10,000 steps.
The old `1.5 |a||v|` bound fails at `p=(15,8)`, `v=(1,0)`, `a=(.01,0)`:
`ΔE=.02015 > .015`; this case now passes against the corrected bound.

## Validation and 500-step timing

```bash
xvfb-run -a -s "-screen 0 640x480x24" /home/yixunhu/miniconda3/envs/k3d/bin/python -m pytest -q tests/test_k3d.py --basetemp=out/pytest-k3d --durations=10
xvfb-run -a -s "-screen 0 640x480x24" /home/yixunhu/miniconda3/envs/k3d/bin/python -m pytest -q tests/test_miniworld.py --basetemp=out/pytest-existing --durations=10
```

For the depth comparison, run the K3D-int command twice with `--steps 500`
and `--seed 1`, with distinct output prefixes. Remove `--no-depth` for the
first clip and retain it for the second. Prefix each entire `xvfb-run` command
with `/usr/bin/time -p` to include process startup and encoding in wall time.
