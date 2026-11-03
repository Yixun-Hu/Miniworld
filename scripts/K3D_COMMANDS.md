# K3D generation

Run from the repository root. All commands use CPU Mesa under Xvfb. The EGL
headless path does not work on this machine. Set a seed `S` for each clip; the
same seed reproduces the block count, initial state, policy and trajectory.

```bash
unset MINIWORLD_HEADLESS PYGLET_HEADLESS
export LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=3.3 MESA_GLSL_VERSION_OVERRIDE=330 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 LP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
S=1
mkdir -p out/k3d-int out/k3d-v out/k3d-a out/k3d-smoke
```

These are the released 60-step, 128×128 settings. Only the executable/output
paths and the specified dynamics, depth and seed flags differ. K3D-int keeps
the original grid code path; `--k3d-integer` is an additional law validation mode,
not the K3D-int data split. The released command already includes
`--store-block-info`, so it appears once below.

## K3D-int

```bash
xvfb-run -a -s "-screen 0 640x480x24" /home/yixunhu/miniconda3/envs/k3d/bin/python -m scripts.generate_videos --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 --steps 60 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.92 --cam-fov-y 60 --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette \
  --store-block-info --out-prefix out/k3d-int/0000 --no-depth --seed "$S"
```

## K3D-v

```bash
xvfb-run -a -s "-screen 0 640x480x24" /home/yixunhu/miniconda3/envs/k3d/bin/python -m scripts.generate_videos --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero --k3d-mode --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 --steps 60 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 --agent-box-allow-overlap --box-allow-overlap \
  --policy biased_walk_v2 --forward-prob 0.92 --cam-fov-y 60 --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette \
  --store-block-info --out-prefix out/k3d-v/0000 --no-depth --seed "$S"
```

## K3D-a

```bash
xvfb-run -a -s "-screen 0 640x480x24" /home/yixunhu/miniconda3/envs/k3d/bin/python -m scripts.generate_videos --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero --k3d-mode --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 --steps 60 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 --agent-box-allow-overlap --box-allow-overlap \
  --policy biased_walk_v2 --forward-prob 0.92 --cam-fov-y 60 --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette \
  --store-block-info --out-prefix out/k3d-a/0000 --no-depth --seed "$S" \
  --k3d-accel-min 2e-3 --k3d-accel-max 1e-2
```

## One-clip smoke test

```bash
xvfb-run -a -s "-screen 0 640x480x24" /home/yixunhu/miniconda3/envs/k3d/bin/python -m scripts.generate_videos --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero --k3d-mode --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 --steps 20 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 --agent-box-allow-overlap --box-allow-overlap \
  --policy biased_walk_v2 --forward-prob 0.92 --cam-fov-y 60 --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette \
  --store-block-info --out-prefix out/k3d-smoke/0000 --no-depth --seed 1
```

## Batch launcher

Place `--seed "$S"` before the separator to assign seed `S + index` to each clip.
Place all generation options, including the k3d flags, `--no-depth` and
`--store-block-info`, after `--`; they pass through verbatim. A seed after `--`
is also passed verbatim and therefore intentionally repeats the same clip.
Use at most two workers with the thread settings above (at most eight compute
threads across the two worker processes, including Mesa).

```bash
xvfb-run -a -s "-screen 0 640x480x24" /home/yixunhu/miniconda3/envs/k3d/bin/python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 --dataset-root out/k3d-a-batch \
  --num-videos 10 --block-size 100 --num-processes 2 --seed "$S" -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero --k3d-mode --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 --steps 60 --output-2d-map --room-size 16 \
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
and the last completed step's bounce flags. Static blocks have zero outgoing
metadata velocity, while their sampled state remains available on the env.

Acceleration defaults to disabled. Set both acceleration bounds for K3D-a;
when only a positive maximum is supplied, a zero minimum selects `0.002`.
Integer mode always forces acceleration to zero. `vmax` controls the specified
initial rejection envelope; the dynamics never clamp velocity.

The specified kick-drift-mirror law does **not** satisfy the requested universal
energy-error bound `|ΔE| ≤ 1.5 |a||v|`. At `p=(15,8)`, `v=(1,0)`, `a=(.01,0)`,
it gives `ΔE=.02015`, exceeding `.015` (also exceeding `.01515` when using
post-kick speed). Tests preserve this counterexample as a strict expected
failure and verify the exact identity
`ΔE = -|a|²/2 + a·(candidate_position - reflected_position)` over 10,000 steps.

## Validation and 500-step timing

```bash
xvfb-run -a -s "-screen 0 640x480x24" /home/yixunhu/miniconda3/envs/k3d/bin/python -m pytest -q tests/test_k3d.py --basetemp=out/pytest-k3d --durations=10
xvfb-run -a -s "-screen 0 640x480x24" /home/yixunhu/miniconda3/envs/k3d/bin/python -m pytest -q tests/test_miniworld.py --basetemp=out/pytest-existing --durations=10
```

For the depth comparison, run the K3D-int command twice with `--steps 500`
and `--seed 1`, with distinct output prefixes. Remove `--no-depth` for the
first clip and retain it for the second. Prefix each entire `xvfb-run` command
with `/usr/bin/time -p` to include process startup and encoding in wall time.
