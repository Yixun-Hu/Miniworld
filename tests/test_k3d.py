"""K3D law tests run without GL; rendering tests run under xvfb-run.

Load the pure module directly because miniworld/__init__.py eagerly imports
OpenGL to register the existing environments. No GL imports occur at collection.
"""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Optional

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


law = load_module("k3d_law", ROOT / "miniworld/k3d_dynamics.py")
rendering = pytest.mark.skipif(
    not os.environ.get("DISPLAY"),
    reason="rendering requires a live display; run pytest under xvfb-run",
)


def test_integer_reference_exactness():
    rng = np.random.default_rng(923)
    p, v, a = law.sample_blocks(rng, 5000, integer=True)
    ref_p, ref_v = p.copy(), v.copy()
    for _ in range(60):
        candidate = ref_p + ref_v
        hit = (candidate < 1) | (candidate > 15)
        ref_v = np.where(hit, -ref_v, ref_v)
        ref_p = ref_p + ref_v
        p, v, bounce = law.step_blocks(p, v, a)
        # Bytes compare signed zeros too, beyond array_equal's numeric equality.
        assert p.tobytes() == ref_p.tobytes()
        assert v.tobytes() == ref_v.tobytes()
        np.testing.assert_array_equal(bounce, hit)


def test_continuous_mirrors_and_exact_speed():
    p, v, a = law.sample_blocks(np.random.default_rng(1), 512)
    speed_squared = (v * v).sum(axis=1)
    for _ in range(10000):
        p, v, bounce = law.step_blocks(p, v, a)
        assert np.all((p >= 1) & (p <= 15))
        np.testing.assert_array_equal((v * v).sum(axis=1), speed_squared)


def test_accelerated_energy_identity_and_speed_envelope():
    # Exactly the sampler called by MovingBlockWorld.reset, without an env/GL.
    p, v, a = law.sample_blocks(
        np.random.default_rng(7), 1024, accel_min=.002, accel_max=.01,
    )
    assert np.all(np.sqrt((v*v).sum(1) + 28*np.linalg.norm(a, axis=1)) <= 1.2)
    fixed_acc = a.copy()
    acc_norm = np.linalg.norm(a, axis=1)
    acc_squared = (a*a).sum(1)
    for step in range(10000):
        energy = (v*v).sum(1)/2 - (a*p).sum(1)
        invariant = ((v + a/2)**2).sum(1)/2 - (a*p).sum(1)
        kicked = v + a
        candidate = p + kicked
        p, v, bounce = law.step_blocks(p, v, a)
        delta = (v*v).sum(1)/2 - (a*p).sum(1) - energy
        # Kick-drift error plus the potential change from mirroring overshoot.
        expected = -.5*(a*a).sum(1) + (a*(candidate-p)).sum(1)
        np.testing.assert_allclose(delta, expected, atol=2e-15, rtol=0)
        # A reflected overshoot is no longer than the kicked displacement.
        bound = acc_squared/2 + 2*acc_norm*np.linalg.norm(kicked, axis=1)
        assert np.all(np.abs(delta) <= bound + 2e-15)
        free = ~bounce.any(axis=1)
        next_invariant = ((v + a/2)**2).sum(1)/2 - (a*p).sum(1)
        np.testing.assert_allclose(next_invariant[free], invariant[free], atol=2e-15, rtol=0)
        assert np.all((p >= 1) & (p <= 15))
        if step < 5000:
            assert np.linalg.norm(v, axis=1).max() <= 1.25
    np.testing.assert_array_equal(a, fixed_acc)


@pytest.mark.parametrize("p, v, a", [
    ([15., 8.], [1., 0.], [.01, 0.]),
    ([1., 1.], [-.7, -.7], [-.006, -.008]),
    ([15., 15.], [.7, .7], [.006, .008]),
    ([8., 8.], [0., 0.], [.01, 0.]),
])
def test_energy_bound_at_walls_and_free_steps(p, v, a):
    p, v, a = (np.array([value]) for value in (p, v, a))
    energy = (v*v).sum()/2 - (a*p).sum()
    p1, v1, _ = law.step_blocks(p, v, a)
    delta = abs((v1*v1).sum()/2 - (a*p1).sum() - energy)
    bound = np.linalg.norm(a)**2/2 + 2*np.linalg.norm(a)*np.linalg.norm(v + a)
    assert delta <= bound + 2e-15


def test_sampling_reproducibility_rejection_and_empty():
    kwargs = dict(bounds=(2., 12.), speed_min=.9, speed_max=1.2,
                  accel_min=.002, accel_max=.01, vmax=1.05)
    first = law.sample_blocks(np.random.default_rng(3), 1000, **kwargs)
    second = law.sample_blocks(np.random.default_rng(3), 1000, **kwargs)
    for x, y in zip(first, second):
        np.testing.assert_array_equal(x, y)
    p, v, a = first
    assert np.all(np.sqrt((v*v).sum(1) + 20*np.linalg.norm(a, axis=1)) <= 1.05)
    assert np.all((p >= 2) & (p <= 12))
    assert np.all((np.linalg.norm(a, axis=1) >= .002) & (np.linalg.norm(a, axis=1) <= .01))
    for kwargs in ({"speed_min": 2., "speed_max": 3.}, {"bounds": (2, 2)},
                   {"accel_min": .02, "accel_max": .01},
                   {"integer": True, "bounds": (1.5, 15.)}):
        with pytest.raises(ValueError):
            law.sample_blocks(np.random.default_rng(0), 1, **kwargs)
    p, v, a = law.sample_blocks(np.random.default_rng(0), 0)
    assert law.step_blocks(p, v, a)[0].shape == (0, 2)


@pytest.mark.parametrize("maximum", [1e-4, 2e-3, 1e-2])
def test_implicit_acceleration_minimum(maximum):
    _, _, acc = law.sample_blocks(np.random.default_rng(9), 100, accel_max=maximum)
    magnitude = np.linalg.norm(acc, axis=1)
    assert np.all(magnitude >= min(2e-3, maximum) - 1e-15)
    assert np.all(magnitude <= maximum + 1e-15)


def env_kwargs(**extra):
    from miniworld.params import DEFAULT_PARAMS
    params = DEFAULT_PARAMS.no_random()
    for key, value in (("turn_step", 90.), ("forward_step", 1.), ("cam_fov_y", 60.)):
        params.set(key, value, value, value)
    return dict(size=16, params=params, num_blocks=8, ensure_base_palette=True,
                spawn_wall_buffer=1., block_size_xy=.7, block_height=1.5,
                box_allow_overlap=True, agent_box_allow_overlap=True,
                render_mode="rgb_array", obs_width=128, obs_height=128,
                window_width=128, window_height=128, **extra)


@rendering
@pytest.mark.parametrize("integer", [False, True])
def test_env_reset_and_steps(integer):
    from miniworld.envs.moving_blockworld import MovingBlockWorld
    env = MovingBlockWorld(**env_kwargs(k3d_mode=True, k3d_integer=integer,
                                       k3d_accel_min=.002, k3d_accel_max=.01))
    try:
        seed = 3 if integer else 7
        env.reset(seed=seed)
        initial = env.k3d_block_state()
        env.reset(seed=seed)
        for key, value in initial.items():
            np.testing.assert_array_equal(value, env.k3d_block_state()[key])
        assert len(env._k3d_blocks) == env.num_blocks == 8
        # State accessors must never expose mutable internal arrays.
        env.k3d_block_state()["pos"][:] = -999
        np.testing.assert_array_equal(initial["pos"], env.k3d_pos)
        rng = np.random.default_rng(12)
        for _ in range(30):
            p, v, bounce = law.step_blocks(env.k3d_pos, env.k3d_vel, env.k3d_acc)
            env.step(int(rng.integers(0, 4)))
            for key, value in (("pos", p), ("vel", v), ("bounce", bounce)):
                np.testing.assert_array_equal(env.k3d_block_state()[key], value)
            for ent, pos, vel in zip(env._k3d_blocks, env.k3d_pos, env.k3d_vel):
                np.testing.assert_array_equal(ent.pos[[0, 2]], pos)
                np.testing.assert_array_equal(ent.velocity[[0, 2]], vel)
            np.testing.assert_array_equal(env.agent.pos[[0, 2]], np.round(env.agent.pos[[0, 2]]))
            assert env.agent.dir / (np.pi/2) == pytest.approx(round(env.agent.dir / (np.pi/2)))
            if integer:
                np.testing.assert_array_equal(env.k3d_pos, np.round(env.k3d_pos))
                assert np.isin(env.k3d_vel, [-1., 0., 1.]).all()
                np.testing.assert_array_equal(np.abs(env.k3d_vel).sum(1), np.ones(8))
                assert np.all(env.k3d_acc == 0)
        env.blocks_static = True
        before = env.k3d_block_state()
        env.step(env.actions.do_nothing)
        np.testing.assert_array_equal(before["pos"], env.k3d_pos)
        assert not env.k3d_bounce.any()
        assert not env.k3d_vel.any() and not env.k3d_acc.any()
        assert all(not ent.velocity.any() for ent in env._k3d_blocks)
        env.reset(seed=seed)
        before = env.k3d_pos.copy()
        assert not env.k3d_vel.any() and not env.k3d_acc.any()
        for _ in range(3):
            env.step(env.actions.do_nothing)
            np.testing.assert_array_equal(before, env.k3d_pos)
            assert not env.k3d_block_state()["speed"].any()
            assert all(not ent.velocity.any() for ent in env._k3d_blocks)
    finally:
        env.close()
    with pytest.raises(ValueError, match="mutually exclusive"):
        MovingBlockWorld(grid_mode=True, k3d_mode=True)


@rendering
def test_k3d_single_observation_render_preserves_rgb(monkeypatch):
    from miniworld.envs.moving_blockworld import MovingBlockWorld
    results = []
    for extra_render in (False, True):
        env = MovingBlockWorld(**env_kwargs(k3d_mode=True, k3d_accel_max=1e-4))
        observed = []
        render_obs = env.render_obs
        def capture_obs(*args, **kwargs):
            obs = render_obs(*args, **kwargs)
            observed.append(obs)
            return obs
        monkeypatch.setattr(env, "render_obs", capture_obs)
        try:
            frames = []
            for action in [None] + [0, 1, 2, 4]*5:
                observed.clear()
                result = env.reset(seed=7) if action is None else env.step(action)
                assert len(observed) == 1
                assert result[0] is observed[0]
                # Reproduce the pre-fix second render after the block update.
                if extra_render:
                    env.render_obs()
                frames.append(env.render())
            assert env.k3d_accel_min == env.k3d_accel_max == 1e-4
            results.append(np.stack(frames))
        finally:
            env.close()
    np.testing.assert_array_equal(*results)


RELEASED = """--env-name MiniWorld-MovingBlockWorld-v0
--turn-step-deg 90 --forward-step 1.0 --heading-zero
--grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit
--render-width 128 --render-height 128 --obs-width 128 --obs-height 128
--steps 40 --output-2d-map --room-size 16 --block-size-xy 0.7 --block-height 1.5
--agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only
--policy biased_walk_v2 --forward-prob 0.92 --cam-fov-y 60
--num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette --store-block-info""".split()


def k3d_cli_args(steps):
    argv = RELEASED.copy()
    for flag, count in (("--grid-mode", 1), ("--grid-vel-min", 2),
                        ("--grid-vel-max", 2), ("--grid-cardinal-only", 1)):
        index = argv.index(flag)
        del argv[index:index+count]
    argv[argv.index("--steps") + 1] = str(steps)
    return argv + ["--k3d-mode", "--no-depth", "--ffmpeg-threads", "1"]


def parse_args(module, monkeypatch, argv):
    class Parsed(Exception):
        pass
    result = []
    def capture(args, **kwargs):
        result.append(args)
        raise Parsed
    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", ["generate_videos"] + argv)
        patch.setattr(module, "build_env", capture)
        with pytest.raises(Parsed):
            module.main()
    return result[0]


def load_rgb_writer(path):
    # Execute the actual writer in isolation: importing the generator imports GL.
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "write_mp4_rgb")
    namespace = {"os": os, "np": np, "Optional": Optional}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["write_mp4_rgb"]


def test_default_rgb_writer_matches_base_md5(tmp_path):
    frames = np.random.default_rng(815).integers(0, 256, (16, 128, 128, 3), dtype=np.uint8)
    digests = []
    for name, source in (("current", ROOT / "scripts/generate_videos.py"),
                         ("base", ROOT / "tests/reference/generate_videos_pre_k3d.py")):
        path = tmp_path / f"{name}.mp4"
        load_rgb_writer(source)(str(path), frames)
        assert path.stat().st_size > 0
        digests.append(hashlib.md5(path.read_bytes()).hexdigest())
    assert digests[0] == digests[1]


def test_rgb_writer_thread_option(monkeypatch):
    import imageio
    from contextlib import nullcontext
    options = []
    def get_writer(path, **kwargs):
        options.append(kwargs)
        return nullcontext()
    monkeypatch.setattr(imageio, "get_writer", get_writer)
    writer = load_rgb_writer(ROOT / "scripts/generate_videos.py")
    writer("unused.mp4", [])
    writer("unused.mp4", [], ffmpeg_threads=3)
    assert "ffmpeg_params" not in options[0]
    assert options[1]["ffmpeg_params"] == ["-threads", "3", "-filter_threads", "3"]
    with pytest.raises(ValueError, match="positive"):
        writer("unused.mp4", [], ffmpeg_threads=0)


@rendering
def test_released_generator_bit_identical(monkeypatch):
    import gymnasium as gym
    from scripts import generate_videos as current
    reference = load_module("pre_k3d_generator", ROOT / "tests/reference/generate_videos_pre_k3d.py")
    reference_env = load_module("pre_k3d_env", ROOT / "tests/reference/moving_blockworld_pre_k3d.py")
    gym.register("MiniWorld-PreK3D-Test-v0", entry_point=reference_env.MovingBlockWorld)
    # Base main ignored --seed. Control both RNGs explicitly for a valid baseline.
    class Seeded(gym.Wrapper):
        def reset(self, **kwargs):
            return self.env.reset(seed=37)
    results = []
    for module, env_id in ((reference, "MiniWorld-PreK3D-Test-v0"),
                           (current, "MiniWorld-MovingBlockWorld-v0")):
        args = parse_args(module, monkeypatch, RELEASED)
        args.env_name = env_id
        with monkeypatch.context() as patch:
            patch.setattr(random, "randint", random.Random(37).randint)
            env = Seeded(module.build_env(args))
        try:
            extra = {} if module is reference else {"capture_depth": False}
            results.append(module.run_rollout(
                env, 40, True, 10, "biased_walk_v2", {"forward_prob": .92},
                capture_top=True, store_block_info=True, **extra,
            ))
        finally:
            env.close()
    # RGB, actions, map, agent pose, deltas, and absolute heading, exactly.
    for index in (0, 2, 3, 4, 5, 6, 7):
        np.testing.assert_array_equal(results[0][index], results[1][index])
    np.testing.assert_array_equal(results[0][-1]["block_pos_world"], results[1][-1]["block_pos_world"])


@rendering
@pytest.mark.parametrize("accel", [False, True])
def test_generator_outputs(monkeypatch, tmp_path, accel):
    import cv2
    import torch
    cv2.setNumThreads(1)
    from miniworld.miniworld import MiniWorldEnv
    from scripts import generate_videos as generator
    def forbidden_depth(*args, **kwargs):
        pytest.fail("--no-depth called render_depth")
    monkeypatch.setattr(MiniWorldEnv, "render_depth", forbidden_depth)
    argv = k3d_cli_args(20) + ["--seed", "1"]
    if accel:
        argv += ["--k3d-accel-min", "2e-3", "--k3d-accel-max", "1e-2"]
    saved = []
    for name in ("one", "repeated"):
        prefix = tmp_path / name
        monkeypatch.setattr(sys, "argv", ["generate_videos"] + argv + ["--out-prefix", str(prefix)])
        generator.main()
        assert not Path(f"{prefix}_depth.pt").exists()
        for suffix in ("rgb", "map_2d"):
            cap = cv2.VideoCapture(f"{prefix}_{suffix}.mp4", cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_N_THREADS, 1])
            try:
                assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 20
                assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 128
            finally:
                cap.release()
        meta = torch.load(f"{prefix}_actions.pt", weights_only=True)
        assert set(meta) == {"actions", "agent_pos", "delta_xz", "delta_dir", "agent_dir", "top_view_scale"}
        info = torch.load(f"{prefix}_block_info.pt", weights_only=True)
        n = random.Random(1).randint(6, 10)
        assert info["seed"] == 1 and info["num_blocks"] == n
        for key, shape in (("block_pos_world", (20,n,3)), ("block_vel_world", (20,n,3)),
                           ("block_acc_world", (20,n,2)), ("block_vel_k3d", (20,n,2)),
                           ("block_speed", (20,n)), ("block_bounce", (20,n,2))):
            assert tuple(info[key].shape) == shape
            assert info[key].dtype == (torch.bool if key == "block_bounce" else torch.float32)
        assert set(info["k3d"]) == {"integer", "speed_min", "speed_max", "accel_min", "accel_max", "vmax", "bounds"}
        assert info["k3d"]["bounds"] == (1., 15.)
        pos = info["block_pos_world"][:, :, (0, 2)].numpy()
        vel = info["block_vel_k3d"].numpy()
        bounce = info["block_bounce"].numpy()
        free = ~bounce[:-1]
        np.testing.assert_allclose(pos[1:][free], (pos[:-1]+vel[:-1])[free], atol=2e-6, rtol=0)
        candidate = pos[:-1] + vel[:-1]
        mirrored = np.where(candidate < 1, 2-candidate, np.where(candidate > 15, 30-candidate, candidate))
        np.testing.assert_allclose(pos[1:], mirrored, atol=2e-6, rtol=0)
        state_vel = np.where(bounce[:-1], -vel[:-1], vel[:-1])
        np.testing.assert_allclose(vel[1:], state_vel+info["block_acc_world"].numpy()[1:], atol=2e-7, rtol=0)
        np.testing.assert_allclose(info["block_speed"], np.linalg.norm(vel, axis=2), atol=2e-7, rtol=0)
        saved.append((meta, info))
    for first, second in zip(saved[0], saved[1]):
        for key, value in first.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, second[key])
            else:
                assert value == second[key]


@pytest.mark.parametrize("before, after", [
    (["--seed", "7"], []),
    ([], ["--seed", "7"]),
    ([], ["--seed=7"]),
    (["--seed", "7"], ["--seed", "7"]),
    (["--seed", "7"], ["--seed=7"]),
])
def test_batch_forwards_flags_and_item_seeds(monkeypatch, tmp_path, before, after):
    from scripts import generate_videos_batch as batch
    commands = []
    def capture(cmd, env):
        commands.append(cmd)
        return True
    monkeypatch.setattr(batch, "_run_one", capture)
    forwarded = ["--k3d-mode", "--k3d-integer", "--k3d-bounds", "2", "14",
                 "--k3d-speed-min", ".3", "--k3d-speed-max", ".9",
                 "--k3d-accel-min", ".002", "--k3d-accel-max", ".01",
                 "--k3d-vmax", "1.1", "--no-depth"]
    monkeypatch.setattr(sys, "argv", ["generate_videos_batch", "--env-name", "MiniWorld-MovingBlockWorld-v0",
        "--dataset-root", str(tmp_path), "--num-videos", "2", "--num-processes", "1",
        "--store-block-info"] + before + ["--"] + forwarded + after)
    batch.main()
    assert len(commands) == 2
    for index, command in enumerate(commands):
        assert command[7:7+len(forwarded)] == forwarded
        assert command.count("--store-block-info") == 1
        assert command.count("--seed") == 1
        assert command[-2:] == ["--seed", str(7+index)]


@pytest.mark.parametrize("after", [["--seed", "8"], ["--seed=8"]])
def test_batch_rejects_conflicting_seeds(monkeypatch, tmp_path, capsys, after):
    from scripts import generate_videos_batch as batch
    monkeypatch.setattr(sys, "argv", ["generate_videos_batch", "--env-name", "unused",
        "--dataset-root", str(tmp_path / "not-created"), "--num-videos", "2",
        "--seed", "7", "--"] + after)
    with pytest.raises(SystemExit) as exc:
        batch.main()
    assert exc.value.code == 2
    assert "must agree" in capsys.readouterr().err
    assert not (tmp_path / "not-created").exists()


@rendering
def test_batch_forwarded_seed_generates_distinct_clips(tmp_path):
    import torch
    base_seed, minimum, maximum = 100, 6, 10
    forwarded = k3d_cli_args(5)[2:]  # --env-name is supplied by the launcher.
    command = [sys.executable, "-m", "scripts.generate_videos_batch",
               "--env-name", "MiniWorld-MovingBlockWorld-v0",
               "--dataset-root", str(tmp_path), "--num-videos", "2",
               "--num-processes", "1", "--"] + forwarded + ["--seed", str(base_seed)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "failures:" not in completed.stdout, completed.stdout + completed.stderr
    states = []
    for index in range(2):
        prefix = tmp_path / "0" / f"{index:04}"
        info = torch.load(f"{prefix}_block_info.pt", weights_only=True)
        seed = base_seed + index
        count = random.Random(seed).randint(minimum, maximum)
        assert info["seed"] == seed
        assert info["num_blocks"] == count
        assert info["block_pos_world"].shape == (5, count, 3)
        assert not Path(f"{prefix}_depth.pt").exists()
        states.append(info["block_pos_world"])
    # Compare one block too, so unequal counts alone cannot satisfy this check.
    assert not torch.equal(states[0][:, 0], states[1][:, 0])


@rendering
def test_main_decoded_rgb_matches_direct_rollout(monkeypatch, tmp_path):
    import cv2
    import gymnasium as gym
    from miniworld.envs.moving_blockworld import MovingBlockWorld
    from scripts import generate_videos as generator
    cv2.setNumThreads(1)
    # A gray palette retains moving geometry while avoiding YUV420 chroma-edge
    # losses, which can exceed 8/255 even for a perfect source RGB sequence.
    env_id = "MiniWorld-K3D-Gray-Codec-Test-v0"
    gym.register(env_id, entry_point=lambda **kwargs: MovingBlockWorld(color_pool=["grey"], **kwargs))
    prefix = tmp_path / "cli"
    argv = k3d_cli_args(12)
    argv[argv.index("--env-name") + 1] = env_id
    argv.remove("--ensure-base-palette")
    argv.remove("--output-2d-map")
    argv += ["--seed", "7", "--out-prefix", str(prefix),
             "--floor-tex", "white", "--wall-tex", "white", "--ceil-tex", "white"]
    args = parse_args(generator, monkeypatch, argv)
    monkeypatch.setattr(sys, "argv", ["generate_videos"] + argv)
    generator.main()
    assert not Path(f"{prefix}_depth.pt").exists()

    env = generator.build_env(args, count_rng=random.Random(args.seed))
    try:
        expected = generator.run_rollout(
            env, 12, True, 10, "biased_walk_v2", {"forward_prob": .92},
            capture_depth=False, seed=args.seed,
        )[0]
    finally:
        env.close()
    cap = cv2.VideoCapture(f"{prefix}_rgb.mp4", cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_N_THREADS, 1])
    try:
        assert cap.isOpened()
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 12
        decoded = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            decoded.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    assert len(decoded) == len(expected) == 12
    assert np.any(expected[1:] != expected[:-1])
    assert np.max(np.abs(np.stack(decoded).astype(np.int16) - expected.astype(np.int16))) <= 8
