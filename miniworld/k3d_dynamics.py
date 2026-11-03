"""NumPy-only kick, drift, mirror dynamics for block centres in the XZ plane.

Acceleration is a constant world-frame field and is never reflected. The
rejection envelope uses |a| * (hi - lo); the exact worst-case potential drop
over the square is (|a_x| + |a_z|) * (hi - lo). The latter gives an energy-based
speed envelope of about 1.18 at the defaults, before discrete bounce errors.
Neither envelope rejects any pairs at the K3D-a defaults. ``vmax`` is not a
hard bound: review measurements over 4000 blocks found maximum outgoing speeds
of 1.148 / 1.209 / 1.249 at 500 / 5000 / 50000 steps.

The discrete law conserves |v + a/2|^2/2 - a.p on free steps; reflections can
change it. For E = |v|^2/2 - a.p, every step satisfies
|delta E| <= |a|^2/2 + 2|a||v + a|, where v is the pre-kick velocity.
"""

import numpy as np


def step_blocks(pos, vel, acc, bounds=(1.0, 15.0)):
    """Return new ``(pos, vel, bounce)`` arrays without modifying inputs.

    Arrays have shape (N, 2). Kick velocity by ``acc``, drift, then reflect
    each out-of-bounds coordinate once. Returned velocity is the *post-bounce*
    state velocity. ``bounce`` describes this transition, per block and axis.
    Speeds must be small enough that one mirror per axis suffices.
    """
    lo, hi = bounds
    vel = vel + acc
    pos = pos + vel
    below, above = pos < lo, pos > hi
    bounce = below | above
    pos = np.where(below, 2 * lo - pos, np.where(above, 2 * hi - pos, pos))
    vel = np.where(bounce, -vel, vel)
    assert np.all((lo <= pos) & (pos <= hi)), "one reflection did not restore bounds"
    return pos, vel, bounce


def sample_blocks(
    rng, num_blocks, bounds=(1.0, 15.0), *, integer=False,
    speed_min=0.25, speed_max=1.0, accel_min=0.0, accel_max=0.0, vmax=1.2,
):
    """Sample ``(pos, vel, acc)`` using the environment's NumPy RNG.

    When acceleration is enabled, a zero ``accel_min`` selects
    min(0.002, accel_max). Resample each rejected (velocity, acceleration) pair
    without changing its position.
    The required envelope is sqrt(speed**2 + 2*|acc|*(hi-lo)) <= vmax.
    Integer mode uses cardinal unit velocities and zero acceleration.
    """
    lo, hi = map(float, bounds)
    values = [lo, hi, speed_min, speed_max, accel_min, accel_max, vmax]
    if not np.all(np.isfinite(values)) or lo >= hi or vmax <= 0:
        raise ValueError("k3d requires finite parameters, lo < hi, and vmax > 0")
    if num_blocks < 0:
        raise ValueError("num_blocks must be nonnegative")
    if integer:
        if lo != int(lo) or hi != int(hi) or vmax < 1:
            raise ValueError("integer k3d requires integral bounds and vmax >= 1")
        pos = rng.integers(int(lo), int(hi) + 1, size=(num_blocks, 2)).astype(float)
        cardinal = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        vel = cardinal[rng.integers(4, size=num_blocks)]
        return pos, vel, np.zeros_like(pos)
    if not 0 <= speed_min <= speed_max:
        raise ValueError("k3d requires 0 <= speed_min <= speed_max")
    if accel_max < 0 or accel_min < 0:
        raise ValueError("k3d acceleration magnitudes must be nonnegative")
    if accel_max > 0:
        accel_min = accel_min or min(2e-3, accel_max)
        if accel_min > accel_max:
            raise ValueError("k3d requires accel_min <= accel_max when enabled")
    min_acc = accel_min if accel_max > 0 else 0.0
    min_energy = speed_min**2 + 2 * min_acc * (hi - lo)
    variable = speed_max > speed_min or accel_max > min_acc
    if min_energy > vmax**2 or (variable and min_energy == vmax**2):
        raise ValueError("k3d rejection envelope cannot accept these parameters")

    pos = rng.uniform(lo, hi, size=(num_blocks, 2))
    vel, acc = np.empty_like(pos), np.zeros_like(pos)
    pending = np.arange(num_blocks)
    # Bound retries for configurations with a vanishingly small acceptance rate.
    for _ in range(10000):
        if not len(pending):
            return pos, vel, acc
        theta = rng.uniform(-np.pi, np.pi, size=len(pending))
        speed = rng.uniform(speed_min, speed_max, size=len(pending))
        vel[pending] = speed[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))
        if accel_max > 0:
            theta = rng.uniform(-np.pi, np.pi, size=len(pending))
            magnitude = np.exp(rng.uniform(np.log(accel_min), np.log(accel_max), size=len(pending)))
            acc[pending] = magnitude[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))
        envelope = np.sqrt(np.sum(vel[pending]**2, axis=1)
                           + 2 * np.linalg.norm(acc[pending], axis=1) * (hi - lo))
        pending = pending[envelope > vmax]
    raise ValueError("k3d rejection sampler exhausted 10000 attempts")
