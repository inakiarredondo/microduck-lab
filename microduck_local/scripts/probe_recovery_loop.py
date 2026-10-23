"""CLOSE THE LOOP: stand -> get shoved over -> get up again -> stand.

Until now the mode switch was ONE WAY. `handoff_fn` hands the get-up to
Menlo's walker once the robot is up and nothing ever hands back, so a shove
hard enough to floor it left the robot down for good even though it can now
rise from either posture. This drives the full cycle and counts it.

THE SWITCH IS DEPLOYABLE, both directions. Menlo's 78-obs layout is gyro,
projected gravity, joint positions, joint velocities and the last action --
no world-frame trunk height, no contact list, no linear velocity -- so both
predicates read only that:

  up   : uprightness > 0.925 AND |elbow| < 0.40 AND |ankle_pitch| < 0.525,
         held 4 ticks. Uprightness alone agreed with the privileged test on
         99.0% of frames and still cost 11 of 48 rises, because the switch
         latches and the frames that fool it are the robot propped on its
         FOREARMS, torso already vertical.
  down : uprightness < 0.40, held `DOWN_DWELL` ticks. The dwell is the settle
         clause: handing the get-up a state mid-fall is the mistake that cost
         7.5M steps ("reachable is not recoverable"), and 21 of 24 starts in
         that cache were the robot already horizontal with downward momentum.

Shove magnitude from `probe_push_envelope.py`: the walker survives <=0.8
bodyweight 12/12 and 1.6 bw floors it 9 of 12.

    uv run python scripts/probe_recovery_loop.py <run> [episodes] [bodyweights]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from microduck_local.robots import asimov_getup_ref as R
from microduck_local.robots import asimov_walk as W
from microduck_local.robots import spec as S
from microduck_local.run_record import replay_kwargs

UP_THRESH, UP_DWELL = 0.925, 4
ELBOW, ANKLE = 0.400, 0.525
DOWN_THRESH, DOWN_DWELL = 0.40, 30      # 0.6 s on the floor before re-entry
PUSH_S = 0.15
EPISODE_S = 30.0                        # long enough for several cycles
FIRST_PUSH_S = 6.0                      # after the first stand
PUSH_EVERY_S = 6.0
QUIET_S = 8.0       # no shoves in the last stretch: time to get back up


def onnx(p):
    import onnxruntime as ort
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    n = s.get_inputs()[0].name
    return lambda o: s.run(None, {n: o.reshape(1, -1).astype(np.float32)})[0][0]


_EL = [i for i, n in enumerate(W.JOINT_NAMES) if "elbow" in n]
_AN = [i for i, n in enumerate(W.JOINT_NAMES) if "ankle_pitch" in n]


def obs_up(o) -> bool:
    a = np.asarray(o)
    jp = a[slice(*W.OBS_JOINT_POS)]
    return bool(float(-a[slice(*W.OBS_GRAVITY)][2]) > UP_THRESH
                and max(abs(float(jp[i])) for i in _EL) < ELBOW
                and max(abs(float(jp[i])) for i in _AN) < ANKLE)


def obs_down(o) -> bool:
    return bool(float(-np.asarray(o)[slice(*W.OBS_GRAVITY)][2]) < DOWN_THRESH)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    run = argv[0]
    n_ep = int(argv[1]) if len(argv) > 1 else 12
    bw = float(argv[2]) if len(argv) > 2 else 1.6

    kw = dict(replay_kwargs(Path("runs") / run))
    kw.pop("task", None)
    kw.pop("handoff_pay", None)
    kw["max_episode_s"] = EPISODE_S
    env = S.get("asimov1").env_class("getup")(
        domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
    getup = onnx(Path("runs") / run / "policy.onnx")
    walk = onnx(".cache/asimov_1/policy/policy.onnx")
    m, d = env.model, env.data
    TB = env.trunk_body_id
    MASS = float(m.body_subtreemass[TB])
    horizon = int(EPISODE_S * 50)

    def episode(ep, two_way: bool):
        o, _ = env.reset(seed=ep)
        on_walker = False
        up_n = down_n = 0
        stands = 0            # times it reached a stand
        shoves = caught = recovered = 0
        next_push = None
        push_until = -1
        was_down = False
        for k in range(horizon):
            d.xfrc_applied[:] = 0.0
            if not on_walker and obs_up(o):
                up_n += 1
                if up_n >= UP_DWELL:
                    on_walker, up_n, down_n = True, 0, 0
                    stands += 1
                    if was_down:
                        recovered += 1
                        was_down = False
                    if next_push is None:
                        next_push = k + int(FIRST_PUSH_S * 50)
            elif not on_walker:
                up_n = 0
            if on_walker and two_way:
                if obs_down(o):
                    down_n += 1
                    if down_n >= DOWN_DWELL:
                        on_walker, down_n = False, 0   # HAND BACK
                        was_down = True
                else:
                    down_n = 0
            elif on_walker and obs_down(o):
                was_down = True                        # one-way: stays down
            # STOP SHOVING with time left to rise, or "up at the end" just
            # measures whether the clock happened to stop mid-recovery: at
            # 1.6 bw the first run read 2/12 up while recovering 34 times.
            if (next_push is not None
                    and next_push > horizon - int(QUIET_S * 50)):
                next_push = None
            if next_push is not None and k == next_push:
                push_until = k + int(PUSH_S * 50)
                shoves += 1
                next_push = k + int(PUSH_EVERY_S * 50)
            if k < push_until:
                d.xfrc_applied[TB, 1] = bw * MASS * 9.81
            o, _r, t, tr, _ = env.step((walk if on_walker else getup)(o))
            # IGNORE `terminated`. Once the robot has stood, this env ends
            # the episode the moment it goes back down (`_stood` +
            # `FELL_GRAVITY`/`FELL_HEIGHT_FRAC`) — a TRAINING cut that stops
            # wasting samples on a failed episode. A real robot that falls
            # does not vanish, and honouring that cut here made the whole
            # loop unmeasurable: episodes ended at the shove, so 12 episodes
            # fired 13 shoves instead of ~50 and recovery could never have
            # happened on either arm. "Score at the horizon the deliverable
            # is defined at, with only the DEPLOYMENT's own cuts."
            # `terminated` only writes episode info here, so stepping past it
            # is physically safe; a non-finite state is the real stop.
            if tr:
                break
            if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
                break
        caught = int(R.is_standing(env))
        return stands, shoves, recovered, caught

    print(f"{run}: {n_ep} episodes of {EPISODE_S:.0f}s, shoved at {bw} "
          f"bodyweights ({bw * MASS * 9.81:.0f} N) every "
          f"{PUSH_EVERY_S:.0f}s once standing, quiet for the last "
          f"{QUIET_S:.0f}s\n")
    print(f"{'switch':12} {'stands':>8} {'shoves':>8} {'re-ups':>8} "
          f"{'up at the end':>14}")
    for name, two_way in (("ONE WAY", False), ("TWO WAY", True)):
        tot = np.array([episode(ep, two_way) for ep in range(n_ep)])
        print(f"{name:12} {tot[:, 0].sum():8} {tot[:, 1].sum():8} "
              f"{tot[:, 2].sum():8} {tot[:, 3].sum():>9}/{n_ep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
