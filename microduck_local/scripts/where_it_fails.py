"""WHERE a get-up dies, and WHICH physical thing stopped it.

Rise/hold counts say a policy failed; they never say where or why, so six
experiments on this body were aimed at the whole motion when ONE transition
was broken. This localises it. For every episode it records the furthest
authored waypoint the trunk reached, the cut that ended the episode, and — at
the moment the robot stopped climbing — what was actually maxed out:

  * a JOINT RANGE problem leaves joints pinned against `jnt_range`
  * a TORQUE problem leaves actuators against `actuator_forcerange`
  * neither means the pose is reachable and holdable and the policy simply
    is not going there, which is an exploration problem and no amount of
    reward or range buys it

The zero-action arm is always measured, because a floor that outscores the
policy is the only way to tell a weak policy from a broken instrument.

    uv run python scripts/where_it_fails.py <run> [episodes]
"""
import json
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
from stable_baselines3 import PPO

from microduck_local.robots import asimov_getup_ref as R
from microduck_local.robots import spec as S

RUN = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
#: Within this fraction of a joint's half-range of its stop counts as pinned.
AT_LIMIT = 0.02
#: Above this fraction of an actuator's rated force counts as saturated.
SATURATED = 0.90

ek = json.load(open(f"runs/{RUN}/run.json")).get("env_kwargs") or {}
# Score on the DEPLOYMENT task with the deployment's own cuts: no span cut, no
# curriculum start. A curriculum's terminations turned 23/24 into 12/24 once.
keep = ("action_clip", "getup_blend", "getup_assist")
base = {k: v for k, v in ek.items() if k in keep}
env = S.get("asimov1").env_class("getup")(
    domain_rand=False, obs_noise=True, action_delay=True, seed=0, **base)
m, d = env.model, env.data
# The CONTROL step, read off the contract — not the model timestep, which
# is the physics substep and 4x shorter here.
_c = S.get("asimov1").contract
DT = 1.0 / float((_c() if callable(_c) else _c).rate_hz)

TB = env.trunk_body_id
LF = m.geom_bodyid[env.foot_geoms["left"]]
RF = m.geom_bodyid[env.foot_geoms["right"]]
LADDER = [(w.name, w.settled_z) for w in R.WAYPOINTS]

# The actuated joints, their stops, and their rated forces — read off THIS
# model, never a datasheet.
act = []
for a in range(m.nu):
    j = m.actuator_trnid[a, 0]
    lo, hi = m.jnt_range[j]
    fr = float(np.abs(m.actuator_forcerange[a]).max()) or 0.0
    act.append((a, j, m.joint(j).name, float(lo), float(hi), fr))

net = vn = None
try:
    net = PPO.load(f"runs/{RUN}/model", device="cpu")
    vn = pickle.load(open(f"runs/{RUN}/vecnormalize.pkl", "rb"))
except Exception as exc:                                   # pragma: no cover
    print(f"no policy loadable for {RUN}: {exc}")
    raise


def phase_of(z: float) -> str:
    """The furthest waypoint whose settled height the trunk has passed."""
    name = "below prone"
    for nm, sz in LADDER:
        if z >= sz - 0.01:
            name = nm
    return name


def run(policy):
    reached = Counter()
    ended = Counter()
    pinned = defaultdict(float)
    hot = defaultdict(float)
    stalls = []
    for ep in range(N):
        obs, _ = env.reset(seed=ep)
        best = float(d.xpos[TB][2])
        best_at = 0
        # What was maxed out at the moment it stopped climbing.
        snap_pin: list[str] = []
        snap_hot: list[str] = []
        cause = "ran out of time"
        for k in range(int(env.max_steps)):
            if policy is None:
                a = np.zeros(env.nj, np.float32)
            else:
                n = np.clip((obs - vn.obs_rms.mean)
                            / np.sqrt(vn.obs_rms.var + vn.epsilon),
                            -vn.clip_obs, vn.clip_obs).astype(np.float32)
                a = net.predict(n, deterministic=True)[0]
            obs, _r, term, trunc, _i = env.step(np.asarray(a, np.float32))
            z = float(d.xpos[TB][2])
            now_pin, now_hot = [], []
            for _a, j, nm, lo, hi, fr in act:
                q = float(d.qpos[m.jnt_qposadr[j]])
                span = hi - lo
                if span > 0 and (q - lo < AT_LIMIT * span
                                 or hi - q < AT_LIMIT * span):
                    now_pin.append(nm)
                    pinned[nm] += DT
                if fr > 0 and abs(float(d.actuator_force[_a])) > SATURATED * fr:
                    now_hot.append(nm)
                    hot[nm] += DT
            if z > best:
                best, best_at, snap_pin, snap_hot = z, k, now_pin, now_hot
            if term or trunc:
                cause = ("stood" if bool(getattr(env, "_stood", False))
                         else "cut")
                break
        reached[phase_of(best)] += 1
        ended[cause] += 1
        stalls.append((best, best_at * DT, tuple(snap_pin), tuple(snap_hot),
                       float(np.hypot(d.xpos[LF][0] - d.xpos[RF][0],
                                      d.xpos[LF][1] - d.xpos[RF][1]))))
    return reached, ended, pinned, hot, stalls


print(f"{RUN}   {N} episodes from prone, deployment cuts only")
print(f"  env rebuilt with {base}\n")
for label, pol in (("THE POLICY", net), ("zero action", None)):
    reached, ended, pinned, hot, stalls = run(pol)
    peaks = np.array([s[0] for s in stalls])
    print(f"{label}:  peak trunk median {np.median(peaks):.3f} m "
          f"(best {peaks.max():.3f})")
    print("  furthest waypoint reached: "
          + ", ".join(f"{k} x{v}" for k, v in reached.most_common()))
    print("  episode ended by: "
          + ", ".join(f"{k} x{v}" for k, v in ended.most_common()))
    tot = N * env.max_steps * DT
    pin = [(k, v) for k, v in sorted(pinned.items(), key=lambda x: -x[1])[:6]]
    sat = [(k, v) for k, v in sorted(hot.items(), key=lambda x: -x[1])[:6]]
    print("  joints pinned at a stop: "
          + (", ".join(f"{k} {v / tot:5.1%}" for k, v in pin) or "none"))
    print("  actuators above 90 % rated: "
          + (", ".join(f"{k} {v / tot:5.1%}" for k, v in sat) or "none"))
    # At the stall itself, not averaged over the episode.
    sp = Counter(x for s in stalls for x in s[2])
    sh = Counter(x for s in stalls for x in s[3])
    print(f"  AT THE STALL: pinned {dict(sp.most_common(4)) or 'nothing'}"
          f"   saturated {dict(sh.most_common(4)) or 'nothing'}")
    print(f"  stall time median {np.median([s[1] for s in stalls]):.2f} s"
          f"   foot span at the end, median "
          f"{np.median([s[4] for s in stalls]):.3f} m\n")
