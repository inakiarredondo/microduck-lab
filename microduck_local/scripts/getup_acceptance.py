"""DOES THE GET-UP MEET THE GOAL? One script, pass/fail, no human in the loop.

The goal, in Jonathan's words (2026-10-09/10): get up off the floor, stand
RELIABLY, go back to the default stand/idle pose, and CORRECT — not freeze in
whatever pose it happened to land in.

Every earlier scorer measured the rise. None measured the last clause, and it
is the one he kept seeing and I kept missing: a policy that reaches a pose and
holds it open-loop looks identical to a policy that balances, right up until
something pushes it. So this adds a DISTURBANCE sweep — shove the trunk once
the robot is up, and ask whether it is still standing afterwards.

Usage:  uv run python scripts/getup_acceptance.py <run> [<run> ...]
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

#: How many paired episodes. 24 was the default for every comparison made on
#: 2026-10-10, and an exact McNemar needs SIX all-one-sided discordant pairs
#: to clear p<=0.05 — so at 24 episodes this script had **3 % power** to
#: detect the 20/24-vs-22/24 difference it was being asked about, and every
#: "beats s11" / "fails to beat s11" that day was read off an instrument that
#: could not see the effect. 200 paired episodes gives 0.80.
#: `null-means-no-instrument`, the hard way.
EPISODES = 24

GATES = {                       # what "done" means, stated up front
    "rises":        (22, "reaches a stand from the floor, of 24"),
    "stays":        (20, "still standing at the end of the episode, of 24"),
    "home_deg":     (10.0, "MEAN joint error vs default_pose, degrees (<=)"),
    "home_max_deg": (20.0, "WORST single joint vs default_pose, degrees (<=)"),
    "jam_rad":      (0.010, "median radians outside a joint limit (<=)"),
    "span_m":       (0.40, "median peak foot span, metres (<=)"),
    "recovers":     (0.75, "fraction of SHOVED episodes still standing at the end"),
}


def load_onnx(path):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def act(o):
        import numpy as _np
        return sess.run(None, {name: o.reshape(1, -1).astype(_np.float32)})[0][0]
    return act


def load(run):
    from stable_baselines3 import PPO
    net = PPO.load(f"runs/{run}/model", device="cpu")
    vn = pickle.load(open(f"runs/{run}/vecnormalize.pkl", "rb"))

    def act(o):
        n = np.clip((o - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                    -vn.clip_obs, vn.clip_obs).astype(np.float32)
        return np.asarray(net.predict(n, deterministic=True)[0], np.float32)
    return act


HANDOFF = False
POSE = ""        # "" = each run's own; else a label or comma-separated set
OBS_SWITCH = False   # --obs-switch: the DEPLOYABLE predicate (see below)

#: THE SHIPPABLE MODE SWITCH, and why it is a different thing from the one
#: the lab uses. `_getup_handoff_due` asks `standing_report` (world-frame
#: trunk height + MuJoCo's contact list) and `_trunk_speed` (world-frame
#: linear velocity). An Asimov has none of the three: Menlo's 78-obs layout
#: is gyro, projected gravity, joint positions, joint velocities and the last
#: action — no linear velocity at all. So the switch that makes the whole
#: two-policy architecture work could not run on the robot, which defeats the
#: point of staying inside their contract.
#:
#: Measured 2026-10-11 over 15,458 steps across 40 episodes (supine+prone):
#: uprightness straight off the IMU separates standing from down at **4.58
#: sd** and reproduces the privileged test on **99.0 %** of frames at this
#: threshold. Trunk speed — the unobservable clause — is the WEAKEST
#: separator at 0.75 sd, and among upright frames 97.7 % already satisfy it,
#: so the dwell does that clause's work.
OBS_UPRIGHT = 0.925
OBS_DWELL = 4
#: ...and uprightness ALONE is not enough, which the end-to-end score caught
#: and the 99.0 % frame agreement hid: it cost **11 of 48 rises** (48 -> 37).
#: The frames that fool the IMU are not kneels — separated at 4.5-5.9 sd they
#: are the robot PROPPED ON ITS FOREARMS with the ankles cocked, torso
#: already vertical, still pushing up (elbows -0.38/-0.44 rad against ~0.00
#: when truly standing; ankle_pitch -0.46/+0.75 against ~0.00). Both are
#: encoder channels, so the discriminator ships.
#:
#: Thresholds chosen on a COST, not on agreement: the switch latches one way,
#: so a false fire hands a floored robot to the walker and cannot be undone,
#: while a false miss costs 20 ms. At 10:1 that gives false-UP 1.04 % ->
#: **0.07 %** for false-DOWN 0.85 % -> 3.41 % — and slightly WORSE overall
#: agreement (98.5 % vs 99.0 %), which is why agreement was the wrong
#: objective.
OBS_ELBOW = 0.400
OBS_ANKLE = 0.525


def obs_switch_due(o, state) -> bool:
    """Is it standing, by what the ROBOT can sense? `o` is the 78-obs vector.

    Gyro, projected gravity, joint positions, joint velocities, last action —
    Menlo's published layout, all of it available on hardware. No world-frame
    trunk height, no contact list, no linear velocity.
    """
    import numpy as _np

    from microduck_local.robots import asimov_walk as _W
    a = _np.asarray(o)
    up = float(-a[slice(*_W.OBS_GRAVITY)][2])
    jp = a[slice(*_W.OBS_JOINT_POS)]
    names = list(_W.JOINT_NAMES)
    elbow = max(abs(float(jp[i])) for i, n in enumerate(names) if "elbow" in n)
    ankle = max(abs(float(jp[i])) for i, n in enumerate(names)
                if "ankle_pitch" in n)
    ok = up > OBS_UPRIGHT and elbow < OBS_ELBOW and ankle < OBS_ANKLE
    state["n"] = state["n"] + 1 if ok else 0
    return state["n"] >= OBS_DWELL


def mcnemar(a, b):
    """Exact two-sided McNemar on two paired boolean sequences.

    The arms see the SAME 24 seeds in the same env, so the marginal counts
    (`stays 22` vs `stays 20`) are not the comparison — the DISCORDANT pairs
    are. A 5.15 -> 6.67 s median in this repo was a null at p = 0.424 on the
    same 72 episodes; see AGENTS.md, "Compare paired".
    """
    from math import comb
    bb = sum(1 for x, y in zip(a, b) if x and not y)
    cc = sum(1 for x, y in zip(a, b) if y and not x)
    n = bb + cc
    if n == 0:
        return bb, cc, 1.0
    k = min(bb, cc)
    pv = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)
    return bb, cc, pv


def _one_sided_p(n):
    """p for n discordant pairs ALL in one direction — the best case at n."""
    from math import comb
    return min(1.0, 2.0 * comb(n, 0) / 2.0 ** n)


def main(runs):
    from pathlib import Path

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs

    def build(run):
        """The env THIS run trained in, from its own `run.json`.

        `train-loop`'s first rule, and this script broke it: it built ONE env
        from hardcoded flags and scored every policy in it. That is how a
        base-locked MOSS policy was once measured turning its chassis 23.7
        deg, which it physically could not do. Reward weights do not move the
        robot, but `push_bw`, `action_clip`, `start_rung` and `getup_label`
        all do — and `limit_cost` has to be ON regardless, because the jam
        gate below is read off `_limit_charge`.
        """
        kw = dict(replay_kwargs(Path("runs") / run))
        kw.pop("task", None)
        # `handoff_pay` ENDS the episode at the handoff — that is a TRAINING
        # device, not a cut the deployment has. Scoring with it on measures
        # the handoff state and calls it the outcome: episodes ran 110 steps
        # instead of 400, the 4 s shove landed in 4 of 24, and `home` was
        # read before Menlo's walker had settled anything. Score at the
        # horizon the deliverable is defined at, with only the deployment's
        # own cuts — the rule this repo has under "Compare paired".
        kw.pop("handoff_pay", None)
        # A DELIBERATE, LOUD override of the posture the policy trained on.
        # Rule 1 of train-loop is "evaluate under the flags it TRAINED with",
        # so changing one has to be stated, not slipped in: this is here to
        # ask what a supine-trained get-up does from the postures a PUSH
        # actually produces (the cache holds 477 across 6 labels; training
        # used one).
        if POSE:
            kw["getup_label"] = POSE
        kw.setdefault("limit_cost", 1.0)
        if not kw.get("limit_cost"):
            kw["limit_cost"] = 1.0
        return S.get("asimov1").env_class("getup")(
            domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)

    per_ep: dict[str, dict[str, list]] = {}

    env = build(runs[0])
    m, d = env.model, env.data
    ADR = np.atleast_1d(getattr(env, "_q_adr", env.joint_qpos_adr))
    HOME = np.asarray(env.robot.default_pose, float)
    TB = env.trunk_body_id
    MASS = float(m.body_subtreemass[TB])

    def run_episode(act, shove=None):
        """shove = (bodyweights, at_second, duration_s) applied to the trunk."""
        o, _ = env.reset(seed=run_episode.seed)
        if getattr(run_episode, "reset_handoff", None):
            run_episode.reset_handoff()
        up_end = rose = False
        herr: list[float] = []
        hmax: list[float] = []
        jam: list[float] = []
        span = 0.0
        shoved_at = None
        for k in range(int(env.max_steps)):
            d.xfrc_applied[:] = 0.0
            if shove is not None:
                g, t0, dur = shove
                if R.is_standing(env) and shoved_at is None and k / 50 >= t0:
                    shoved_at = k
                if (shoved_at is not None
                        and shoved_at <= k < shoved_at + int(dur * 50)):
                    d.xfrc_applied[TB, 0] = g * MASS * 9.81
            o, r, t, tr, _ = env.step(act(o))
            st = R.is_standing(env)
            rose = rose or st
            if st:
                e = np.abs(np.asarray(d.qpos[ADR], float) - HOME)
                herr.append(float(e.mean()))
                hmax.append(float(e.max()))
                # RADIANS, measured here — not `_limit_charge()`, which is
                # that number multiplied by the run's own `limit_cost`. Read
                # through the charge it showed 1.765 for a 0.035 rad jam,
                # because the weight was 50. A gate that compares a REWARD to
                # a threshold in radians is comparing two different things.
                jam.append(float(sum(
                    max(0.0, float(d.qpos[a]) - hi) + max(0.0, lo - float(d.qpos[a]))
                    for a, (lo, hi) in RANGES)))
            span = max(span, env._foot_span())
            up_end = st
            if t or tr:
                break
        d.xfrc_applied[:] = 0.0
        return dict(rose=rose, up_end=up_end, span=span, shoved=shoved_at,
                    herr=np.median(herr) if herr else np.nan,
                    hmax=np.median(hmax) if hmax else np.nan,
                    jam=np.median(jam) if jam else np.nan)

    n_ep = EPISODES
    scale = n_ep / 24.0

    for run in runs:
        env = build(run)
        m, d = env.model, env.data
        ADR = np.atleast_1d(getattr(env, "_q_adr", env.joint_qpos_adr))
        HOME = np.asarray(env.robot.default_pose, float)
        TB = env.trunk_body_id
        MASS = float(m.body_subtreemass[TB])
        RANGES = []
        for a in ADR:
            j = int(np.argmax(m.jnt_qposadr == a))
            lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
            if hi > lo:
                RANGES.append((int(a), (lo, hi)))
        act = load(run)
        # --handoff: the deliverable is the get-up PLUS Menlo's own walker
        # taking over once there is a stand to take over. Scored as one
        # controller, because that is what would ship.
        if HANDOFF:
            from microduck_local.behaviors import asimov_tasks as _T
            _walk = load_onnx(".cache/asimov_1/policy/policy.onnx")
            _getup = act
            _state = {"on": False, "n": 0}

            def act(o, _s=_state, _g=_getup, _w=_walk, _T=_T):
                if not _s["on"]:
                    due = (obs_switch_due(o, _s) if OBS_SWITCH
                           else _T._getup_handoff_due(env))
                    if due:
                        _s["on"] = True
                return (_w if _s["on"] else _g)(o)
            run_episode.reset_handoff = lambda: (_state.__setitem__("on", False),
                                                 _state.__setitem__("n", 0),
                                                 setattr(env, "_handoff_streak", 0),
                                                 setattr(env, "_handoff_prev_xy", None))
        plain, pushed = [], []
        for ep in range(n_ep):
            run_episode.seed = ep
            plain.append(run_episode(act))
            run_episode.seed = ep
            pushed.append(run_episode(act, shove=(0.20, 4.0, 0.15)))
        got = {
            "rises": sum(p["rose"] for p in plain),
            "stays": sum(p["up_end"] for p in plain),
            "home_deg": float(np.degrees(np.nanmedian([p["herr"] for p in plain]))),
            "home_max_deg": float(np.degrees(np.nanmedian([p["hmax"] for p in plain]))),
            "jam_rad": float(np.nanmedian([p["jam"] for p in plain])),
            "span_m": float(np.median([p["span"] for p in plain])),
            "recovers": (sum(p["up_end"] for p in pushed
                                 if p["shoved"] is not None)
                         / max(1, sum(p["shoved"] is not None for p in pushed))),
        }
        landed = sum(p["shoved"] is not None for p in pushed)
        per_ep[run] = {
            "stays": [bool(p["up_end"]) for p in plain],
            "rises": [bool(p["rose"]) for p in plain],
            "recovers": [bool(p["up_end"]) if p["shoved"] is not None else None
                         for p in pushed],
        }
        print(f"\n=== {run} ==="
              + ("   [DEPLOYABLE obs-only switch]" if OBS_SWITCH else "")
              + (f"   [POSE OVERRIDDEN to {POSE!r} — this run TRAINED on "
                 f"{dict(replay_kwargs(Path('runs') / run)).get('getup_label')!r}]"
                 if POSE else ""))
        ok = True
        for key, (gate, what) in GATES.items():
            v = got[key]
            # the two COUNT gates are "22 of 24" and "20 of 24"; at another
            # episode count they are the same FRACTION, not the same count.
            if key in ("rises", "stays"):
                gate = gate * scale
                what = what.replace("of 24", f"of {n_ep}")
            good = v >= gate if key in ("rises", "stays", "recovers") else v <= gate
            ok &= bool(good)
            arrow = "<=" if key not in ("rises", "stays", "recovers") else ">="
            print(f"  [{'PASS' if good else 'FAIL'}] {key:9} {v:8.3f} "
                  f"{arrow} {gate:<7} {what}")
        print(f"  (the shove landed in {landed}/{n_ep} episodes — it is only "
              f"applied once the robot is actually standing)")
        print(f"  VERDICT: {'MEETS THE GOAL' if ok else 'does not meet the goal'}")

    # PAIRED, because every run above saw seeds 0..23 in an identical env.
    # Printing only the marginals invites "22 beats 20" off 2 episodes.
    if len(runs) > 1:
        base = runs[0]
        print(f"\nPAIRED vs {base} (exact McNemar, same {n_ep} seeds):")
        for run in runs[1:]:
            for key in ("stays", "rises"):
                a = per_ep[base][key]
                b = per_ep[run][key]
                bb, cc, pv = mcnemar(a, b)
                verdict = ("NULL — the margin is inside the noise"
                           if pv > 0.05 else "real at p<=0.05")
                print(f"  {key:8} {base[-6:]} {sum(a):3}/{n_ep} vs "
                      f"{run[-6:]} {sum(b):3}/{n_ep} | discordant {bb}-{cc} "
                      f"| p = {pv:.3f} | {verdict}")
                if pv > 0.05:
                    need = next(i for i in range(1, 40)
                                if _one_sided_p(i) <= 0.05)
                    n_d = bb + cc
                    if n_d < need:
                        print(f"           (only {n_d} of {n_ep} episodes "
                              f"disagreed at all, and an exact McNemar needs "
                              f"{need} ALL one-sided to reach p<=0.05 — so "
                              f"this is NO INSTRUMENT, not no effect: raise "
                              f"--episodes)")
                    else:
                        # paired difference and its interval, which is what a
                        # null is actually entitled to say
                        dd = (bb - cc) / n_ep
                        se = (n_d ** 0.5) / n_ep
                        print(f"           ({n_d} of {n_ep} disagreed, so the "
                              f"test had the power: the paired difference is "
                              f"{dd:+.3f}, 95% CI {dd - 1.96 * se:+.3f}…"
                              f"{dd + 1.96 * se:+.3f} — a real null, not a "
                              f"missing instrument)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--handoff" in argv:
        globals()["HANDOFF"] = True
        argv.remove("--handoff")
    for a in list(argv):
        if a.startswith("--episodes="):
            globals()["EPISODES"] = int(a.split("=", 1)[1])
            argv.remove(a)
        elif a.startswith("--pose="):
            globals()["POSE"] = a.split("=", 1)[1]
            argv.remove(a)
        elif a == "--obs-switch":
            globals()["OBS_SWITCH"] = True
            argv.remove(a)
    main(argv or ["teach-asimov_getup_turn-540bc4"])
