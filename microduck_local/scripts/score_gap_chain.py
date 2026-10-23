"""Score a `asimov_getup_gap` stage UNDER ITS OWN FLAGS — prologue included.

`score_narrow_stand` drops `start_phase` and overwrites `getup_label`, which
for this behaviour deletes the whole first half of the episode: the servos'
scripted push-up. Scored that way the policy is handed a pose it never trains
from and the number means nothing. Here the run's recorded kwargs are used as
they stand, so the episode is the one it practised: prone -> servo push-up ->
policy, at the stage's own assist.

Reports `is_standing` (trunk up AND upright AND nothing but feet down, on the
pose that survives) and the PEAK foot span, which is the clause the whole
narrow track exists for — the recipe's bar is 18/24 at a span under 0.40 m.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--gate", type=float, default=0.40)
    ap.add_argument("--assist", type=float, default=None,
                    help="override the stage's assist — a LOUD override, "
                         "because it changes the robot and not the scoring")
    a = ap.parse_args(argv)

    from microduck_local.robots import asimov_getup_ref as R
    from microduck_local.robots import spec as S
    from microduck_local.run_record import replay_kwargs
    sp = importlib.util.spec_from_file_location(
        "_ga", Path(__file__).with_name("getup_acceptance.py"))
    GA = importlib.util.module_from_spec(sp)
    sys.modules["_ga"] = GA
    sp.loader.exec_module(GA)

    print(f"gap chain, own flags (prologue kept), {a.episodes} eps, "
          f"gate {a.gate:g} m")
    print(f"  {'run':<34s}{'assist':>7s}{'stands':>9s}{'narrow':>9s}"
          f"{'med span':>10s}{'peak z':>9s}")
    for run in a.runs:
        kw = dict(replay_kwargs(Path("runs") / run))
        # The TASK picks the env class, so it must not be discarded: this
        # behaviour trains on `getup_ref`, whose env is the one that owns
        # `start_phase` and the scripted prologue. Popping it and building the
        # plain getup env raised on the first kwarg -- which is the loud
        # version of the failure, and better than the quiet one where a
        # non-ref env silently falls back to its default task.
        task = kw.pop("task", None) or "getup_ref"
        if a.assist is not None:
            kw["getup_assist"] = a.assist
        kw.update(fell_cut=False, max_episode_s=8.0)
        env = S.get("asimov1").env_class(task)(
            domain_rand=False, obs_noise=True, action_delay=True, seed=0, **kw)
        pol = GA.load(run)
        st, nar, spans, zs = [], [], [], []
        for s in range(a.episodes):
            o, _ = env.reset(seed=s)
            if hasattr(pol, "reset"):
                pol.reset()
            mx, up, pz = 0.0, False, 0.0
            for _k in range(int(env.max_steps)):
                o, _r, t, tr, _i = env.step(pol(o))
                mx = max(mx, float(env._foot_span()))
                pz = max(pz, float(env.data.xpos[env.trunk_body_id][2]))
                up = R.is_standing(env)
                if t or tr:
                    break
            st.append(up)
            nar.append(bool(up and mx <= a.gate))
            spans.append(mx)
            zs.append(pz)
        print(f"  {run[-32:]:<34s}{kw.get('getup_assist'):>7}"
              f"{sum(st):5d}/{a.episodes}{sum(nar):5d}/{a.episodes}"
              f"{np.median(spans):10.3f}{np.median(zs):9.3f}", flush=True)
    print("\n  the recipe's bar: 18/24 standing at a peak span under 0.40 m")
    print("  fold is trunk 0.511 m; a stand is ~0.64 m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
