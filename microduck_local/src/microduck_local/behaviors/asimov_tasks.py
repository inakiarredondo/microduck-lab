"""Teachable tasks for Menlo's Asimov 1 — `docs/roadmap.md` Track 14.

Shaped like `g1_tasks.py`: the reward lives in an env subclass
(`robots/asimov_env.AsimovGetupEnv`) and `train-walk --robot asimov1 --task
getup` runs it, so the `terms` below are the rows the 🎓 panel SHOWS rather
than callables anything evaluates.
"""

from .core import *  # noqa: F401,F403 — the package's namespace cascade
from .core import Behavior, CurriculumStage, RewardTerm


def _env_owned(env):  # noqa: ARG001 — the env owns the reward
    """Placeholder: Asimov rewards live in the env, not in this library."""
    return 0.0


ASIMOV_GETUP = Behavior(
    id="asimov_getup",
    suggest="get up off the floor",
    emoji="🛌",
    title="Get up (Asimov 1)",
    description="Stand up from a fallen pose and hold it.",
    how_it_learns=(
        "Every practice episode STARTS it on the floor, in one of hundreds of "
        "poses collected by dropping a limp robot and letting it settle. That "
        "is the part that makes this learnable at all: from a sprawl, random "
        "joint commands almost never reach standing, so a get-up reward on a "
        "robot that always starts upright is a reward it never once collects. "
        "It is then paid for the simple geometry of standing — pelvis off the "
        "floor, chest upright, both feet under it — with a bonus for having "
        "all three at once, and charged for twitchy commands and for straining "
        "its arm motors, so it does not learn to hang off its shoulders."
    ),
    keywords=("get up", "get up off the floor", "stand up", "get back up",
              "recover", "fallen over", "pick itself up", "off the ground",
              "getup", "stand up from the floor"),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup"),
    terms=(
        RewardTerm("height", "gets its hips up off the floor", 3.0, _env_owned),
        RewardTerm("upright", "gets its chest pointing up", 3.0, _env_owned),
        RewardTerm("feet", "gets both feet under it, once it is up", 1.0, _env_owned),
        RewardTerm("standing", "bonus for being up, upright and on both feet", 5.0, _env_owned),
        RewardTerm("action_rate_penalty", "charged for twitchy joint commands",
                   0.5, _env_owned, is_penalty=True),
        RewardTerm("arm_effort_penalty", "charged for straining its arm motors",
                   0.02, _env_owned, is_penalty=True),
        RewardTerm("joint_vel_penalty", "charged for fidgeting once it is up",
                   0.002, _env_owned, is_penalty=True),
    ),
    curriculum=(
        CurriculumStage(
            "almost up already", 600_000,
            {"MICRODUCK_ASIMOV_GETUP_BLEND": "0.8",
             "MICRODUCK_ASIMOV_GETUP_ASSIST": "0.5"},
            detail="Starts 80% of the way back to standing with half its "
                   "weight carried, and MEASURED (2026-10-06) this rung does "
                   "not hold: under zero action it starts at 0.625 m — "
                   "standing is 0.619 — and is already toppling, reaching "
                   "0.268 m within a second and 0.034 m by eight. Every rung "
                   "below is the same, because all four are built by "
                   "INTERPOLATING a fallen pose toward standing and an average "
                   "of two poses is not a pose this robot can hold. Prefer "
                   "`asimov_getup_floor`, whose rungs are replay-validated. "
                   "Kept so the four chains trained on it reproduce."),
        CurriculumStage(
            "half way down", 600_000,
            {"MICRODUCK_ASIMOV_GETUP_BLEND": "0.5",
             "MICRODUCK_ASIMOV_GETUP_ASSIST": "0.3"},
            detail="Starts from a half-risen pose (rise 0.41) with most of "
                   "its weight back. Collapses to 0.067 m under zero action."),
        CurriculumStage(
            "from a crouch", 600_000,
            {"MICRODUCK_ASIMOV_GETUP_BLEND": "0.2",
             "MICRODUCK_ASIMOV_GETUP_ASSIST": "0.0"},
            detail="Barely off the floor, and carrying itself. Collapses to "
                   "0.009 m."),
        CurriculumStage(
            "from where it fell", 1_000_000,
            {"MICRODUCK_ASIMOV_GETUP_BLEND": "0.0",
             "MICRODUCK_ASIMOV_GETUP_ASSIST": "0.0"},
            detail="The real thing: the pose the cache recorded, no help. "
                   "This is the only rung whose policy is worth exporting."),
    ),
    default_steps=2_800_000,
    episode_s=8.0,
    success_metric="stands and holds from a supine pose. NOT yet a sourced "
                   "bar: the 100%/1.70 s figures quoted here earlier are "
                   "two COMMUNITY attempts (cyclotron PR #7, and "
                   "openhorizon-labs/asimov1-getup-safefall), not Menlo's, "
                   "and both break the 78/23 contract — the HF one needs a "
                   "5-step observation history (375 inputs)",
)

from .core import _register  # noqa: E402 — after the Behavior is built

_register(ASIMOV_GETUP)


ASIMOV_GETUP_FLOOR = Behavior(
    id="asimov_getup_floor",
    suggest="push itself up with its arms",
    emoji="🤸",
    title="Get up off the floor, replay-validated rungs (Asimov 1)",
    description="Every rung is a pose this robot has been MEASURED to hold for "
                "6 s when replayed the way the env resets it. The arms are "
                "PAID for pushing instead of only charged for straining.",
    how_it_learns=(
        "This exists because the laddered version never practises the task. "
        "Its four rungs are built by INTERPOLATING between a fallen pose and "
        "the standing pose, and an average of two poses is not a pose this "
        "robot can hold: measured under zero action, rung 1 starts at 0.625 m "
        "— standing height is 0.619 — and is already toppling, reaching 0.268 "
        "m within a second and 0.034 m by eight. Every rung collapses the same "
        "way, so each one teaches 'you are mid-collapse from height X, "
        "recover' and none teaches 'finish the motion and hold'.\n\n"
        "The rungs here come from a search that places a pose, commands it, "
        "and requires the trunk still to be there 6 s later — and then REPLAYS "
        "each survivor the way `reset` will, writing the settled state in and "
        "commanding from it. That last check is not a formality: of 55 poses "
        "that passed the search, 11 failed the replay, one of them dropping 36 "
        "cm. 44 rungs survive, spanning 0.146 m to 0.606 m, and 44 of 44 hold "
        "within 3 cm.\n\n"
        "The arms. The one get-up known to work on this model carries over "
        "HALF its body weight on its arms from the floor up to half standing "
        "height, performs the launch there (trunk rising at 1.0 m/s with 0.59 "
        "of its weight on the arms) and then releases them entirely — the legs "
        "only finish it. This env used to only ever CHARGE for that, through "
        "an arm-effort penalty written to stop it hanging off its shoulders. "
        "'Do not hang off your shoulders' and 'never push off the floor' are "
        "different constraints and only the second one was implemented.\n\n"
        "The head. Along a successful rise, head height and trunk height track "
        "almost perfectly (0.98 against 0.97 of standing), so a head term "
        "looks redundant — but that is measured AFTER it is already rising. "
        "Across the recorded fallen poses the two correlate at only r = 0.317, "
        "and in 59 % of them they differ by more than 0.12 of standing with "
        "the head at 0.05 while the trunk is at 0.21: head flat on the floor, "
        "hips already up, and the height term paying full marks for it. It is "
        "paid on a NEW BEST only — a per-step version earned 253.8 of a 396.6 "
        "return while standing 0 of 12, which is run be7733's failure exactly: "
        "a head held high with the hips down is a posture you can hold all "
        "episode.\n\n"
        "One body fact worth knowing before asking it to push off its hands: "
        "the WRISTS carry no collision geometry (a 38 mm stub, contype 0), and "
        "neither does the upper arm. The forearm — `elbow_link`, 144 mm with a "
        "30x114 mm capsule — is the only hand-side part that can touch "
        "anything, which is why the working get-up plants FOREARMS and has one "
        "down 49 % of its first second."
    ),
    keywords=("push itself up with its arms", "push up off the floor",
              "get up with its arms", "stand up using its arms",
              "arm powered getup"),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup"),
    terms=ASIMOV_GETUP.terms + (
        RewardTerm("arm_push", "paid for pushing itself up off the floor with "
                               "an arm", 4.0, _env_owned),
        RewardTerm("head_up", "paid for getting its head up off the floor",
                   60.0, _env_owned),
        RewardTerm("over_clip_penalty", "charged for asking for a command the "
                   "motors cannot be given", 0.2, _env_owned, is_penalty=True),
    ),

    curriculum=(
        # THE KNOWN-GOOD CHAIN, plus exactly one addition.
        #
        # This ordering is not a design, it is the only one measured to work:
        # floor -> arm_top -> feet -> mix took the floor rise rate to 17/36,
        # and the whole of it came from stage 2 (0/36 -> 16/36 in that one
        # stage). Two later redesigns each changed several things at once and
        # each came back 0/36 — a descending ladder of banded mixes, and a
        # mix/low/floor chain built around a 9 cm hole in the stable set. The
        # addition here is stage 4, the bottom band, which did not exist until
        # the rung search stopped discarding everything below 0.15 m.
        CurriculumStage(
            "off the floor", 2_000_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "",
             # No splits. The unconstrained get-up
             # (teach-asimov_getup_clip-79ef4d) rises 288/288 by
             # sprawling its feet a median 0.989 m apart on a 1.2 m
             # robot whose standing stance is 0.215 m. Capping the
             # span costs NO height coverage down to 0.70 m and
             # starts costing below it, so this forbids the sprawl
             # and admits every rung the ladder needs.
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.75"},
            detail="Starts where it fell. Reaches 0.203 m and no further on "
                   "its own — but it is what gives the chain its NORMALIZER, "
                   "and that is why it goes first: a warm start freezes "
                   "VecNormalize, and the floor's own states reach |z| 13.1 "
                   "under floor-trained stats against 8.4 and 3.9 for the "
                   "rungs. A chain that starts high and descends hands its "
                   "last stage inputs that mean something else."),
        CurriculumStage(
            "finish from high up", 2_000_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "arm_top",
             # No splits. The unconstrained get-up
             # (teach-asimov_getup_clip-79ef4d) rises 288/288 by
             # sprawling its feet a median 0.989 m apart on a 1.2 m
             # robot whose standing stance is 0.215 m. Capping the
             # span costs NO height coverage down to 0.70 m and
             # starts costing below it, so this forbids the sprawl
             # and admits every rung the ladder needs.
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.75"},
            detail="THE stage. Starts on the 20 rungs between 0.231 and 0.280 "
                   "m — the top of the arm plateau — and this alone took the floor rise "
                   "rate from 0/36 to 16/36. Starting the policy LOW taught it "
                   "nothing in two separate chains; the rise is learned from "
                   "the goal end. The band is an explicit height and not a "
                   "quantile, because as a quantile it silently became 10 "
                   "rungs at 0.355-0.440 the moment the cache was "
                   "re-validated — and that explicit height is why it survived "
                   "2026-10-07 rebuild LOUDLY rather than silently: building the "
                   "cache at the clip training actually uses dropped the arm "
                   "plateau's top from 0.440 m to 0.280 m, `>= 0.29` matched "
                   "nothing, and the env raised instead of quietly handing "
                   "this stage the bottom of the plateau."),
        CurriculumStage(
            "catch the crouch", 1_000_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "feet",
             # No splits. The unconstrained get-up
             # (teach-asimov_getup_clip-79ef4d) rises 288/288 by
             # sprawling its feet a median 0.989 m apart on a 1.2 m
             # robot whose standing stance is 0.215 m. Capping the
             # span costs NO height coverage down to 0.70 m and
             # starts costing below it, so this forbids the sprawl
             # and admits every rung the ladder needs.
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.75"},
            detail="The feet-only rungs, upright with both feet down. Held the "
                   "rise at 16/36 and lifted the peak from 0.232 to 0.456 m "
                   "when this band was 5 rungs at 0.471-0.606 m; at the real "
                   "clip it is 78 rungs from 0.416 m, 71 of them splayed. The "
                   "splay is worth 113 mm here (square hips bottom out at "
                   "0.529). It does NOT close the gap: the arm plateau tops at "
                   "0.280, so 136 mm separates the two and the handover is "
                   "still the unlearned move."),
        CurriculumStage(
            "down to the ground", 1_000_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "low",
             # No splits. The unconstrained get-up
             # (teach-asimov_getup_clip-79ef4d) rises 288/288 by
             # sprawling its feet a median 0.989 m apart on a 1.2 m
             # robot whose standing stance is 0.215 m. Capping the
             # span costs NO height coverage down to 0.70 m and
             # starts costing below it, so this forbids the sprawl
             # and admits every rung the ladder needs.
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.75"},
            detail="THE ADDITION, and the only untested part of this chain. 56 "
                   "rungs at 0.134-0.280 m — which at the real clip is the "
                   "WHOLE arm plateau, not just its bottom. The validated rungs sit at 0.146 "
                   "and 0.161 and then nothing until 0.252, while a "
                   "from-scratch policy stalls at 0.203 — past both low rungs "
                   "and below the next foothold. This is the only stage that "
                   "starts it inside that 9 cm hole, and it comes AFTER the "
                   "rise is learned rather than before it."),
        CurriculumStage(
            "all of it", 2_000_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "mix",
             # No splits. The unconstrained get-up
             # (teach-asimov_getup_clip-79ef4d) rises 288/288 by
             # sprawling its feet a median 0.989 m apart on a 1.2 m
             # robot whose standing stance is 0.215 m. Capping the
             # span costs NO height coverage down to 0.70 m and
             # starts costing below it, so this forbids the sprawl
             # and admits every rung the ladder needs.
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.75"},
            detail="Half from where it fell, half over every rung. The best "
                   "floor number so far (17/36) came from a stage of this "
                   "shape, and it is the one whose policy is worth exporting. "
                   "verify_run.py reports the floor-only DELIVERABLE "
                   "separately, because a mixed stage flatters itself: the "
                   "same policy scored 27/36 on this env and 19/36 from the "
                   "floor."),
    ),
    default_steps=8_000_000,
    episode_s=8.0,
    success_metric="rises from a RECORDED fallen pose (not an interpolated "
                   "one) and holds 8 s, scored over the pose cache with obs "
                   "noise and the 20 ms action delay on. Nothing here has run "
                   "on an Asimov.",
)

_register(ASIMOV_GETUP_FLOOR)

# ---------------------------------------------------------------------------

ASIMOV_GETUP_HANDOVER = Behavior(
    id="asimov_getup_handover",
    suggest="push off its arms and stand",
    emoji="\U0001f91d",
    title="The handover: push off the arms and stand (Asimov 1)",
    description="The bottom of the get-up is SCRIPTED, not learned. This "
                "trains only the one transition above it.",
    how_it_learns=(
        "Nine chains have tried to learn a whole get-up here and the staged "
        "ones all die in the same place. The measurement that reframes it: "
        "the arm plateau TRAVERSES OPEN LOOP. Ramping the position servos "
        "from the fallen pose up through seven measured rungs carries the "
        "robot 0.136 -> 0.280 m with no policy at all, at every tempo from "
        "0.15 s to 0.90 s a leg. So the bottom half does not need to be "
        "learned. It needs to be PLAYED.\n\nWhat is left is one transition. "
        "The scripted ramp hands over at 0.256 m on forearms and feet, and "
        "the feet-only plateau does not begin until 0.416 m — a 136 mm gap "
        "with nothing holdable in it, which is why no reverse curriculum can "
        "put a rung there. The one get-up known to work crosses it as a "
        "LAUNCH: between 0.35 and 0.50 of standing its arms carry 0.594 of "
        "body weight with the trunk rising at 1.0 m/s, and above 0.5 the arm "
        "load drops to 0.037 and the legs finish it. That push-off is the "
        "whole job here, which is why `arm_push` is paid and arm strain is "
        "charged only 0.02.\n\nAnd it is NOT the splits. The reference every "
        "working policy was cloned from sprawls its feet a median 0.929 m "
        "apart on a 1.2 m robot; an episode here ends the moment the span "
        "passes 0.70 m, which admits every rung in the ladder and forbids "
        "the scramble.\n\nThe floor to beat is measured: from the handover, "
        "doing NOTHING rises 0/24 and sits at 0.256 m."
    ),
    keywords=("push off its arms and stand", "push off the arms",
              "the handover", "launch off the forearms",
              "stand up from the quadruped", "finish the get-up"),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup"),
    terms=tuple(ASIMOV_GETUP_FLOOR.terms),
    curriculum=(
        CurriculumStage(
            "push off and up", 4_000_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "handover",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.75"},
            detail="Every episode starts where the scripted ramp leaves the "
                   "robot: 0.256 m, forearms and feet down, essentially at "
                   "rest (median -0.021 m/s). It has to generate the launch "
                   "itself. Zero action scores 0/24 here, so any rise is a "
                   "real one."),
        CurriculumStage(
            "catch it and hold", 2_000_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "feet",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.75"},
            detail="The feet-only plateau, 0.416-0.606 m, upright on both "
                   "feet. Scored at the HANDOVER start as well as its own — "
                   "a rung trained only on its newest start is what once "
                   "collapsed this robot's hold from 7.76 s to 0.24 s while "
                   "every per-rung score still read 12/12."),
    ),
    default_steps=6_000_000,
    episode_s=8.0,
    success_metric="rises from the scripted ramp's handover at 0.256 m, "
                   "against zero action's 0/24 — and does it with a foot "
                   "span under 0.70 m, where the cloned get-up sprawls 0.989",
)

_register(ASIMOV_GETUP_HANDOVER)



# ---------------------------------------------------------------------------
# RECONSTRUCTED 2026-10-06. The source of these three recipes was destroyed by
# a regex that matched the WRONG `curriculum=(` block — this file holds several
# near-identical ones and the pattern anchored on a `default_steps` value that
# belonged to a different recipe, so it ate everything in between. The file was
# untracked, so git had nothing to restore.
#
# Rebuilt from three sources, none of them the source file:
#   * the RUNNING lab's `GET /behaviors`, for every title, description,
#     how-it-learns, term, success metric and stage label/steps/detail;
#   * the RUN RECORDS' `env_kwargs`, for each stage's knobs — which the API
#     does not expose (`asimov_getup_clip` walks `track_tol` 0.5/0.35/0.25,
#     `asimov_getup_ref` walks `start_phase` cumulatively through the waypoint
#     ladder);
#   * a stale `.pyc`, for the two recipes that predated it.
#
# The prose and the stage ladders are the originals. The stage ENV dicts are
# reconstructed from what the runs recorded, so a key a stage set but no run
# ever exercised could be missing. Anything load-bearing is covered by
# `tests/test_teach_robots.py` and the reconstruction was checked field by
# field against the captured JSON.
#
# The lesson, which cost three recipes: never pattern-match a block in a file
# that repeats the block's shape. Anchor on the unique `id=`, or edit by line
# range after reading it. And verify a scripted edit landed before moving on —
# the repair for this was itself wrong and dropped three more recipes.
# ---------------------------------------------------------------------------

ASIMOV_GETUP_CLIP = Behavior(
    id="asimov_getup_clip",
    suggest="imitate the get-up",
    emoji='🎬',
    title='Imitate the get-up (Asimov 1)',
    description="Track a get-up that WORKS, harvested from a published "
                "policy running in this repo's own MuJoCo.",
    how_it_learns=(
        "Every earlier attempt tracked an authored get-up that somebody "
        "drafted by hand, and the route it describes has been measured "
        "impassable: six independent optimisers and about 50 million "
        "training steps all stop in the same place, and the one leg "
        "they cannot do needs the trunk to come up 62 degrees where the "
        "best search found 14.\n\nThis one tracks a get-up that "
        "actually happens. openhorizon-labs/asimov1-getup-safefall is "
        "an Apache-2.0 policy on Hugging Face, and run on THIS repo's "
        "own copy of Menlo's model, in this repo's own MuJoCo, it "
        "stands up 20 times out of 20 with a median of 1.22 seconds. "
        "That retires every conclusion here that the robot cannot do "
        "it: those were about the route and the method, not the "
        "machine. Its joint trajectories are recorded by "
        "scripts/harvest_getup_clips.py and this is the reward for "
        "matching them.\n\nTwo things make this different from tracking "
        "a draft. The reference is REACHABLE — replaying the "
        "demonstration's own joints through the tracker scores a "
        "perfect 10 of 10, where the authored reference asked for joint "
        "angles the actuators cannot command by up to 1.65 rad. And it "
        "has a real TEMPO: the demonstrations run 2.7 to 3.4 seconds, "
        "so the reference is indexed by time rather than by hitting a "
        "waypoint, and the policy is asked to be where the "
        "demonstration was at that moment.\n\nWhat the practice "
        "tightens is not the pay but the PATIENCE: an episode ends when "
        "the robot drifts too far from the demonstration, and each "
        "stage allows less drift than the last. On the G1's kick the "
        "imitation reward alone changed nothing and that ladder took it "
        "from 10 cm to 62 cm."
    ),
    keywords=("imitate the get-up", "copy the get-up", "track the get-up",
              "imitate a get-up", "follow the demonstration"),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup_clip"),
    terms=(
        RewardTerm('track', 'matches the reference pose for this phase',
                   10.0, _env_owned),
        RewardTerm('up', 'is up, upright and on both feet',
                   8.0, _env_owned),
        RewardTerm('action_rate_penalty', 'charged for twitchy joint commands',
                   0.5, _env_owned, is_penalty=True),
    ),
    curriculum=(
        CurriculumStage(
            'follow loosely', 1_500_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.6',
                     'MICRODUCK_ASIMOV_GETUP_ACTION_CLIP': '13.0',
                     # the only run on this task that stands (teach-asimov_getup_clip-0e31e5)
                     # trained with the 1-step command lag ON, and the recipe did
                     # not set it — so a relaunch silently trained a different env.
                     # WITHOUT it that same policy scores 15/36 instead of 26/36.
                     'MICRODUCK_ASIMOV_GETUP_LAG': '1',
                    },
            detail="The widest cut, so episodes survive long enough to "
                   "be paid at all. The policy starts in the "
                   "demonstration's own settled fallen pose, so "
                   "tracking error begins at zero and grows — which is "
                   "what makes a drift cut the right curriculum knob "
                   "here."),
        CurriculumStage(
            'follow closely', 1_500_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.5',
                     'MICRODUCK_ASIMOV_GETUP_ACTION_CLIP': '13.0',
                     # the only run on this task that stands (teach-asimov_getup_clip-0e31e5)
                     # trained with the 1-step command lag ON, and the recipe did
                     # not set it — so a relaunch silently trained a different env.
                     # WITHOUT it that same policy scores 15/36 instead of 26/36.
                     'MICRODUCK_ASIMOV_GETUP_LAG': '1',
                    },
            detail="Half the drift allowed."),
        CurriculumStage(
            'follow tightly', 1_500_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.35',
                     'MICRODUCK_ASIMOV_GETUP_ACTION_CLIP': '13.0',
                     # the only run on this task that stands (teach-asimov_getup_clip-0e31e5)
                     # trained with the 1-step command lag ON, and the recipe did
                     # not set it — so a relaunch silently trained a different env.
                     # WITHOUT it that same policy scores 15/36 instead of 26/36.
                     'MICRODUCK_ASIMOV_GETUP_LAG': '1',
                    },
            detail="Tighter again. This is the rung the G1 kick's "
                   "equivalent ladder was won on."),
        CurriculumStage(
            'hold the line', 1_500_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.25',
                     'MICRODUCK_ASIMOV_GETUP_ACTION_CLIP': '13.0',
                     # the only run on this task that stands (teach-asimov_getup_clip-0e31e5)
                     # trained with the 1-step command lag ON, and the recipe did
                     # not set it — so a relaunch silently trained a different env.
                     # WITHOUT it that same policy scores 15/36 instead of 26/36.
                     'MICRODUCK_ASIMOV_GETUP_LAG': '1',
                    },
            detail="Only this rung's policy is worth exporting."),
    ),
    default_steps=6_000_000,
    episode_s=8.0,
    success_metric="stands from the demonstration's own fallen poses. "
                   "The bar is theirs: torso up > 0.9 and pelvis > 0.55 "
                   "m held 2 s, which their policy makes 20/20 at a "
                   "median 1.22 s in this same model",
)

_register(ASIMOV_GETUP_CLIP)

# ---------------------------------------------------------------------------


def _mixed_clips() -> str:
    """Absolute path: the trainer is a SUBPROCESS and a relative one would
    resolve against whatever directory it happened to start in."""
    from ..robots import asimov as _a
    return str(_a.CACHE_DIR / "getup_clips_mixed.npz")


ASIMOV_GETUP_HOLD = Behavior(
    id="asimov_getup_hold",
    suggest="hold the stand after getting up",
    emoji='\U0001f9cd',
    title='Get up AND STAY up (Asimov 1)',
    description="Fine-tune the get-up that works on clips that START where it "
                "actually falls — the moment it first stands.",
    how_it_learns=(
        "The get-up is solved; the HOLD is not. "
        "teach-asimov_getup_clip-0e31e5 rises 34 times out of 36 and is "
        "still standing at the buzzer only 26 — **29 of 36 episodes "
        "stand and then fall**. The cause is measured and it is not "
        "reward or torque: the clone's action error is 0.000702 rad^2 "
        "on the rise against 0.000064 on the hold, so it is faithful on "
        "the teacher's states and fails on its OWN. It arrives upright "
        "in configurations the demonstration never visited, and nothing "
        "in its training ever showed it what to do there.\n\nSo this "
        "trains on those configurations. Every episode of the deployed "
        "policy is rolled out, the pose at the instant it first stands "
        "is recorded, and the teacher is asked what IT would do from "
        "exactly there. That the question has an answer is the "
        "measurement this rests on: the teacher holds from the "
        "student's own post-stand states in **89 of 89** of them. Those "
        "states are recoverable; the policy simply does not know "
        "how.\n\nThe clip set is MIXED on purpose — 24 original "
        "fallen starts plus 89 post-stand starts. Training on the "
        "post-stand ones alone would buy the hold and lose the rise, "
        "which is exactly how a five-rung curriculum here once collapsed "
        "its hold from 7.76 s to 0.24 s while every per-rung score still "
        "read 12 out of 12."
    ),
    keywords=("stay up after getting up", "hold the stand", "stop falling "
              "after standing", "get up and stay up"),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup_clip"),
    terms=(
        RewardTerm('track', 'matches the reference pose for this phase',
                   10.0, _env_owned),
        RewardTerm('up', 'is up, upright and on both feet',
                   8.0, _env_owned),
        RewardTerm('action_rate_penalty', 'charged for twitchy joint commands',
                   0.5, _env_owned, is_penalty=True),
    ),
    curriculum=(
        CurriculumStage(
            'learn the catch', 1_500_000,
            {
                'MICRODUCK_ASIMOV_GETUP_CLIPS': _mixed_clips(),
                # the donor trained with the 1-step command lag
                # on, and WITHOUT it the donor itself drops
                # 26/36 to 15/36. Leaving it unset made the
                # first run of this recipe a two-variable
                # experiment and its result uninterpretable.
                'MICRODUCK_ASIMOV_GETUP_LAG': '1',
                'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.35',
                'MICRODUCK_ASIMOV_GETUP_ACTION_CLIP': '13.0',
            },
            detail="Both clip sets, with the drift cut opened a little so "
                   "the post-stand episodes survive long enough to be paid "
                   "at all. 13.0 is NOT optional: the teacher exceeds the "
                   "spec's 4.0 clip on 15.5 % of its frames, and the only "
                   "run on this task that stands used 13.0."),
        CurriculumStage(
            'hold the line', 1_500_000,
            {
                'MICRODUCK_ASIMOV_GETUP_CLIPS': _mixed_clips(),
                # the donor trained with the 1-step command lag
                # on, and WITHOUT it the donor itself drops
                # 26/36 to 15/36. Leaving it unset made the
                # first run of this recipe a two-variable
                # experiment and its result uninterpretable.
                'MICRODUCK_ASIMOV_GETUP_LAG': '1',
                'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.25',
                'MICRODUCK_ASIMOV_GETUP_ACTION_CLIP': '13.0',
            },
            detail="The donor's own tolerance. Only this rung's policy is "
                   "worth exporting — and score it with verify_run, which "
                   "reads the deterministic export and never ep_rew."),
    ),
    default_steps=3_000_000,
    episode_s=8.0,
    success_metric="beats teach-asimov_getup_clip-0e31e5's 26/36 still up "
                   "at the buzzer, on the same 36 episodes, paired. At 36 "
                   "episodes the resolution is about 23 points, so a "
                   "smaller gap is not a result",
)

_register(ASIMOV_GETUP_HOLD)

ASIMOV_GETUP_REF = Behavior(
    id='asimov_getup_ref',
    suggest='get up by reference',
    emoji='🪜',
    title='Get up by reference (Asimov 1)',
    description="Track an authored get-up: prone, forearms, quadruped, "
                "pike, rock back onto the feet, stand.",
    how_it_learns=(
        "Each stage of practice KEEPS every starting position it has "
        "already learned, and adds one. That sounds fussy and it is the "
        "single biggest thing that was wrong: training each stage only "
        "on its newest, harder start made the robot forget. Measured "
        "across one 2.7 million-step run, scoring every stage from the "
        "same easy position, the time it could stay standing went 7.8 "
        "seconds, 3.3, 0.4, 0.3, 0.2 — a 94 % collapse by the third "
        "stage, and it never came back. It could always still GET to "
        "standing; it just could not stay there any more, because after "
        "the first stage it never practised standing again.\n\nThe "
        "first four attempts paid it for pelvis height, chest "
        "uprightness and feet on the floor, and all four stopped at the "
        "first phase. The reason turned out to be the robot and not the "
        "reward: Asimov has NO HANDS — its wrists carry no collision "
        "geometry at all — so the only way it can push itself up is on "
        "its FOREARMS, which means rolling face down first. Chest-"
        "uprightness therefore goes +0.01 down to -0.37 and only then "
        "up to +1.00, and pelvis height dips as it loads its elbows. "
        "Every reward paying either one per step was paying it to skip "
        "the phases that lift it.\n\nSo this one pays something else: "
        "how closely it matches an authored get-up — the right joints "
        "and the right parts touching the floor, for the phase it is in "
        "— with a bonus each time it reaches the next phase and never "
        "for the same phase twice. The phase advances when it gets "
        "there, not on a clock, so a slow get-up is fine and a skipped "
        "one pays nothing. And the episode ENDS when it drifts too far "
        "from the reference, which is the part that does the work: "
        "tightening that tolerance is the curriculum.\n\nRamping the "
        "motors through this reference with no policy at all, the robot "
        "already does every phase except ONE — rocking back from the "
        "pike onto its feet. That single transition is what it is here "
        "to learn, which is why practice starts at the pike and works "
        "backwards."
    ),
    keywords=('get up by reference', 'get up by the waypoints', 'follow the authored get-up', 'get up step by step'),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup_ref"),
    terms=(
        RewardTerm('track', 'matches the reference pose for this phase',
                   10.0, _env_owned),
        RewardTerm('contact', 'the right parts are touching the floor',
                   2.0, _env_owned),
        RewardTerm('advance', 'reached the next phase, paid once',
                   30.0, _env_owned),
        RewardTerm('up', 'is up, upright and on both feet',
                   8.0, _env_owned),
        RewardTerm('action_rate_penalty', 'charged for twitchy joint commands',
                   0.5, _env_owned, is_penalty=True),
    ),
    curriculum=(
        CurriculumStage(
            'stand up from the fold', 300_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.6',
                    },
            detail="The only rung measured to WORK, and now trained "
                   "against the COMMAND DELAY on every episode rather "
                   "than half of them. At the inherited 50/50 it "
                   "learned a controller with no delay margin: two "
                   "independent chains both held the full 7.76 s on "
                   "every episode WITHOUT the delay and fell at 1.3-2.0 "
                   "s on every episode WITH it, perfectly separated, no "
                   "exceptions. One control step is 20 ms -- the CAN-"
                   "bus latency Menlo's own randomisation models -- so "
                   "that stand does not ship. From a single 78-float "
                   "frame the policy cannot tell which episode it is "
                   "in, so it has to use one controller for both and "
                   "settled on the one that is optimal with no delay; "
                   "forcing the delay removes the choice, and a "
                   "controller with margin at 20 ms is stable at 0. "
                   "from `fold` (pelvis 0.517 m, on its feet, trunk 33 "
                   "deg forward) the servos alone reach a held stand — "
                   "0.608 m, uprightness +1.00, four seconds — once the "
                   "leg timings are 0.25 s rather than the 0.60 s the "
                   "reference shipped. Starting here means the policy "
                   "learns the finish on a rung it can actually "
                   "complete, instead of warm-starting every later rung "
                   "from something that has never stood."),
        CurriculumStage(
            'catch it on the way up', 400_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold,rock_back',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.6',
                    },
            detail="FREE from here down: the reference is dropped for "
                   "the intermediates and the pay is the product "
                   "`pelvis height x uprightness`, counted only while a "
                   "foot is on the floor. Five chains were spent asking "
                   "the policy to TRACK the authored route through this "
                   "gap, and that route has since been measured "
                   "impossible — `com - foot` is 0.618 m at the pike "
                   "against 0.114 at `fold`, it does not move while "
                   "four contacts are planted, and the trunk cannot "
                   "rotate up first. A tracking reward against an "
                   "unreachable waypoint does not merely fail to help, "
                   "it CHARGES the policy for trying anything else. The "
                   "waypoints still serve as the `advance` ladder, so "
                   "the structure is kept and only the imitation is let "
                   "go. Starts at the gap state itself — pelvis 0.377 "
                   "m, on its FEET, trunk folded horizontal — and has "
                   "only to catch itself and come up. This is the "
                   "reverse curriculum done with a REAL rung: the state "
                   "is recorded off `fold` falling forward, not "
                   "interpolated, so it is a configuration the body "
                   "demonstrably passes through. It is not an "
                   "equilibrium, so the episode places it and goes "
                   "rather than settling — settling it just lets it "
                   "fall to 0.206 m, which is abandoning the rung, not "
                   "reaching it."),
        CurriculumStage(
            'the one hard move', 500_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold,rock_back,pike',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.6',
                    },
            detail="One phase earlier: the pike — hips high, weight on "
                   "the feet and forearms, pelvis 0.300 m. This is the "
                   "only leg of the reference the motors cannot do "
                   "unaided at ANY timing: ramped through it at every "
                   "speed from 0.08 s to 0.60 s the pelvis never once "
                   "rises above the pike's own height. Everything after "
                   "it they can, now that the tail's timing is measured "
                   "rather than guessed."),
        CurriculumStage(
            'from the quadruped', 300_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold,rock_back,pike,quadruped',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.6',
                    },
            detail="One phase earlier: on forearms and knees, pelvis "
                   "0.211 m. The motors already make this leg unaided, "
                   "so a policy that can do the rock-back should pick "
                   "it up nearly free."),
        CurriculumStage(
            'knees under the hips', 250_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold,rock_back,pike,quadruped,draw_knees_in',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.6',
                    },
            detail="Earlier again, and a SMALL step on purpose. The "
                   "first chain went straight from the quadruped to the "
                   "pose cache and lost everything — it had reached "
                   "`rock_back` from the quadruped and went back to "
                   "achieving nothing at all. That jump moved the start "
                   "two waypoints AND swapped one authored pose for a "
                   "distribution of 220 real ones, in a single rung."),
        CurriculumStage(
            'forearms down', 250_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold,rock_back,pike,quadruped,draw_knees_in,plant_forearms',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.7',
                    },
            detail="Flat on the floor with the elbows coming in under "
                   "the shoulders, pelvis 0.020 m. The last authored "
                   "rung before the cache."),
        CurriculumStage(
            'from where it fell', 400_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold,rock_back,pike,quadruped,draw_knees_in,plant_forearms,prone',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.9',
                    },
            detail="Face down, from the cache — 220 real settled poses "
                   "rather than one authored one. The tolerance is 0.90 "
                   "here and not 0.60 for a measured reason: a cache "
                   "pose sits 0.78 rad of mean joint error from the "
                   "first waypoint, so at 0.60 every episode is cut on "
                   "its FIRST step for being a different fallen robot."),
        CurriculumStage(
            'hold the line', 300_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold,rock_back,pike,quadruped,draw_knees_in,plant_forearms,prone',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.65',
                    },
            detail="The same rung with the not-tracking cut tightened. "
                   "This is the ladder that matters — on the G1 kick "
                   "the reward alone did nothing and the termination "
                   "ladder took it from 0.10 m to 0.62 m. Only this "
                   "rung's policy is worth exporting."),
    ),
    default_steps=2_700_000,
    episode_s=8.0,
    success_metric="reaches the `stand` waypoint from the prone cache. "
                   "The 100 %/1.70 s figures once quoted here are two "
                   "COMMUNITY attempts, not Menlo's, and both break the "
                   "contract — so there is no published bar for a 78/23 "
                   "get-up, which is the whole point of doing one",
)

_register(ASIMOV_GETUP_REF)

# ---------------------------------------------------------------------------

ASIMOV_GETUP_LEG = Behavior(
    id="asimov_getup_leg",
    suggest="rock up off the forearms onto the feet",
    emoji="\U0001fa9c",
    title="The one leg the servos cannot do (Asimov 1)",
    description="Everything below pike plays open loop. This trains ONLY the "
                "rock-back onto the feet, and never leaves a narrow stance.",
    how_it_learns=(
        "The authored get-up is already the narrow, human-looking route — "
        "prone, forearms down, knees under the hips, quadruped, pike — and "
        "EVERY waypoint in it sits at a 0.215 m foot span, exactly the width "
        "of standing. The position servos play all of it with no policy at "
        "all, up to `pike` at 0.300 m. So none of that needs to be "
        "learned.\n\nOne leg does. `pike -> rock_back` asks the robot to "
        "lift both forearms off the floor and rock onto its feet, and open "
        "loop it has now failed three separate ways: straight interpolation "
        "at 25 tempo combinations from 0.10 s to 0.60 s a leg, every one "
        "collapsing to the floor; and an authored COUNTER-MOVEMENT (load the "
        "arms, then extend them fast) across 27 variants, whose best bought "
        "23 mm and whose arm-load term changed the peak by nothing at all. "
        "The reverse is free — shove `fold` forward and it drops onto its "
        "forearms in 0.12 s — so the move is momentum-dependent in a way a "
        "position ramp cannot supply.\n\nThat is what a policy is for, and "
        "it is a far smaller ask than a get-up: one transition, from a pose "
        "measured to hold, with the finish above it rehearsed in the same "
        "start set so it cannot be forgotten. The floor to beat is measured: "
        "from pike, doing NOTHING rises 0/24 and peaks at its own 0.263 m "
        "reset height."
    ),
    keywords=("rock up off the forearms onto the feet",
              "rock back onto the feet", "the one hard move",
              "lift the forearms", "pike to standing", "train just the leg"),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup_ref"),
    terms=tuple(ASIMOV_GETUP_REF.terms),
    curriculum=(
        CurriculumStage(
            "learn the finish", 1_500_000,
            {"MICRODUCK_ASIMOV_GETUP_START": "fold,rock_back",
             "MICRODUCK_ASIMOV_GETUP_TRACK_TOL": "0.6",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.75"},
            detail="Above the hard leg, where the servos already work — so "
                   "the policy owns the finish before it is asked to reach "
                   "it. Both starts, not just the newest."),
        CurriculumStage(
            "the pike leg", 3_000_000,
            {"MICRODUCK_ASIMOV_GETUP_START": "fold,rock_back,pike",
             "MICRODUCK_ASIMOV_GETUP_TRACK_TOL": "0.6",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.75"},
            detail="THE stage. `pike` joins the start SET rather than "
                   "replacing it: training a rung only on its newest start is "
                   "what once collapsed this robot's hold from 7.76 s to "
                   "0.24 s while every per-rung score still read 12/12. The "
                   "span cut keeps it out of the splits the cloned get-up "
                   "uses — that one sprawls a median 0.989 m on a 1.2 m "
                   "robot, and every pose here stays at 0.215."),
    ),
    default_steps=4_500_000,
    episode_s=8.0,
    success_metric="rises from pike, where zero action scores 0/24 and peaks "
                   "at 0.263 m — and does it inside a 0.75 m foot span",
)

_register(ASIMOV_GETUP_LEG)

# ---------------------------------------------------------------------------

ASIMOV_GETUP_CATCH = Behavior(
    id="asimov_getup_catch",
    suggest="catch itself falling and stand back up",
    emoji="\U0001f9ff",
    title="Catch the fall: a reverse curriculum in PERTURBATION space (Asimov 1)",
    description="Shove a standing fold toward pike and learn to recover, then "
                "grow the shove. Every start state is reachable by construction.",
    how_it_learns=(
        "The one leg of the narrow get-up — `rock_back` (0.377 m) up to "
        "`fold` (0.517 m) — has failed every way it has been asked. Open "
        "loop at 25 tempos from 0.10 to 0.60 s a leg: all collapse. An "
        "authored counter-movement across 27 variants: best +23 mm, and the "
        "arm-load term moved the peak by nothing. RL trained directly on it: "
        "**0/16**, while the SAME policy stands 16/16 and holds 16/16 from "
        "`fold` — so the harness is sound and the leg is the wall. And it is "
        "not torque: at 5x arm effort the result is unchanged.\n\nWhat "
        "decides it is the contact mode, and the contact mode is decided by "
        "momentum: `fold`'s joint angles ALSO describe the face-down "
        "collapse, 0.05 rad apart, pelvis 0.516 against 0.197.\n\nSo stop "
        "placing poses and use the fall. Shove a settled `fold` forward and "
        "it drops through the whole gap in a second — 0.517 to 0.491 to "
        "0.423 to 0.284 — so those states are reachable BY CONSTRUCTION and "
        "carry the velocity they really have. Three interpolated pose ladders "
        "failed because their rungs were averages of two poses and fell on "
        "their own; a state the robot demonstrably passes through cannot "
        "have that defect. The curriculum grows the shove instead of lowering "
        "a pose.\n\nThe floor is measured: from these starts, doing nothing "
        "stays up **0 of 24 at every depth**."
    ),
    keywords=("catch the fall", "recover from a shove", "stop falling over",
              "reverse curriculum", "catch itself"),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup"),
    terms=tuple(ASIMOV_GETUP_FLOOR.terms),
    curriculum=(
        CurriculumStage(
            "fall depth 0.25", 1500000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "falling",
             "MICRODUCK_ASIMOV_GETUP_FALL_DEPTH": "0.25",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.75"},
            detail="Shallow: the shove has barely started and the robot is still near `fold`, where this policy family already stands. The recovery here is almost the finish it owns. The depth is a CUT, not a replacement: every "
                   "shallower state stays in the set, so the rung cannot "
                   "forget what it already learned."),
        CurriculumStage(
            "fall depth 0.50", 1500000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "falling",
             "MICRODUCK_ASIMOV_GETUP_FALL_DEPTH": "0.5",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.75"},
            detail="Half way down. The states now carry real downward speed, up to -0.68 m/s. The depth is a CUT, not a replacement: every "
                   "shallower state stays in the set, so the rung cannot "
                   "forget what it already learned."),
        CurriculumStage(
            "fall depth 0.75", 1500000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "falling",
             "MICRODUCK_ASIMOV_GETUP_FALL_DEPTH": "0.75",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.75"},
            detail="Deep enough that the trunk is below `rock_back`'s 0.377 m, which is where every static-start attempt died. The depth is a CUT, not a replacement: every "
                   "shallower state stays in the set, so the rung cannot "
                   "forget what it already learned."),
        CurriculumStage(
            "fall depth 1.00", 2000000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "falling",
             "MICRODUCK_ASIMOV_GETUP_FALL_DEPTH": "1.0",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.75"},
            detail="The whole fall, down to 0.318 m and -1.39 m/s — effectively `pike`, but ARRIVED AT rather than placed. Only this rung's policy is worth exporting. The depth is a CUT, not a replacement: every "
                   "shallower state stays in the set, so the rung cannot "
                   "forget what it already learned."),
    ),
    default_steps=6_500_000,
    episode_s=8.0,
    success_metric="stays UP at the buzzer from the deep end of the fall, "
                   "where zero action scores 0/24 at every depth — and does "
                   "it inside a 0.75 m foot span",
)

_register(ASIMOV_GETUP_CATCH)

# ---------------------------------------------------------------------------

ASIMOV_GETUP_HYBRID = Behavior(
    id="asimov_getup_hybrid",
    suggest="stand up from the pike, sprawling only for the launch",
    emoji="\u2696",
    title="The hybrid: narrow everywhere, sprawl only in the launch (Asimov 1)",
    description="Keep the 0.215 m stance the authored route uses, and permit "
                "a wide base ONLY while crossing the 136 mm ballistic gap.",
    how_it_learns=(
        "Two measured facts set this up. First, the authored get-up is "
        "already the narrow human route and the position servos play it with "
        "NO policy from `prone` to `pike` — 0.007 to 0.300 m, every waypoint "
        "at a 0.215 m foot span, the same width as standing. So the push-up "
        "half is solved and nothing needs to learn it.\n\nSecond, the one "
        "leg above it crosses 136 mm with nothing holdable in between, and "
        "the ONLY crossing anyone has demonstrated does it by sprawling: all "
        "24 of the teacher's demonstrations reach their maximum foot span "
        "(median 0.929 m) INSIDE that window. Five attempts to cross it "
        "narrowly have failed — 25 open-loop tempos, a 27-variant authored "
        "counter-movement, RL trained directly on it, and a perturbation "
        "curriculum twice.\n\nOne of those five forbade the answer: the "
        "handover run trained under a FLAT 0.75 m cut, which bans the "
        "0.929 m sprawl outright, and then scored 0/24. A ban is not an "
        "impossibility.\n\nSo bound it instead of banning it. The span cut "
        "here is a function of HEIGHT: narrow (0.45 m) everywhere the robot "
        "is supported or standing, wide only while the trunk is between "
        "`pike` and `fold`. The sprawl becomes a brief, measurable moment "
        "rather than the whole motion — and the second stage tightens it."
    ),
    keywords=("stand up from the pike, sprawling only for the launch",
              "bound the sprawl", "sprawl only for the launch",
              "hybrid get-up", "narrow except the launch"),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup"),
    terms=tuple(ASIMOV_GETUP_FLOOR.terms),
    curriculum=(
        CurriculumStage(
            "launch, sprawl allowed", 3_000_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "pike",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.45",
             "MICRODUCK_ASIMOV_GETUP_LAUNCH_SPAN": "1.00"},
            detail="Starts on the AUTHORED pike (0.300 m, 0.215 m span, "
                   "forearms and feet) — not an arm-plateau rung, which is "
                   "splayed to 0.70 and a different pose. A wide base is "
                   "permitted while crossing, and forbidden the moment the "
                   "trunk leaves the window. Zero action from here rises "
                   "0/24."),
        CurriculumStage(
            "tighten the launch", 2_000_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "pike",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.45",
             "MICRODUCK_ASIMOV_GETUP_LAUNCH_SPAN": "0.80"},
            detail="Same task, less room. If the launch survives 0.80 m the "
                   "sprawl is bounded to well under the teacher's 0.929 "
                   "median; if it does not, that is the width this body "
                   "needs and it is worth knowing as a number."),
    ),
    default_steps=5_000_000,
    episode_s=8.0,
    success_metric="stands from the narrow authored pike, where zero action "
                   "rises 0/24 — with the foot span exceeding 0.45 m ONLY "
                   "while the trunk is between pike and fold",
)

_register(ASIMOV_GETUP_HYBRID)

# ---------------------------------------------------------------------------

ASIMOV_GETUP_TIDY = Behavior(
    id="asimov_getup_tidy",
    suggest="get up without doing the splits",
    emoji="\U0001f9f9",
    title="Narrow the working get-up (Asimov 1)",
    description="Fine-tune the get-up that rises 288/288 under a span cut "
                "that is wide ONLY while crossing the gap.",
    how_it_learns=(
        "Training a narrow launch from scratch does not work — six attempts, "
        "including one where the sprawl was explicitly PERMITTED inside the "
        "launch window and the policy never used it (0/36, span 0.203 against "
        "an allowed 1.00). Permission is not discovery, and from a static "
        "start below `fold` the standing state is never sampled, so its value "
        "is never learned.\n\nSo start from something that already crosses. "
        "`teach-asimov_getup_clip-79ef4d` rises 288/288 — and it sprawls the "
        "whole way: 0.986 m on the floor, 0.989 m through the launch, and "
        "still 0.743 m while STANDING, against a 0.215 m stance. The launch "
        "is the one place that width is measured to be necessary; everywhere "
        "else it is habit.\n\nThis charges for it everywhere else. The span "
        "cut is a function of height — 0.45 m below `pike` and above `fold`, "
        "wide in between — so the episode ends if the robot scrambles on the "
        "floor or stands splay-legged, and survives if it keeps the sprawl "
        "to the crossing. The failure mode to watch for is the policy losing "
        "the get-up entirely, which is why the rise is scored as well as the "
        "span."
    ),
    keywords=("get up without doing the splits",
              "narrow the get-up", "stop the splits", "tidy the get-up",
              "get up without sprawling"),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup_clip"),
    terms=tuple(ASIMOV_GETUP_CLIP.terms),
    curriculum=(
        CurriculumStage(
            "narrow everywhere but the launch", 2_000_000,
            {"MICRODUCK_ASIMOV_GETUP_TRACK_TOL": "0.25",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "13.0",
             "MICRODUCK_ASIMOV_GETUP_LAG": "1",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.45",
             "MICRODUCK_ASIMOV_GETUP_LAUNCH_SPAN": "1.05"},
            detail="The donor's own knobs — 13.0 clip and the 1-step command "
                   "lag, without which it scores 15/36 instead of 26/36 — "
                   "plus the height-dependent cut. The launch keeps the room "
                   "it measurably needs (its own max is 1.042 m)."),
        CurriculumStage(
            "squeeze the launch", 2_000_000,
            {"MICRODUCK_ASIMOV_GETUP_TRACK_TOL": "0.25",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "13.0",
             "MICRODUCK_ASIMOV_GETUP_LAG": "1",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "0.45",
             "MICRODUCK_ASIMOV_GETUP_LAUNCH_SPAN": "0.85"},
            detail="Then take the launch down too. Whatever width survives "
                   "here is the number this body actually needs, and it is "
                   "worth knowing even if it is not small."),
    ),
    default_steps=4_000_000,
    episode_s=8.0,
    success_metric="keeps the rise (288/288 for the donor) while the foot "
                   "span outside the launch window drops from 0.986 on the "
                   "floor and 0.743 standing toward the 0.215 m stance",
)

_register(ASIMOV_GETUP_TIDY)

ASIMOV_GETUP_RISE = Behavior(
    id="asimov_getup_rise",
    suggest="stand up out of the push-up",
    emoji="\U0001f9cd",
    title="Stand out of the push-up (Asimov 1)",
    description="The servos play the push-up; this learns the ONE half that "
                "is left — pike to standing, without the splits.",
    how_it_learns=(
        "Measured 2026-10-08, and it changes the task. The authored push-up "
        "— prone, forearms, knees in, quadruped, pike — plays open loop on "
        "the position servos with NO policy at all, at a 0.215 m foot span "
        "the whole way. Hand the rising policy a settled pike and it stands "
        "**15/16**, against 18/36 when it does the whole thing itself. So "
        "the staged get-up is not only possible, it is better, and the floor "
        "half is safe by construction because nothing is learned there.\n\n"
        "What is left is one leg: pike to standing. Rendered, the policy "
        "throws the pike away — trunk 0.280 -> 0.094 m in under half a "
        "second — drops back to the floor and rises out of a 1.029 m lunge. "
        "That is the move to remove.\n\nThe episodes START at the settled "
        "pike, so the floor scramble is out of scope, and the imitation is "
        "OFF: the harvested clips are the teacher's, every one of them "
        "sprawls 0.929 m, so tracking them while charging for span is a "
        "reward fighting itself.\n\n**The width is CHARGED, not cut.** Two "
        "runs established why. A flat 0.45 m termination scored 0/36 with an "
        "episode length of 28 steps. A 0.85 m one let the episodes live and "
        "was worse: the policy's own rise spans 0.99, so standing always "
        "ended the episode, and it learned to stop standing — the `up` term "
        "hit 0.0 at 98 k steps and never came back, and the return settled "
        "at 35.5 against a do-nothing floor of 39.8. A termination has no "
        "gradient at 0.99 m; a cost does. The charge is 0 below a 0.40 m "
        "stance and rises per metre above it, and the hard cut stays at "
        "1.30 m as a guard rail nothing has ever touched."
    ),
    keywords=("stand up out of the push-up", "stand out of the push-up",
              "rise from the pike",
              "finish the stand", "stand up from kneeling"),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup_clip"),
    terms=tuple(ASIMOV_GETUP_CLIP.terms),
    curriculum=(
        CurriculumStage(
            "sees the width", 1_500_000,
            {# NO START_RUNG. This task resets onto a demonstration frame,
             # so the rung it used to carry here was discarded in silence —
             # these runs trained from clip starts at a median trunk height
             # of 0.153 m while `run.json` said "pike". The env now refuses
             # it outright; `asimov_getup_descend` is the one that starts
             # where it says it does.
             "MICRODUCK_ASIMOV_GETUP_FREE": "1",
             "MICRODUCK_ASIMOV_GETUP_TRACK_TOL": "0.25",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "13.0",
             "MICRODUCK_ASIMOV_GETUP_LAG": "1",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "40.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "1.30"},
            detail="17 % of the income a rise earns. The 3-16 ladder cost 0.7-4.0 % and moved the span 0.989 -> 0.991: nothing. A charge the return cannot feel is not a curriculum."),
        CurriculumStage(
            "pays for it", 1_500_000,
            {# NO START_RUNG. This task resets onto a demonstration frame,
             # so the rung it used to carry here was discarded in silence —
             # these runs trained from clip starts at a median trunk height
             # of 0.153 m while `run.json` said "pike". The env now refuses
             # it outright; `asimov_getup_descend` is the one that starts
             # where it says it does.
             "MICRODUCK_ASIMOV_GETUP_FREE": "1",
             "MICRODUCK_ASIMOV_GETUP_TRACK_TOL": "0.25",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "13.0",
             "MICRODUCK_ASIMOV_GETUP_LAG": "1",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "80.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "1.30"},
            detail="33 %. This is where a cheaper route, if one exists, starts to be worth more than the lunge."),
        CurriculumStage(
            "makes it expensive", 1_500_000,
            {# NO START_RUNG. This task resets onto a demonstration frame,
             # so the rung it used to carry here was discarded in silence —
             # these runs trained from clip starts at a median trunk height
             # of 0.153 m while `run.json` said "pike". The env now refuses
             # it outright; `asimov_getup_descend` is the one that starts
             # where it says it does.
             "MICRODUCK_ASIMOV_GETUP_FREE": "1",
             "MICRODUCK_ASIMOV_GETUP_TRACK_TOL": "0.25",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "13.0",
             "MICRODUCK_ASIMOV_GETUP_LAG": "1",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "140.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "1.30"},
            detail="58 %. Rising still beats lying still by two orders of magnitude, because the settled stance is 0.366 m and so costs nothing — only the 38 wide steps are charged."),
        CurriculumStage(
            "a human stance", 1_500_000,
            {# NO START_RUNG. This task resets onto a demonstration frame,
             # so the rung it used to carry here was discarded in silence —
             # these runs trained from clip starts at a median trunk height
             # of 0.153 m while `run.json` said "pike". The env now refuses
             # it outright; `asimov_getup_descend` is the one that starts
             # where it says it does.
             "MICRODUCK_ASIMOV_GETUP_FREE": "1",
             "MICRODUCK_ASIMOV_GETUP_TRACK_TOL": "0.25",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "13.0",
             "MICRODUCK_ASIMOV_GETUP_LAG": "1",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "200.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_MAX_SPAN": "1.30"},
            detail="84 %, and about as far as this can go: at 238 the charge would exceed the whole income and the policy would be right to lie down instead, which is the failure this replaced."),
    ),
    default_steps=6_000_000,
    episode_s=8.0,
    success_metric="rises from the settled pike in at least 15/16 episodes "
                   "— what the donor already does — with a peak foot span "
                   "under 0.40 m, against the donor's 1.026 m",
)

_register(ASIMOV_GETUP_RISE)

ASIMOV_GETUP_DESCEND = Behavior(
    id="asimov_getup_descend",
    suggest="get up narrow, starting from the fold",
    emoji="\U0001f53d",
    title="Descend from the fold, narrow (Asimov 1)",
    description="Start where the narrow rise already works and push the "
                "start DOWN, charging the sprawl so it is never learned.",
    how_it_learns=(
        "**At action clip 13.0 the target of this curriculum does not "
        "exist.** `fold`'s recorded 0.517 m holds only when cold-placed at "
        "the robot's own 4.0; at 13.0 it collapses to 0.242 m, and the repo "
        "recorded that cause next to `_search_rungs` before it was used as a "
        "curriculum target anyway. The launch window and the fall-state "
        "cache are both defined by that number.\n\nAt 4.0 the top of the "
        "narrow ladder is real and needs no policy at all. Open loop, 0.25 s "
        "a leg:\n\n| clip | tempo | fold -> crouch -> stand |\n|---|---|---|\n"
        "| **4.0** | **0.25 s** | **stands: z 0.608, upright +1.00, span "
        "0.215 m** |\n| 4.0 | 0.50 s | collapses |\n| 13.0 | any | collapses "
        "|\n\nSo this trains at 4.0. It costs reach — 4.0 is 1.0 rad from "
        "the standing pose where the cloned get-up wanted 2.88 — and that is "
        "the trade: the clone's route is unreachable here, and the narrow "
        "one is not.\n\n**Nothing is free below fold, and an earlier "
        "version of this recipe claimed otherwise.** 'Zero action rises "
        "23/24 from fold' scored PEAK TRUNK HEIGHT against 0.523 m while the "
        "fold start IS 0.523 m, so it counted the starting pose. Scored on "
        "ENDING UP — height, uprightness and feet after a hold — zero action "
        "stands **0/16** from every fall depth at both clips, and the chain "
        "built on that premise learned nothing, matching zero action "
        "exactly. What is genuinely free is the settled `crouch`: 12/12 at a "
        "0.215 m span.\n\nThe curriculum pushes the start down the fall "
        "from `fold`, where the ramp above it works. Nothing is warm-started "
        "from the sprawling policy, whose route needs a reach this clip does "
        "not have. The span charge is a guard rather than a lever: every "
        "free motion on this ladder is already 0.215 m wide, and 10.0 makes "
        "a sprawling success worth about nothing if the policy finds one."
    ),
    keywords=("get up narrow, starting from the fold", "descend from the fold",
              "narrow get-up from falling", "reverse curriculum get-up"),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup"),
    terms=tuple(ASIMOV_GETUP_FLOOR.terms),
    curriculum=(
        CurriculumStage(
            "from the fold", 1_500_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "falling",
             "MICRODUCK_ASIMOV_GETUP_FALL_DEPTH": "0.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "10.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0"},
            detail="Where it already works: 23/24 with zero action. If a trained policy cannot match THAT, something is wrong with the economy and nothing below is worth running."),
        CurriculumStage(
            "a hand's breadth down", 1_500_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "falling",
             "MICRODUCK_ASIMOV_GETUP_FALL_DEPTH": "0.35",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "10.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0"},
            detail="Start 0.492 m. Zero action manages 5/24, so the rise is sampled but rare — the band where learning should show."),
        CurriculumStage(
            "into the gap", 1_500_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "falling",
             "MICRODUCK_ASIMOV_GETUP_FALL_DEPTH": "0.7",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "10.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0"},
            detail="Start 0.459 m, 4/24 free. This is the depth at which every earlier curriculum stopped improving."),
        CurriculumStage(
            "the whole fall", 1_500_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "falling",
             "MICRODUCK_ASIMOV_GETUP_FALL_DEPTH": "1.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "10.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0"},
            detail="Start 0.426 m, 3/24 free — as deep as the cached fall reaches. A policy that rises here, narrow, is the deliverable: the servos hand it a pike and this finishes the job."),
        CurriculumStage(
            "take the handover", 1_500_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "pike",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "10.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0"},
            detail="THE ONE THAT MATTERS, and the cached fall cannot reach "
                   "it: those states bottom out at 0.440 m while the servos "
                   "hand over at pike, 0.300 m — so this stage starts there "
                   "directly. Nothing is free here: zero action rises 0/8 "
                   "from pike against 3/24 at 0.426 m, which is why it is "
                   "last and why the four stages above it exist. If it "
                   "lands, the servo push-up and the policy meet and the "
                   "staged get-up is complete end to end."),
    ),
    default_steps=7_500_000,
    episode_s=8.0,
    success_metric="rises from the deepest band in at least 18/24 against "
                   "the 3/24 a zero-action policy already gets there, with a "
                   "peak foot span under 0.40 m (the sprawling policy uses "
                   "0.99)",
)

_register(ASIMOV_GETUP_DESCEND)

ASIMOV_GETUP_GAP = Behavior(
    id="asimov_getup_gap",
    suggest="cross the gap from the push-up to standing",
    emoji="\U0001f309",
    title="Cross the pike-to-fold gap (Asimov 1)",
    description="The servos play the push-up, then the policy crosses the "
                "one stretch nothing has ever crossed narrow — assisted at "
                "first, then not.",
    how_it_learns=(
        "Both ends are done. Prone to `pike` plays OPEN LOOP on the position "
        "servos at a 0.215 m foot span, and `fold` to standing is solved "
        "narrow — `teach-asimov_getup_ref-3e895f-s8` stands **24/24** from "
        "fold at 0.409 m where zero action stands 0/24 and the cloned "
        "teacher needs 1.03 m. What is left is 217 mm: `pike` 0.300 -> "
        "`fold` 0.517, with no equilibrium anywhere between them. Scored "
        "properly it is 0/24 for every policy and for doing nothing.\n\n"
        "**The assist is measured, not guessed.** `getup_assist` holds a "
        "fraction of body weight at the pelvis, and from `pike` the peak "
        "climbs with it — 0.266 at 0.00, 0.294 at 0.20, 0.334 at 0.50, 0.385 "
        "at 0.70 — and at **0.85 the crossing appears: fold reached 12/12 at "
        "a 0.355 m span**. Above that it is not physics: 1.00 launches the "
        "robot to 1.8 m and 1.50 to 11 m. So 0.85 is the anchor and the "
        "stages fade it to nothing. A curriculum can only amplify a success "
        "that occurs, and this is the first setting in which one does.\n\n"
        "**Every episode begins with the push-up**, driven by the servos "
        "with the policy's action discarded. That is why the gap gets the "
        "whole budget — no compute is spent relearning a half that needs no "
        "policy — and it is why the lab finally SHOWS the sequence: a reset "
        "that placed the robot at `pike` was one frame, and every run for a "
        "week started above the push-up where nobody could see it. It also "
        "hands the policy a pike it ARRIVED at, carrying real velocity, "
        "rather than one placed at rest.\n\nNo rehearsal stages are "
        "needed: each episode runs pike -> gap -> fold -> crouch -> stand, "
        "so the solved upper half is practised every time. That is the "
        "failure of `…-descend-3ea6b5-s5`, which trained on one start that "
        "admitted no successes and forgot everything above it."
    ),
    keywords=("cross the gap from the push-up to standing", "cross the gap",
              "pike to fold", "the last 217 mm", "assisted get-up"),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup_ref"),
    terms=tuple(ASIMOV_GETUP_REF.terms),
    curriculum=(
        CurriculumStage(
            "assisted all the way", 1_200_000,
            {"MICRODUCK_ASIMOV_GETUP_PROLOGUE": "pushup",
             "MICRODUCK_ASIMOV_GETUP_START": "prone",
             "MICRODUCK_ASIMOV_GETUP_TRACK_TOL": "0.9",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "10.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_ASSIST": "0.85"},
            detail="The anchor: fold reached 12/12 at a 0.355 m span. If the policy cannot do this WITH the assist, nothing below it means anything."),
        CurriculumStage(
            "take a sixth of it away", 1_200_000,
            {"MICRODUCK_ASIMOV_GETUP_PROLOGUE": "pushup",
             "MICRODUCK_ASIMOV_GETUP_START": "prone",
             "MICRODUCK_ASIMOV_GETUP_TRACK_TOL": "0.9",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "10.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_ASSIST": "0.70"},
            detail="0.70 on its own reaches only 0.385 m, so from here the policy has to supply the difference itself."),
        CurriculumStage(
            "half the help", 1_200_000,
            {"MICRODUCK_ASIMOV_GETUP_PROLOGUE": "pushup",
             "MICRODUCK_ASIMOV_GETUP_START": "prone",
             "MICRODUCK_ASIMOV_GETUP_TRACK_TOL": "0.9",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "10.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_ASSIST": "0.55"},
            detail="Past the midpoint of what the measurement says the assist was worth."),
        CurriculumStage(
            "a third", 1_200_000,
            {"MICRODUCK_ASIMOV_GETUP_PROLOGUE": "pushup",
             "MICRODUCK_ASIMOV_GETUP_START": "prone",
             "MICRODUCK_ASIMOV_GETUP_TRACK_TOL": "0.9",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "10.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_ASSIST": "0.40"},
            detail=""),
        CurriculumStage(
            "a quarter", 1_200_000,
            {"MICRODUCK_ASIMOV_GETUP_PROLOGUE": "pushup",
             "MICRODUCK_ASIMOV_GETUP_START": "prone",
             "MICRODUCK_ASIMOV_GETUP_TRACK_TOL": "0.9",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "10.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_ASSIST": "0.25"},
            detail=""),
        CurriculumStage(
            "almost none", 1_200_000,
            {"MICRODUCK_ASIMOV_GETUP_PROLOGUE": "pushup",
             "MICRODUCK_ASIMOV_GETUP_START": "prone",
             "MICRODUCK_ASIMOV_GETUP_TRACK_TOL": "0.9",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "10.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_ASSIST": "0.10"},
            detail="Whatever survives here is nearly the real thing; the last stage is the honest test."),
        CurriculumStage(
            "no help at all", 1_200_000,
            {"MICRODUCK_ASIMOV_GETUP_PROLOGUE": "pushup",
             "MICRODUCK_ASIMOV_GETUP_START": "prone",
             "MICRODUCK_ASIMOV_GETUP_TRACK_TOL": "0.9",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "10.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_ASSIST": "0.00"},
            detail="THE DELIVERABLE: prone to standing, the servos doing the push-up and the policy doing the rest, with nothing holding it up and a foot span a person would call safe."),
    ),
    default_steps=8_400_000,
    episode_s=8.0,
    success_metric="at assist 0.00, stands from the scripted push-up in at "
                   "least 18/24 — scored with asimov_getup_ref.is_standing, "
                   "which needs the trunk up AND upright AND nothing but "
                   "feet on the floor — with a peak foot span under 0.40 m "
                   "against the cloned teacher's 1.03",
)

_register(ASIMOV_GETUP_GAP)

ASIMOV_GETUP_SEAT = Behavior(
    id="asimov_getup_seat",
    suggest="stand up out of a sit",
    emoji="\U0001fa91",
    title="Stand up out of a sit (Asimov 1)",
    description="The pose the authored route never had — chest up while "
                "still low, the way the teacher actually does it.",
    how_it_learns=(
        "The push-up is a dead end on this body and the reason is measured. "
        "It ends FACE DOWN — uprightness **-0.37** at `pike` — and from "
        "there hip flexion swings a leg **751-1095 mm into the air**, "
        "because the pelvis is inverted, so the feet can never come under "
        "the mass. The centre of mass sits **560 mm ahead of the feet** and "
        "rotating the body back over them needs **112 deg** while the hips "
        "and ankles together supply about **59**. There is no back joint to "
        "help: pelvis to torso is `waist_yaw` only, zero bend.\n\nThe "
        "teacher does the opposite and it is in its own data: it reaches "
        "uprightness **+0.49 at 0.06-0.12 m** — chest up while still low — "
        "and rises already upright. This trains that. The SEAT is a long "
        "sit, legs out, on the backs of the thighs: trunk 0.106 m, "
        "uprightness +0.64, foot span **0.215 m**. Settled it comes onto "
        "both feet at 0.43 m and +0.93, above `rock_back` and more upright "
        "than `fold`.\n\nNothing stands from it yet — the narrow policy "
        "that takes `fold` 15/16 scores **0/24** here, because straight "
        "knees and the weight sitting back are off everything it has seen. "
        "That is the case for training on it rather than hoping: an "
        "unsampled state's value is never learned.\n\nThe span charge is a "
        "guard, not a lever. Every pose on this route is already 0.215 m "
        "wide; 10.0 only makes a sprawling success worth nothing if the "
        "policy finds one.\n\n**The reward now pays ROTATION.** The first "
        "run of this recipe scored 0/24 against a reward whose only progress "
        "term was monotone trunk HEIGHT — blind to the one axis this task "
        "turns on. `rotate` is the mirror of `progress` on uprightness, and "
        "when it was first written it paid **0.00 at a weight of 40** "
        "because its baseline latched onto the standing pose at construction "
        "and never reset; it now pays 6.98 where 6.99 is expected."
    ),
    keywords=("stand up out of a sit", "get up from sitting", "the long sit",
              "chest up first", "sit to stand"),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup"),
    terms=tuple(ASIMOV_GETUP_FLOOR.terms),
    curriculum=(
        CurriculumStage(
            "stand out of the sit", 1_500_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "seat",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "10.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "40.0"},
            detail="Start 0.43 m, uprightness +0.93, on both feet at 0.215 m. Zero action is 0/24 here and so is every policy trained so far, so anything above 0 is new."),
        CurriculumStage(
            "keep it, and keep it narrow", 1_500_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "seat",
             "MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "30.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "40.0"},
            detail="Three times the charge. The route is already narrow, so this should cost nothing — and if it does, the policy found the sprawl and is being told not to."),
    ),
    default_steps=3_000_000,
    episode_s=8.0,
    success_metric="stands from the SEAT in at least 12/24 — scored with "
                   "asimov_getup_ref.is_standing — against 0/24 for zero "
                   "action and 0/24 for the narrow fold policy, with a peak "
                   "foot span under 0.40 m",
)

_register(ASIMOV_GETUP_SEAT)

#: When the get-up hands the robot over to Menlo's own walk policy.
#:
#: Measured 2026-10-10, 24 episodes, same seeds:
#:
#:   our get-up alone                  13/24 up   49.1 deg from idle  span 0.376
#:   -> Menlo `hold_pose`               9/24 up   10.9 deg            span 0.321
#:   -> Menlo WALK policy              16/24 up    5.2 deg            span 0.298
#:
#: Handing to the shipped walk policy is better than our policy alone on
#: EVERY axis: it stands more often, it ends 5.2 deg from the default pose
#: instead of 49.1 (arms down, not out), and it brings the feet to 0.298 m —
#: inside the 0.40 m target the whole chain has been chasing.
#:
#: THE TIMING, tuned in the env the LAB actually runs (domain_rand and
#: obs_noise ON — with them off the condition fired twice as readily and the
#: numbers were a fiction). Swept there:
#:
#:   speed 0.22, dwell 0.4 s   fires 10/24   up 10/24   36.1 deg   span 0.277
#:   speed 0.45, dwell 0.12 s  fires 21/24   up 20/24    4.2 deg   span 0.235
#:   speed 0.60, dwell 0.08 s  fires 23/24   up 22/24    3.1 deg   span 0.227
#:   the instant it stands     fires 24/24   up 21/24    3.4 deg   span 0.218
#:
#: EARLIER IS BETTER, which inverts the obvious guess. The 0.22 m/s figure
#: is what a SERVO-HELD stance absorbs; Menlo's walker is an ACTIVE balance
#: controller and takes far more, so waiting for the robot to go quiet only
#: gives our own policy more time to topple first. Waiting too long is the
#: failure mode here, not handing over too soon.
#:
#: Jonathan watched the raw get-up and said it never goes back to idle — the
#: arms stay out and the feet never reshuffle. They do now, because a
#: controller that already knows the idle pose takes over as soon as there
#: is a stand to take over.
def _getup_handoff_due(env) -> bool:
    """Delegates to `AsimovGetupEnv.handoff_ready` — ONE definition, so the
    condition the reward trains against and the one the lab deploys cannot
    drift apart. The thresholds and the measurements behind them live on the
    env (`HANDOFF_SPEED`, `HANDOFF_DWELL`, `HANDOFF_ENV`)."""
    fn = getattr(env, "handoff_ready", None)
    return bool(fn()) if fn is not None else False


ASIMOV_GETUP_TURN = Behavior(
    id="asimov_getup_turn",
    suggest="turn not sprawl",
    emoji="\U0001f504",
    title="Get up off the floor, turning not sprawling (Asimov 1)",
    description="The whole task from the floor, from scratch, paying the "
                "axis the get-up actually turns on.",
    how_it_learns=(
        "The LAST untried configuration, and it is worth one long run "
        "because every earlier one was compromised in a way now known.\n\n"
        "Runs that warm-started from `teach-asimov_getup_clip-79ef4d` "
        "inherited its sprawl — it needs 1.03 m and no charge up to 84 % of "
        "its income moved it below 0.897. Runs that started above the floor "
        "(`pike`, `fold`, the fall cache, the seat) never sampled the part "
        "that matters. And every run before 2026-10-09 optimised a reward "
        "whose only progress term was monotone trunk HEIGHT, while the "
        "blocker is ROTATION: at `pike` the trunk is 112 deg from vertical, "
        "there is no bending joint between pelvis and torso, and lifting the "
        "pelvis only dangles the trunk (assist 0.85 reaches fold's height at "
        "uprightness **-0.86**).\n\n`rotate` fixes that blindness — "
        "monotone gain in uprightness, the mirror of `progress`. When it was "
        "first written it paid **0.00 at a weight of 40**, because its "
        "baseline latched onto the standing pose at construction and never "
        "reset per episode; it now pays 6.98 where 6.99 is expected, and a "
        "test fails the build on the original bug.\n\nSo: from the floor, "
        "from scratch, no sprawling donor, rotation paid, width charged. The "
        "floor is priced — zero action returns **-3.5** with a peak of 0.153 "
        "and stands 0/12 — so any climb is real. Expectations are low and "
        "stated: the teacher's narrowest crossing of this band is 0.917 m "
        "and 0 of 1682 poses below 0.46 m hold on their feet. If this fails "
        "too, the answer is the hardware, and that is the finding to send "
        "Menlo."
    ),
    keywords=("turn not sprawl",
              "get up off the floor without the splits",
              "rotation get-up", "narrow get-up from scratch"),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup"),
    terms=tuple(ASIMOV_GETUP_FLOOR.terms),
    #: Hand over to Menlo's own walker once the rise is finished and calm.
    handoff_fn=_getup_handoff_due,
    handoff_policy="asimov1-shipped",
    #: The get-up chooses its own heading on the way up; the lab's post-
    #: handoff yaw correction would spin that choice away.
    handoff_recenter=False,
    curriculum=(
        CurriculumStage(
            "turn before you climb", 3_000_000,
            {"MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "60.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "10.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40"},
            detail="Rotation paid heavily and height barely: at this weight a 0.1 rise in uprightness is worth 6, against a `stand` term that pays 2.6-9.2 an episode from the floor."),
        CurriculumStage(
            "turn and climb", 3_000_000,
            {"MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "40.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "10.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40"},
            detail="Both, with rotation still the larger signal."),
        CurriculumStage(
            "the whole thing", 3_000_000,
            {"MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "20.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "20.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40"},
            detail="Rotation eased off, width charged harder. Whatever survives here is the honest answer for this robot."),
        CurriculumStage(
            "now get off the floor", 3_000_000,
            {"MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "0.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "20.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40"},
            detail="ROTATION PAY OFF. By stage 3 the policy sits bolt "
                   "upright — uprightness +0.98 from flat, where the "
                   "authored seat was +0.64 and every earlier policy was "
                   "~0 — and then stops, because it has maxed the term it "
                   "is paid for and rising earns almost nothing. Taking "
                   "the rotation pay away leaves height as the only thing "
                   "left to earn, from a policy that can already get "
                   "upright. Watched in the lab it sits up and bounces on "
                   "its backside; its 0.350 m peak is NOT progress — at "
                   "that moment it is head-down and airborne, two separate "
                   "maxima reported as one pose."),
        CurriculumStage(
            "put your feet under you", 3_000_000,
            {"MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "0.0",
             "MICRODUCK_ASIMOV_GETUP_TUCK": "50.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "20.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40"},
            detail="THE MEASURED BLOCKER. At the end of the sit-up the knee "
                   "sits at **-1 deg — its 0 deg stop** — legs straight out, "
                   "with the centre of mass **300 mm from the feet**. It is "
                   "a long sit, and 86 deg of knee travel is completely "
                   "unused. Nothing paid for using it: tucking the feet in "
                   "LOWERS the trunk and costs uprightness for a moment, so "
                   "`rise`, `stand`, `progress` and `rotate` all charge for "
                   "the one move that makes standing possible. `tuck` pays "
                   "monotone reduction in the CoM-to-feet distance — "
                   "measured, pike to fold is worth 0.304 of it."),
        CurriculumStage(
            "feet under you, and mean it", 3_000_000,
            {"MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "0.0",
             "MICRODUCK_ASIMOV_GETUP_TUCK": "200.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "20.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40"},
            detail="FOUR TIMES THE WEIGHT, because the measurement says the "
                   "lever was underpowered rather than wrong. At 50 the "
                   "`tuck` term earned 5.3 an episode against `stand` at "
                   "125 — **2.4 % of the reward** — and still unlocked the "
                   "knee from its 0 deg stop to 84.7 deg, closed the "
                   "CoM-to-feet gap from 300 mm to 5 mm in one episode, and "
                   "took the highest-while-upright pose from 0.158 to a "
                   "median 0.345 (best 0.404), past `rock_back`. That is the "
                   "same immateriality that sank the span charge, and the "
                   "same fix: make it worth something."),
        CurriculumStage(
            "and stay up", 4_000_000,
            {"MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "0.0",
             "MICRODUCK_ASIMOV_GETUP_TUCK": "200.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "30.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40"},
            detail="IT GETS UP — 4/24 reach a genuine stand from the floor "
                   "at trunk 0.605, uprightness +0.85, on both feet and "
                   "nothing else, at a 0.500 m span against the teacher's "
                   "0.917. What it cannot do is STAY up: the longest "
                   "unbroken stand is 0.24 s. The move is found, so this "
                   "stage is practice — same lever, more of it, and the "
                   "width charged harder now that 0.500 is the thing to "
                   "beat rather than 0.99. `standing` pays per step while "
                   "up, and until now it has never had a pose to pay for.\n\n"
                   "WHAT IT ACTUALLY DOES, filmed and counted frame by "
                   "frame: 23/24 reach a stand, and the route there is a "
                   "HOP. Of the 3.4 s rise, **1.04 s has zero floor "
                   "contacts** — 8 separate flights, the lowest geom of a "
                   "32 kg robot reaching **105 mm** of clearance — and 0.00 "
                   "s airborne once it is up. There is no push-up and no "
                   "knee because it never needed one: a bounce buys height, "
                   "uprightness and feet-under-mass in a single impulse, "
                   "and nothing in this economy charged for the impulse. "
                   "The stand is real; it is not a get-up."),
        CurriculumStage(
            "and keep something on the ground", 4_000_000,
            {"MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "0.0",
             "MICRODUCK_ASIMOV_GETUP_TUCK": "200.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "30.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_GROUND": "6.0"},
            detail="THE ROUTE, not the arrival. Everything up to here paid "
                   "for ending up vertical and nothing priced how the robot "
                   "got there, so it found the cheapest route through that "
                   "reward and the cheapest route is ballistic.\n\n"
                   "`ground` charges 6.0 for any step with NO part of the "
                   "robot touching the floor. Priced against the measured "
                   "economy, not chosen: the riser earns **1283.5** an "
                   "episode and spends **61 steps** airborne, so the charge "
                   "costs it 368 — 29 % of its income, material enough to "
                   "beat the immateriality that sank the first span charge "
                   "and the first tuck — while doing nothing still earns "
                   "**2.9**. A penalty that can make lying still the better "
                   "deal is the recorded failure of "
                   "`teach-asimov_getup_rise-126f42-s2`, and 368 against "
                   "1283 is nowhere near it.\n\n"
                   "A charge and not a termination, for the same reason "
                   "`span_cost` is: a cut at the moment of flight ends every "
                   "attempt to rise, and the policy correctly learns not to "
                   "rise. Asked of MuJoCo's contact list rather than of a "
                   "clearance threshold, because a robot resting on one heel "
                   "has every other geom well clear of the floor and is not "
                   "in the air. Warm-started from the stage that already "
                   "stands: the arrival is solved, this is the route.\n\n"
                   "IT WORKED, PARTLY. Airborne time per episode fell from a "
                   "median **1.19 s to 0.54 s** (cleanest seed 0.18 s), "
                   "stands went 23/24 to **24/24** and the span 0.555 to "
                   "0.526. The clean route now reads the way it should: flat "
                   "on the floor, roll onto one foot, push up on that foot, "
                   "standing at a **0.35 m** span. What it cost was the "
                   "hold — longest unbroken stand 4.68 s down to 1.60 s."),
        CurriculumStage(
            "stand over your own feet", 4_000_000,
            {"MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "0.0",
             "MICRODUCK_ASIMOV_GETUP_TUCK": "200.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "30.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_GROUND": "6.0",
             "MICRODUCK_ASIMOV_GETUP_BALANCE": "40.0"},
            detail="IT WAS NEVER BALANCED. Measured on the stage above: "
                   "while the robot is STANDING by `standing_report` — trunk "
                   "up, upright, nothing but feet down — its centre of mass "
                   "is OUTSIDE its own support polygon **74 % of the time** "
                   "(median -22 mm, p10 -131 mm, only 5 of 8 contact pads "
                   "down). Every lost stand has one trace and it is not a "
                   "topple: uprightness holds at **+0.98** while the gap "
                   "from the feet grows 200 -> 320 mm and the trunk slides "
                   "down. It passes through vertical on the way over.\n\n"
                   "Nothing paid for staying over the feet. `tuck` pays the "
                   "CoM-to-feet gap closing but is MONOTONE BEST-SO-FAR — it "
                   "pays for touching a value once, never for holding it — "
                   "and `standing` pays a toppling robot exactly what it "
                   "pays a balanced one. Same blindness as `rotate` before "
                   "it, one level up.\n\n"
                   "`balance` charges the distance the CoM sits OUTSIDE the "
                   "polygon, gated on being up (upright past STAND_UPRIGHT, "
                   "nothing but feet down) so it is exactly zero for the "
                   "whole rise. Priced: a standing step earns **11.58**, the "
                   "median excursion is 55 mm, so 40 costs 2.2 a step — "
                   "**24 %** of standing income, past the immateriality that "
                   "sank the first span charge and the first tuck. A "
                   "BALANCED robot pays nothing at all, so the charge only "
                   "ever prices the failure; and dodging the gate by staying "
                   "below upright 0.85 forfeits `stand` AND `standing`, 100 "
                   "% of that income, which is strictly worse.\n\n"
                   "IT MOVED ITS OWN TARGET AND NOT THE DELIVERABLE. "
                   "Balanced stands 8/24 -> **14/24**, time with the CoM "
                   "inside the polygon 14 % -> **30 %**, airborne 0.54 -> "
                   "0.45 s (paired, p=0.035). But the hold is a null "
                   "(+0.19 s, p=0.152) and **0/24** are still up at the "
                   "end, the worst of the three. The reason is mechanical "
                   "and it is the next stage."),
        CurriculumStage(
            "come home", 4_000_000,
            {"MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "0.0",
             "MICRODUCK_ASIMOV_GETUP_TUCK": "200.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "30.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_GROUND": "6.0",
             "MICRODUCK_ASIMOV_GETUP_BALANCE": "40.0",
             "MICRODUCK_ASIMOV_GETUP_HOME": "10.0",
             "MICRODUCK_ASIMOV_GETUP_LIMIT": "50.0"},
            detail="IT IS NOT BALANCING, IT IS JAMMING — and Jonathan saw "
                   "it before it was measured: *it gets all the way up to "
                   "standing and then it just stops, it doesn't move "
                   "again*.\n\nThe pose it reaches is **19 deg from the "
                   "home stand on average, 57 deg at the worst joint**: "
                   "`waist_yaw` +56.8 against home 0 (and that is the ONLY "
                   "joint between pelvis and torso, so the torso is twisted "
                   "57 deg), the hips scissored -53.6/+51.4 against home's "
                   "-8.6/+8.6, `left_ankle_pitch` +20.9 against home's "
                   "-17.2 — the full width of a 20 deg range, on the wrong "
                   "side — and `left_knee` on its 0 deg stop against home's "
                   "+25.8.\n\nAnd it is not a pose the robot can hold: it "
                   "puts **eight joints outside the model's own limits at "
                   "once** (both ankle pitches, both ankle rolls, a knee, "
                   "both shoulder rolls, an elbow). MuJoCo's limits are "
                   "soft, so the policy pushes through the stops and stands "
                   "on the constraint forces like a scaffold. "
                   "`_refuse_illegal_joints` will not even load that pose. "
                   "It freezes because there is no range left in any "
                   "direction, slides off because a jam is not a balance, "
                   "and on hardware it would not exist at all.\n\n"
                   "The target is measured, not invented: the home stand IS "
                   "`robot.default_pose` (0.0 deg from the authored `stand` "
                   "waypoint), and under ZERO action the servos hold it "
                   "**8.00 s of 8** at trunk 0.608, uprightness +1.00, CoM "
                   "**+47 mm inside** the polygon. `home` pays "
                   "exp(-err/0.20) per step for sitting in it, gated on "
                   "being upright AND on the feet being the only thing down "
                   "— a KNEEL in the default pose is vertical and would "
                   "otherwise collect full price for never standing up. "
                   "Priced: 0.184 of the pay is already earned, so weight "
                   "10 leaves **8.2 a step** on the table against a "
                   "standing income of 11.58, and it can only be collected "
                   "while actually up. `limit` charges 50 per radian past a "
                   "stop — the median jam is 0.049 rad, so 2.4 a step, "
                   "21 % of standing income.\n\n"
                   "IT STANDS AND STAYS. `teach-asimov_getup_turn-e044c4`, "
                   "paired against s9 on the same 24 episodes:\n\n"
                   "  * STILL UP AT THE END **0/24 -> 7/24** (McNemar exact "
                   "p = **0.0156**) — the deliverable, moving for the first "
                   "time\n"
                   "  * longest hold **2.52 -> 7.04 s** of an 8 s episode, "
                   "median 1.08 -> 1.73 s; balanced hold 0.80 -> **6.56 s**"
                   "\n  * joint error vs home 19.6 -> **15.8 deg**, better "
                   "on **24 of 24** seeds, p = 0.000\n"
                   "  * over-limit jam 0.050 -> **0.038 rad**, better on 23 "
                   "of 24, p = 0.000\n"
                   "  * CoM inside the polygon 30 -> 38 %\n\n"
                   "Filmed, seed 2: face down at t=0 (trunk 0.142, "
                   "uprightness -0.11, eight parts on the floor), standing "
                   "at **t=1.08**, and still standing at t=7.56 — trunk "
                   "0.609, uprightness **+1.00**, feet **0.352 m** apart "
                   "against the teacher's 0.917, with **0.14 s** of flight "
                   "in the whole rise.\n\nNOT SOLVED: 17 of 24 still fall "
                   "before the end, it is still 15.8 deg from home and "
                   "still 0.038 rad into its stops, and the span widened "
                   "0.519 -> 0.562 while airborne time went 0.45 -> 0.54 s "
                   "— both small, neither significant, both worth watching "
                   "if `home` is pushed harder."),
        CurriculumStage(
            "come home, and mean it", 4_000_000,
            {"MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "0.0",
             "MICRODUCK_ASIMOV_GETUP_TUCK": "200.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "30.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_GROUND": "6.0",
             "MICRODUCK_ASIMOV_GETUP_BALANCE": "40.0",
             "MICRODUCK_ASIMOV_GETUP_HOME": "30.0",
             "MICRODUCK_ASIMOV_GETUP_LIMIT": "50.0"},
            detail="THREE TIMES THE WEIGHT, on the lever that worked. "
                   "Priced on s10's OWN economy rather than s9's: a "
                   "standing step earns **14.29**, of which `home` is only "
                   "**+2.41 (17 %)** while **0.757 of the pay is still "
                   "unearned** — the policy sits 15.8 deg from the default "
                   "pose and collects a quarter of what is there. At 30 it "
                   "earns 7.3 a step with **22.7 still on the table**, "
                   "which makes coming home the largest single thing in the "
                   "economy.\n\nThat dominance is safe here in a way it "
                   "would not be for a charge: `home` is gated on being "
                   "upright AND on the feet being the only thing down, so "
                   "the ONLY way to collect it is to be standing. A policy "
                   "that chases it has to stand up first and stay up, and "
                   "the pose it is being paid to reach is the one that "
                   "holds 8.00 s of 8 under zero action with the CoM +47 mm "
                   "inside the polygon. The failure mode a dominant term "
                   "usually brings — a proxy that failure also satisfies — "
                   "has no route in.\n\nTwo things to WATCH rather than "
                   "fix, because s10 moved both the wrong way and neither "
                   "was significant: foot span 0.519 -> 0.562 (target 0.40) "
                   "and airborne 0.45 -> 0.54 s. One lever at a time; "
                   "`span_cost` and `ground` stay where they are so this "
                   "stage answers one question."),
        CurriculumStage(
            "now take a shove", 4_000_000,
            {"MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "0.0",
             "MICRODUCK_ASIMOV_GETUP_TUCK": "200.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "30.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_GROUND": "6.0",
             "MICRODUCK_ASIMOV_GETUP_BALANCE": "40.0",
             "MICRODUCK_ASIMOV_GETUP_HOME": "30.0",
             "MICRODUCK_ASIMOV_GETUP_LIMIT": "50.0",
             "MICRODUCK_ASIMOV_GETUP_PUSH": "0.15"},
            detail="THE STATE 38 M STEPS NEVER SAMPLED. Jonathan watched s11 "
                   "and said *it doesn't look like it's doing any "
                   "correction*. It was not — and no reward could have made "
                   "it, because across this entire curriculum the robot was "
                   "never disturbed: `xfrc_applied` carried the vertical "
                   "ASSIST and nothing else. A recovery is a state no "
                   "rollout ever contained, and an unsampled state's value "
                   "is never learned. Three stages of balance economics were "
                   "spent on a curriculum fault.\n\n"
                   "CALIBRATED, not picked. A servo-held home stand — the "
                   "best open-loop stand on this robot, CoM +47 mm inside "
                   "its polygon — survives **0.15 BW** and falls at "
                   "**0.20 BW** (0.15 s horizontal, 3 s to recover). So a "
                   "shove past 0.20 cannot be answered by stiffness, only by "
                   "correcting. Training draws magnitude uniform in [0.5, "
                   "1.5] x 0.15 BW = **0.075 to 0.225**, which straddles "
                   "that line: some are absorbable, some demand a real "
                   "correction, and none are hopeless. A distribution where "
                   "nothing recovers has no anchor and no gradient — that is "
                   "the `start_rung=\"falling\"` mistake, one level up.\n\n"
                   "Shoves land only once `is_standing` holds (a shove "
                   "during the rise is noise on a motion that is already "
                   "hard), repeat every 0.8-1.8 s so one long stand is many "
                   "recovery samples, and are uniform in yaw so no single "
                   "canned answer works. Scored by "
                   "`scripts/getup_acceptance.py`, which gates on recovery "
                   "from 0.20 BW — a bar no passive stance can clear.\n\n"
                   "IT MADE THINGS WORSE, and the dose-response says why: at "
                   "a shove of **0.00** s12 holds 0.37 where s11 holds 0.68, "
                   "so the undisturbed stand itself regressed. Recovery fell "
                   "0.75 -> 0.42 and `stays` 13 -> 7/24. Pushes cost 15 % of "
                   "the return (1734 -> 1479) and standing arms the fall "
                   "cut, so standing became the state that ends episodes. "
                   "The real blocker was found by measuring the next stage's "
                   "term instead."),
        CurriculumStage(
            "arrive quietly", 4_000_000,
            {"MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "0.0",
             "MICRODUCK_ASIMOV_GETUP_TUCK": "200.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "30.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_GROUND": "6.0",
             "MICRODUCK_ASIMOV_GETUP_BALANCE": "40.0",
             "MICRODUCK_ASIMOV_GETUP_HOME": "30.0",
             "MICRODUCK_ASIMOV_GETUP_LIMIT": "50.0",
             "MICRODUCK_ASIMOV_GETUP_CALM": "10.0"},
            detail="THE RISE IS A LUNGE. Measured at the instant it first "
                   "satisfies `is_standing`: horizontal trunk speed "
                   "**0.89 m/s** (p90 1.06), angular **5.15 rad/s** — "
                   "against a servo-held home stand that absorbs "
                   "**0.22 m/s** and falls at 0.29. It arrives four times "
                   "too fast. Within the same 24 episodes the split is "
                   "clean: survivors arrive at 0.60-0.87 m/s, fallers at "
                   "0.88-0.98.\n\nThis is what three stages of pose work "
                   "were missing. `balance` pays the CoM over the feet and "
                   "`home` pays the joints at default, and NEITHER can be "
                   "held by a 32 kg body arriving with a metre per second "
                   "and a 20 deg ankle to stop it. It also explains the jam "
                   "(bracing), the saturated ankles (absorbing), and why "
                   "handing off to the servos at the standing instant makes "
                   "it WORSE — 13/24 -> 3/24, and a SLOWER ramp is worse "
                   "still (1/24), because no passive stance absorbs "
                   "0.89 m/s.\n\n`calm` charges the excess over "
                   "`CALM_FREE` = 0.22, which is measured and not chosen: "
                   "it is exactly what the passive stand demonstrably takes. "
                   "One-sided, so a quiet arrival pays nothing; gated on "
                   "being on its feet, so the rise is never charged for "
                   "moving. Priced: 0.67 m/s of excess at weight 10 is 6.7 a "
                   "step against a standing income of 14.29.\n\nWarm "
                   "started from **s11**, not s12, and `push` is back to 0 "
                   "— s12 regressed the undisturbed stand, so this stage "
                   "tests one lever from the better parent.\n\n"
                   "IT REGRESSED TOO: `stays` 13/24 -> 6/24. The charge ran "
                   "for as long as the robot was UP, which prices the "
                   "CORRECTIONS that keep it there. The lunge is how it "
                   "ARRIVES, not how it holds."),
        CurriculumStage(
            "arrive quietly (the arrival only)", 4_000_000,
            {"MICRODUCK_ASIMOV_GETUP_ACTION_CLIP": "4.0",
             "MICRODUCK_ASIMOV_GETUP_ROTATE": "0.0",
             "MICRODUCK_ASIMOV_GETUP_TUCK": "200.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_COST": "30.0",
             "MICRODUCK_ASIMOV_GETUP_SPAN_TARGET": "0.40",
             "MICRODUCK_ASIMOV_GETUP_GROUND": "6.0",
             "MICRODUCK_ASIMOV_GETUP_BALANCE": "40.0",
             "MICRODUCK_ASIMOV_GETUP_HOME": "30.0",
             "MICRODUCK_ASIMOV_GETUP_LIMIT": "50.0",
             "MICRODUCK_ASIMOV_GETUP_CALM": "10.0"},
            detail="THE ARMS ARE LOAD-BEARING, and that is the measurement "
                   "that ties the whole thing together. Jonathan, watching "
                   "it beside the shipped Menlo walk: *it's not really "
                   "standing straight with both arms down, its legs are "
                   "still tweaked*. Measured on s11, the standing pose is "
                   "13.2 deg from neutral on AVERAGE but **49.4 deg** at "
                   "`right_shoulder_pitch` (+63.8 against +14.3) and 37 deg "
                   "at `right_shoulder_yaw` — arms 16.4 deg mean / 49.4 "
                   "worst, legs 10.9 / 27.4.\n\nThat arm is not sloppiness. "
                   "FORCE the arms to neutral on a standing policy and it "
                   "goes **13/24 -> 0/24**. It is catching itself with them, "
                   "because it arrives at 0.89 m/s against a stance that "
                   "absorbs 0.22 and a 20 deg ankle. So no pose reward can "
                   "put the arms down — the neutral pose is exactly what it "
                   "cannot afford — and `home` is left at s11's weight and "
                   "s11's formula on purpose.\n\nThe only lever upstream "
                   "of all of it is the arrival, and `calm` now charges the "
                   "first **1.0 s** after it first stands rather than the "
                   "whole hold. Same threshold (0.22 m/s, measured), same "
                   "weight, same parent — one changed line against the "
                   "stage that regressed."),
    ),
    default_steps=37_000_000,
    episode_s=8.0,
    success_metric="stands from the floor and STAYS up — asimov_getup_ref."
                   "is_standing, so trunk up AND upright AND nothing but "
                   "feet down — with a peak foot span under 0.60 m AND the "
                   "rise SUPPORTED THROUGHOUT (under 0.10 s of total flight "
                   "against the hopping route's 1.04 s), against 0/12 and a "
                   "-3.5 return for zero action and the teacher's 0.917 m "
                   "minimum",
)

_register(ASIMOV_GETUP_TURN)


ASIMOV_GETUP_WHOLE = Behavior(
    id="asimov_getup_whole",
    suggest="get up the whole economy",
    emoji="\U0001f3d7",
    title="Get up off the floor, whole economy from step one (Asimov 1)",
    description="The same task from the floor, from scratch, but with every "
                "term that was ever retrofitted present from the first step.",
    how_it_learns=(
        "FOUR FINE-TUNES IN A ROW MADE IT WORSE, and that is the finding "
        "this chain acts on. `teach-asimov_getup_turn-540bc4` (s11) stands "
        "13/24 and recovers 0.75; shoves, then an arrival charge over the "
        "whole hold, then the same charge windowed to the arrival, then a "
        "per-joint home term, each took it backwards — 7/24, 6/24, 4/24.\n\n"
        "The reason is mechanical, not statistical. That policy reaches "
        "standing at **0.89 m/s** against a stance that absorbs **0.22**, so "
        "it catches itself with its arms: force the arms to neutral once it "
        "is up and it goes **13/24 to 0/24**. The lunge is LOAD-BEARING. "
        "Every late nudge toward a quiet arrival or a neutral pose is "
        "therefore a nudge toward falling, and no weight on it can win.\n\n"
        "So: start over, with `ground`, `limit`, `balance`, `home` and "
        "`calm` present while the rise is still being shaped. `ground` and "
        "`limit` apply throughout, so the hop and the jam never form. "
        "`balance`, `home` and `calm` are gated on already being up, so they "
        "cost the rise nothing and only ever price the arrival and the hold. "
        "The floor is priced: zero action returns **-53.7** and rises "
        "**0/12**, so any climb is real.\n\nIf this chain also lands at "
        "13/24, the answer is that a 20 deg ankle cannot hold what this "
        "robot's legs can lift, and that is the finding to send Menlo."
    ),
    keywords=("get up the whole economy", "whole economy get up",
              "get up from scratch with everything on",
              "start the get-up over"),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup"),
    terms=tuple(ASIMOV_GETUP_FLOOR.terms),
    curriculum=(
        CurriculumStage(
            "turn before you climb", 3_000_000,
            {
             'MICRODUCK_ASIMOV_GETUP_ACTION_CLIP': '4.0',
             'MICRODUCK_ASIMOV_GETUP_GROUND': '6.0',
             'MICRODUCK_ASIMOV_GETUP_LIMIT': '50.0',
             'MICRODUCK_ASIMOV_GETUP_ROTATE': '60.0',
             'MICRODUCK_ASIMOV_GETUP_SPAN_COST': '10.0',
             'MICRODUCK_ASIMOV_GETUP_SPAN_TARGET': '0.40'},
            detail='Rotation paid heavily, as in the TURN chain — but `ground` and `limit` are ON from the first step. Neither is gated on standing, so both apply through the whole rise: the hop and the joint-stop jam can never be learned in the first place, which is the entire point of starting over.'),
        CurriculumStage(
            "turn and climb", 3_000_000,
            {
             'MICRODUCK_ASIMOV_GETUP_ACTION_CLIP': '4.0',
             'MICRODUCK_ASIMOV_GETUP_GROUND': '6.0',
             'MICRODUCK_ASIMOV_GETUP_LIMIT': '50.0',
             'MICRODUCK_ASIMOV_GETUP_ROTATE': '40.0',
             'MICRODUCK_ASIMOV_GETUP_SPAN_COST': '10.0',
             'MICRODUCK_ASIMOV_GETUP_SPAN_TARGET': '0.40'},
            detail='Both, with rotation still the larger signal.'),
        CurriculumStage(
            "the whole thing", 3_000_000,
            {
             'MICRODUCK_ASIMOV_GETUP_ACTION_CLIP': '4.0',
             'MICRODUCK_ASIMOV_GETUP_GROUND': '6.0',
             'MICRODUCK_ASIMOV_GETUP_LIMIT': '50.0',
             'MICRODUCK_ASIMOV_GETUP_ROTATE': '20.0',
             'MICRODUCK_ASIMOV_GETUP_SPAN_COST': '20.0',
             'MICRODUCK_ASIMOV_GETUP_SPAN_TARGET': '0.40'},
            detail='Rotation eased off, width charged harder.'),
        CurriculumStage(
            "put your feet under you", 3_000_000,
            {
             'MICRODUCK_ASIMOV_GETUP_ACTION_CLIP': '4.0',
             'MICRODUCK_ASIMOV_GETUP_GROUND': '6.0',
             'MICRODUCK_ASIMOV_GETUP_LIMIT': '50.0',
             'MICRODUCK_ASIMOV_GETUP_ROTATE': '0.0',
             'MICRODUCK_ASIMOV_GETUP_TUCK': '50.0',
             'MICRODUCK_ASIMOV_GETUP_SPAN_COST': '20.0',
             'MICRODUCK_ASIMOV_GETUP_SPAN_TARGET': '0.40'},
            detail='Rotation off, `tuck` on — the measured blocker in the TURN chain, where the knee sat at its 0 deg stop with the CoM 300 mm from the feet.'),
        CurriculumStage(
            "feet under you, and mean it", 3_000_000,
            {
             'MICRODUCK_ASIMOV_GETUP_ACTION_CLIP': '4.0',
             'MICRODUCK_ASIMOV_GETUP_GROUND': '6.0',
             'MICRODUCK_ASIMOV_GETUP_LIMIT': '50.0',
             'MICRODUCK_ASIMOV_GETUP_TUCK': '200.0',
             'MICRODUCK_ASIMOV_GETUP_SPAN_COST': '20.0',
             'MICRODUCK_ASIMOV_GETUP_SPAN_TARGET': '0.40',
             'MICRODUCK_ASIMOV_GETUP_BALANCE': '40.0'},
            detail='Four times the tuck, and `balance` joins — gated on being up, so it costs the rise nothing and prices the CoM leaving the support polygon the moment there is a polygon to leave.'),
        CurriculumStage(
            "arrive quietly, and come home", 4_000_000,
            {
             'MICRODUCK_ASIMOV_GETUP_ACTION_CLIP': '4.0',
             'MICRODUCK_ASIMOV_GETUP_GROUND': '6.0',
             'MICRODUCK_ASIMOV_GETUP_LIMIT': '50.0',
             'MICRODUCK_ASIMOV_GETUP_TUCK': '200.0',
             'MICRODUCK_ASIMOV_GETUP_SPAN_COST': '30.0',
             'MICRODUCK_ASIMOV_GETUP_SPAN_TARGET': '0.40',
             'MICRODUCK_ASIMOV_GETUP_BALANCE': '40.0',
             'MICRODUCK_ASIMOV_GETUP_HOME': '30.0',
             'MICRODUCK_ASIMOV_GETUP_CALM': '10.0'},
            detail='THE TWO THAT COULD NEVER BE RETROFITTED. `calm` charges arrival above 0.22 m/s for the first second of standing; `home` pays the default pose. Added as fine-tunes to a finished policy they failed four times running, because by then the lunge was load-bearing — forcing the arms to neutral on that policy took it 13/24 to 0/24. Here they are present while the rise is still being shaped, so the policy never has to unlearn a catch it has already come to depend on.'),
        CurriculumStage(
            "and stay up", 4_000_000,
            {
             'MICRODUCK_ASIMOV_GETUP_ACTION_CLIP': '4.0',
             'MICRODUCK_ASIMOV_GETUP_GROUND': '6.0',
             'MICRODUCK_ASIMOV_GETUP_LIMIT': '50.0',
             'MICRODUCK_ASIMOV_GETUP_TUCK': '200.0',
             'MICRODUCK_ASIMOV_GETUP_SPAN_COST': '30.0',
             'MICRODUCK_ASIMOV_GETUP_SPAN_TARGET': '0.40',
             'MICRODUCK_ASIMOV_GETUP_BALANCE': '40.0',
             'MICRODUCK_ASIMOV_GETUP_HOME': '30.0',
             'MICRODUCK_ASIMOV_GETUP_CALM': '10.0'},
            detail='Practice. Same economy, more of it, scored by `scripts/getup_acceptance.py` — rises, stays, comes home (mean AND worst joint), no jam, narrow, and recovers from a 0.20 BW shove that no passive stance survives.'),
    ),
    default_steps=23_000_000,
    episode_s=8.0,
    success_metric="scripts/getup_acceptance.py says MEETS THE GOAL: rises "
                   ">=22/24, still standing at the end >=20/24, within 10 deg "
                   "mean and 20 deg worst of default_pose, under 0.010 rad of "
                   "joint-stop jam, span <=0.40 m, and >=75 % recovery from a "
                   "0.20 BW shove no passive stance survives",
)

_register(ASIMOV_GETUP_WHOLE)

ASIMOV_GETUP_HIST = Behavior(
    id='asimov_getup_hist',
    suggest='get up with memory',
    emoji='🧠',
    title='Get up with memory (Asimov 1) — CONTROL',
    description="The same get-up, but the policy remembers the last "
                "five moments instead of only the current one. Cannot "
                "run on the robot: it is a measurement, not a "
                "deliverable.",
    how_it_learns=(
        "This is a CONTROL, and it is the one experiment that should "
        "have been run first. Everything about it is identical to 'Get "
        "up by reference' — same reference, same reward, same six "
        "rungs, same number of steps — except for one thing: the policy "
        "is shown the last five moments stacked together instead of "
        "only the current one.\n\nWhy it matters. Getting up is a "
        "SEQUENCE, and the right thing to do depends on where in that "
        "sequence you are. From a single snapshot, two very different "
        "moments can look nearly the same — the joints are in similar "
        "places, but one is on the way up and the other is on the way "
        "down. If the robot genuinely cannot tell them apart, then no "
        "reward and no amount of practice will ever teach it, because "
        "the information simply is not there.\n\nSix training runs have "
        "now failed at exactly one moment of the motion. If this one "
        "gets past it and the single-snapshot version never does, that "
        "is the answer — and it is a result worth publishing either "
        "way, because nobody has established whether a get-up that fits "
        "the robot's real deployment interface is possible at all. The "
        "catch: a policy trained this way needs five times the input "
        "and will NOT load on the robot or in Menlo's viewer. It is "
        "here to settle a question, not to be shipped."
    ),
    keywords=('get up with memory', 'get up with history', 'five frame get up', 'get up remembering'),
    robot="asimov1",
    trainer=("-m", "microduck_local.train", "--robot", "asimov1",
             "--task", "getup_ref"),
    terms=(
        RewardTerm('track', 'matches the reference pose for this phase',
                   10.0, _env_owned),
        RewardTerm('contact', 'the right parts are touching the floor',
                   2.0, _env_owned),
        RewardTerm('advance', 'reached the next phase, paid once',
                   30.0, _env_owned),
        RewardTerm('up', 'is up, upright and on both feet',
                   8.0, _env_owned),
        RewardTerm('action_rate_penalty', 'charged for twitchy joint commands',
                   0.5, _env_owned, is_penalty=True),
    ),
    curriculum=(
        CurriculumStage(
            'stand up from the fold', 300_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.6',
                     'MICRODUCK_ASIMOV_GETUP_HISTORY': '5',
                    },
            detail="The only rung measured to WORK, and now trained "
                   "against the COMMAND DELAY on every episode rather "
                   "than half of them. At the inherited 50/50 it "
                   "learned a controller with no delay margin: two "
                   "independent chains both held the full 7.76 s on "
                   "every episode WITHOUT the delay and fell at 1.3-2.0 "
                   "s on every episode WITH it, perfectly separated, no "
                   "exceptions. One control step is 20 ms -- the CAN-"
                   "bus latency Menlo's own randomisation models -- so "
                   "that stand does not ship. From a single 78-float "
                   "frame the policy cannot tell which episode it is "
                   "in, so it has to use one controller for both and "
                   "settled on the one that is optimal with no delay; "
                   "forcing the delay removes the choice, and a "
                   "controller with margin at 20 ms is stable at 0. "
                   "from `fold` (pelvis 0.517 m, on its feet, trunk 33 "
                   "deg forward) the servos alone reach a held stand — "
                   "0.608 m, uprightness +1.00, four seconds — once the "
                   "leg timings are 0.25 s rather than the 0.60 s the "
                   "reference shipped. Starting here means the policy "
                   "learns the finish on a rung it can actually "
                   "complete, instead of warm-starting every later rung "
                   "from something that has never stood."),
        CurriculumStage(
            'catch it on the way up', 400_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold,rock_back',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.6',
                     'MICRODUCK_ASIMOV_GETUP_HISTORY': '5',
                    },
            detail="FREE from here down: the reference is dropped for "
                   "the intermediates and the pay is the product "
                   "`pelvis height x uprightness`, counted only while a "
                   "foot is on the floor. Five chains were spent asking "
                   "the policy to TRACK the authored route through this "
                   "gap, and that route has since been measured "
                   "impossible — `com - foot` is 0.618 m at the pike "
                   "against 0.114 at `fold`, it does not move while "
                   "four contacts are planted, and the trunk cannot "
                   "rotate up first. A tracking reward against an "
                   "unreachable waypoint does not merely fail to help, "
                   "it CHARGES the policy for trying anything else. The "
                   "waypoints still serve as the `advance` ladder, so "
                   "the structure is kept and only the imitation is let "
                   "go. Starts at the gap state itself — pelvis 0.377 "
                   "m, on its FEET, trunk folded horizontal — and has "
                   "only to catch itself and come up. This is the "
                   "reverse curriculum done with a REAL rung: the state "
                   "is recorded off `fold` falling forward, not "
                   "interpolated, so it is a configuration the body "
                   "demonstrably passes through. It is not an "
                   "equilibrium, so the episode places it and goes "
                   "rather than settling — settling it just lets it "
                   "fall to 0.206 m, which is abandoning the rung, not "
                   "reaching it."),
        CurriculumStage(
            'the one hard move', 500_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold,rock_back,pike',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.6',
                     'MICRODUCK_ASIMOV_GETUP_HISTORY': '5',
                    },
            detail="One phase earlier: the pike — hips high, weight on "
                   "the feet and forearms, pelvis 0.300 m. This is the "
                   "only leg of the reference the motors cannot do "
                   "unaided at ANY timing: ramped through it at every "
                   "speed from 0.08 s to 0.60 s the pelvis never once "
                   "rises above the pike's own height. Everything after "
                   "it they can, now that the tail's timing is measured "
                   "rather than guessed."),
        CurriculumStage(
            'from the quadruped', 300_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold,rock_back,pike,quadruped',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.6',
                     'MICRODUCK_ASIMOV_GETUP_HISTORY': '5',
                    },
            detail="One phase earlier: on forearms and knees, pelvis "
                   "0.211 m. The motors already make this leg unaided, "
                   "so a policy that can do the rock-back should pick "
                   "it up nearly free."),
        CurriculumStage(
            'knees under the hips', 250_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold,rock_back,pike,quadruped,draw_knees_in',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.6',
                     'MICRODUCK_ASIMOV_GETUP_HISTORY': '5',
                    },
            detail="Earlier again, and a SMALL step on purpose. The "
                   "first chain went straight from the quadruped to the "
                   "pose cache and lost everything — it had reached "
                   "`rock_back` from the quadruped and went back to "
                   "achieving nothing at all. That jump moved the start "
                   "two waypoints AND swapped one authored pose for a "
                   "distribution of 220 real ones, in a single rung."),
        CurriculumStage(
            'forearms down', 250_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold,rock_back,pike,quadruped,draw_knees_in,plant_forearms',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.7',
                     'MICRODUCK_ASIMOV_GETUP_HISTORY': '5',
                    },
            detail="Flat on the floor with the elbows coming in under "
                   "the shoulders, pelvis 0.020 m. The last authored "
                   "rung before the cache."),
        CurriculumStage(
            'from where it fell', 400_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold,rock_back,pike,quadruped,draw_knees_in,plant_forearms,prone',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.9',
                     'MICRODUCK_ASIMOV_GETUP_HISTORY': '5',
                    },
            detail="Face down, from the cache — 220 real settled poses "
                   "rather than one authored one. The tolerance is 0.90 "
                   "here and not 0.60 for a measured reason: a cache "
                   "pose sits 0.78 rad of mean joint error from the "
                   "first waypoint, so at 0.60 every episode is cut on "
                   "its FIRST step for being a different fallen robot."),
        CurriculumStage(
            'hold the line', 300_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_START': 'fold,rock_back,pike,quadruped,draw_knees_in,plant_forearms,prone',
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.65',
                     'MICRODUCK_ASIMOV_GETUP_HISTORY': '5',
                    },
            detail="The same rung with the not-tracking cut tightened. "
                   "This is the ladder that matters — on the G1 kick "
                   "the reward alone did nothing and the termination "
                   "ladder took it from 0.10 m to 0.62 m. Only this "
                   "rung's policy is worth exporting."),
    ),
    default_steps=2_700_000,
    episode_s=8.0,
    success_metric="any episode achieving `fold` — the leg that two "
                   "chains and 4M steps of the 78-obs arm never "
                   "crossed. Matched steps against teach-"
                   "asimov_getup_ref-2f6e93, which IS the contract arm "
                   "of this pair",
)

_register(ASIMOV_GETUP_HIST)



# --------------------------------------------------------------- the handoff
#
# EVERY Asimov get-up ends by handing to Menlo's own walker, not just the one
# behavior that happened to declare it.
#
# `ASIMOV_GETUP_TURN` carried `handoff_fn` + `handoff_policy` and the other
# SIXTEEN get-up behaviors did not — including `asimov_getup` and
# `asimov_getup_whole`, which are the ones every working get-up in this repo
# was actually trained under. `viz_server.handoff_fn_for` returns None unless
# the behavior declares `handoff_fn`, so on the lab's stage those runs drove
# the get-up policy forever: Jonathan watched the finished robot SHAKE, at a
# measured **252 deg/s** on the hip yaws, while this session's acceptance
# numbers said 2.3 — because the scoring scripts wired the handoff by hand
# and the lab could not. The numbers and his eyes were both right about
# different programs.
#
# The deliverable was never "it stands up"; it is "it stands up and then
# stays there, still", and the handoff IS that second half (measured over 24
# episodes from prone: worst joint **252 -> 2.3 deg/s**, worst joint
# **16.3 -> 5.1 deg** off the default pose, with rises and stays unchanged at
# 24/24 and 22/24). Attaching it per-behavior meant the deliverable depended
# on which of seventeen near-identical entries a run happened to name.
#
# This attaches it to every get-up behavior that has not chosen its own.
# `_getup_handoff_due` delegates to `AsimovGetupEnv.handoff_ready`, which
# only exists on the get-up envs and only fires once the robot is actually
# up, so a curriculum rung that never stands never hands off.
#
# `Behavior` is `@dataclass(frozen=True)`, so this REPLACES and re-registers
# rather than assigning — assignment there is silently a no-op.
def _attach_getup_handoff() -> list[str]:
    import dataclasses

    from .core import BEHAVIORS

    touched = []
    for bid, b in list(BEHAVIORS.items()):
        if getattr(b, "robot", "") != "asimov1":
            continue
        if str(getattr(b, "task", "")) not in ("getup", "getup_ref",
                                               "getup_clip"):
            continue
        if getattr(b, "handoff_fn", None) is not None:
            continue                      # it chose its own; leave it alone
        _register(dataclasses.replace(
            b, handoff_fn=_getup_handoff_due,
            handoff_policy="asimov1-shipped"))
        touched.append(bid)
    return touched


ASIMOV_GETUP_HANDOFF_ATTACHED = _attach_getup_handoff()
