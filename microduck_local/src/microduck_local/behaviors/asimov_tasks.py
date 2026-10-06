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
              "push off its arms", "arm powered getup"),
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
        CurriculumStage(
            "from everywhere", 3_000_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "mix"},
            detail="Half the episodes from where it fell, the rest spread over "
                   "every replay-validated rung from 0.146 m to 0.606 m. The "
                   "MIX goes first and not last, because warm starts freeze "
                   "VecNormalize: measured, the floor's own states reach "
                   "|z| 13.1 under a floor-trained normalizer while the rungs "
                   "sit at 8.4 and 3.9, so the widest distribution has to be "
                   "the one the stats are built from or a later stage sees "
                   "inputs that mean something else."),
        CurriculumStage(
            "off the ground", 2_000_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": "low"},
            detail="Only the bottom of the ladder — rungs at 0.146-0.26 m, "
                   "three to fifteen centimetres above where the robot lies. "
                   "This band did not exist until the search's own filter was "
                   "removed: it required the trunk above 0.15 m at 1.2 s, so "
                   "every rung near the floor was discarded by construction "
                   "and the first stage of the previous chain had nothing to "
                   "aim at. It stalled at 0.203 m and the floor rise rate was "
                   "19 of 36."),
        CurriculumStage(
            "from where it fell", 3_000_000,
            {"MICRODUCK_ASIMOV_GETUP_START_RUNG": ""},
            detail="The deliverable, and the only stage whose policy is worth "
                   "exporting. Scored from the floor with observation noise "
                   "and the 20 ms action delay, which is what "
                   "scripts/verify_run.py reports as THE DELIVERABLE — the "
                   "previous chain's own env flattered it 8/36 against 1/36 "
                   "here, because its last stage started on a mix."),
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
                    },
            detail="Half the drift allowed."),
        CurriculumStage(
            'follow tightly', 1_500_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.35',
                    },
            detail="Tighter again. This is the rung the G1 kick's "
                   "equivalent ladder was won on."),
        CurriculumStage(
            'hold the line', 1_500_000,
            {
                     'MICRODUCK_ASIMOV_GETUP_TRACK_TOL': '0.25',
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

