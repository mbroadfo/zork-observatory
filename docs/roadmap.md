# Where the Observatory goes next

*Written 30 September 2026, after the multi-level cognitive architecture proposal
(`NEW_ARCHITECTURE.md`) was revised against the September sweep.*

This document exists because the vision is larger than any one week of work and
keeps arriving in pieces. It records the destination, the rule for getting
there, and — most importantly — the things already tried, so that an architecture
that looks new on paper is not quietly a rerun of an arm that lost.

## The principle

> The Observatory grows by falsifiable cognitive increments, not by
> architectural completeness.

No new cognitive layer without an A/B experiment against the architecture below
it. Each addition states a hypothesis, a metric, a baseline, exactly one
intervention, and a result. An increment that cannot be falsified is not an
increment; it is a preference.

The reason this rule is strict rather than aspirational is in the next section.

## The constraint every plan must clear

`docs/experiments/2026-09-scaffold-sweep.md` is not background reading. It is
the binding precedent, and any proposal here has to say how it differs from what
already failed.

Twelve arms, one 14B model, 350 turns each. The bare model — no coaching, no
structured memory, no intervention — scored 2.7× the best scaffolded arm, and
rooms explored fell monotonically as scaffolding accumulated: 27, 17, 14, 11, 7.

The sweep's closing finding is the one that governs this roadmap:

> Every scaffold was measured against the thing it was designed to fix, and every
> one fixed it. Futility fell. Novelty rose. Repeats stopped. And the player got
> worse, every time.
>
> The metrics we chose to optimise were symptoms of a run going badly, not
> causes, and optimising them directly produced runs that failed more tidily.

Three specific precedents retire proposals that otherwise look attractive:

**Candidate sources as prompt content — tested, lost.** The `agenda` arm supplied
untried directions here, unused words from this room's text, and the parser's
accept/refuse record. It worked exactly as designed: futility 2.3%, the lowest in
the sweep; 170 distinct commands, the most of any model arm. It still wasted
62.6% of its turns, reached the house at turn 190 against bare's 124, and scored
10 against 40. *Novelty was not the missing ingredient.*

**Futility suppression — tested, lost.** The `nudge` arm re-asked once when the
model chose a command its own record showed doing nothing here. 259 nudges, 179
ignored (69%), and futility rose to 45.7% against 20.4% unnudged. Telling this
model it is repeating itself did not stop it.

**Structured memory replacing the transcript — tested, worst arm in the sweep.**
The episodic record filed `e → heading Behind House` and discarded the room
description as scenery, so the model saw each object once and never again. Two
takes in 350 turns follows directly. This refutes *that implementation*, not
memory: a structured memory that adds to the transcript and keeps descriptions is
untested and not refuted.

What survived is narrow and worth protecting: `--think off` (~10× faster and
better play), temperature 0.3 over 0.7 (score 10 → 15, house entry 190 → 18),
`--num-ctx` sized to keep the model on the card, the prompts ladder, and the
local-server handling.

## What already exists

So it is not rebuilt under a new name. Roughly Levels 1–2 of the proposed stack
are in place:

| Capability | Where |
|---|---|
| Ground truth from the Z-machine object tree, not from text | `session.py`, `world/objects.py` |
| Map: rooms, edges, blocked directions | `world/graph.py` |
| Map *shape*: radius, fragments, reachability, revisit ratio | `world/frontier.py` |
| Coverage: rooms and objects actually seen | `world/coverage.py` |
| Outcome classification: PROGRESS / BLOCKED / UNKNOWN_WORD / GRAMMAR / ABSENT_NOUN / INERT / META / FUTILE | `world/outcomes.py` |
| Parser vocabulary, with unknown-word and absent-noun held apart | `world/vocabulary.py` |
| Discovery ledger — the turn each ontological realisation becomes observable | `world/discovery.py` |
| Searchable transcript | `world/recall.py` |
| Event bus, trace writer, replay | `events.py`, `trace.py` |
| Mid-run agent handoff with usage banking | `Session.handoff` |
| Batch runs, seeds, temperature, budgets | `cli.py` (`--runs`, `--seed`, `--temperature`, `--turns`, `--max-cost`) |
| Read-only tools for an outside adviser | `mcp_server.py` |
| Prompt-hygiene ladder and its enforcement | `agents/prompts.py`, `tests/test_prompt_hygiene.py` |

Two consequences. Experiment 0 needs an aggregator over traces, not a harness.
And the escalation trigger already has its signal: `since_new_room`, which the
sweep's postscript named as *the measure the sweep should have been stopped on*.

## The target shape

Five layers, replacing the 28-section stack. Nothing below is built until an
experiment demands it.

**1. Ground truth.** Room, visible objects, inventory, map, accepted parser
behaviour, outcomes, coverage. No AI speculation lives here. Largely built.

**2. Candidate generator.** Produces possible commands without choosing among
them. Every candidate carries a *provenance* — the dashboard must be able to
answer "why is this command in the set", because that is what makes the
contamination arms legible.

**3. Action selector.** Fast, local. Ranks a closed set rather than inventing
from an open vocabulary. Where constrained scoring belongs.

**4. Progress / futility monitor.** Deterministic. Watches outcomes, repeats,
turns without map growth, turns without state change. Answers one question: *is
the current approach working?*

**5. Escalation reasoner.** A capable model, called only when layer 4 says the
local loop has stalled. Its output is a redirection, not a command — "local
manipulation appears exhausted, explore another unresolved route" — after which
control returns to the local loop.

Decision lineage wraps all five: ground truth → candidates → filter → ranking →
execution → outcome → progress → escalation, each step emitted on the existing
bus so a command can be explained rather than merely observed. This is the
Observatory's actual product.

## What to measure, and two traps

The sweep is explicit about which measures tracked real play: **the turn it got
inside the house, and how often it took something.** Dense behavioural measures
beat aggregate ratios.

Primary, for any arm:

- **Step at which the score first moved** (`scoring` in the discovery ledger).
  Confirmed 30 September against the sweep's traces: it reproduces the
  hand-derived "enters house" turn for every arm — 18, 80, 124, 190, 206, 239,
  317, and never for `episodic` — because on Zork the points are indoors. The
  sweep's cleanest signal, without a room name only one game has.
- **Takes** — objects actually acquired, not examined.
- **Discovery ledger timings** (`world/discovery.py`) — the turn each realisation
  becomes observable in behaviour. Works on a run that scores zero, and is
  comparable across games.
- **`dead`** — turns since the last new room. Also the stop condition.

Secondary: score, rooms, radius, futility, wasted percentage.

**Trap one: optimising a symptom.** Futility, novelty and repeat rates are
symptoms of a run going badly. Every arm that targeted one fixed it and played
worse. They are diagnostics, never objectives.

**Trap two: radius read alone.** `episodic` had the deepest map in the sweep —
radius 8, further out than `bare` ever reached — and scored zero. It explored,
then locked at turn 105 while its revisit ratio climbed to 91%. Depth without
continued growth is a run that is already over. Read radius beside `dead`.

A related caution the postscript records: `dirs/rm` is highest for the random
floor, because trying every direction everywhere is what random does. High is not
good.

## The ladder

Reordered from the original proposal, because the sweep already ran the
neighbours of three of its rungs.

### Experiment 0 — Baseline distributions

The sweep is one run per arm. Its own caveat: *a single 350-turn run is an
anecdote with good instrumentation, and "40 versus 10" is not a measurement to
quote.* Everything below needs three seeds of a frozen baseline first.

- **Hypothesis** — none; this establishes variance.
- **Deliverable** — an aggregator over `traces/` producing the metrics above per
  run, with spread across seeds.
- **Cost** — GPU time only. No new architecture.
- **Gate** — no arm below is reported until its effect exceeds baseline spread.

### Experiment 1 — Escalation

Promoted from third. It is the only proposal with no relative in the sweep, and
the only one that does not enlarge the prompt on every turn — which matters,
because enlarging the prompt every turn is precisely what made every scaffolded
arm worse.

- **Hypothesis** — rare frontier reasoning, invoked only on a measured stall, can
  rescue a local agent that per-turn scaffolding could not help.
- **Intervention** — when `dead` crosses a threshold, pause the local loop, ask a
  capable model for a *redirection* (not a command), inject it as one line, and
  return control. Nothing else changes.
- **Metric** — first entry indoors, takes, discovery timings. Plus a cost curve:
  dollars per unit of progress, against the measured $0.0127/turn for opus and
  $0.0027/turn for haiku playing every turn.
- **Why it might still fail** — `nudge` also interrupted the model and was
  ignored 69% of the time. The differences are that escalation fires rarely
  rather than constantly, and that its output is a change of objective rather
  than a complaint about the last command. If the redirection is ignored at
  `nudge` rates, that is the result.
- **Mostly built** — `Session.handoff` is an escalation mechanism with a human
  currently supplying the trigger.

### Experiment 2 — Closed-set selection

Kept, with lowered expectations stated up front.

- **Hypothesis** — the local model's weakness is open-vocabulary *generation*,
  not judgement; ranking a closed set is an easier task for a 14B model.
- **How it differs from what lost** — `agenda` put candidate *sources* into the
  prompt and still asked the model to compose a command. The sweep's own
  `candidates` arm asked the model for three commands of its own and filtered
  them ("the model had the same idea three ways"). Neither replaced generation
  with selection. This does: the model returns an index.
- **Intervention** — `world/candidates.py`, a pure function with no model and no
  network, returning `(command, provenance)`. The agent renders a numbered list
  and returns an index; invalid indices fall back to free text **and are
  counted**, because the fallback rate is itself the finding — as "zero lookups
  in 418 turns" was.
- **Risk to name now** — a closed set can only contain what the generator
  imagined. A naive player has to invent `move rug` before anything suggests rugs
  move. Arm A is a floor, not a ceiling; see the contamination arms below.

### Experiment 3 — Memory that adds rather than replaces

- **Hypothesis** — the episodic arm failed because it *discarded room
  descriptions*, not because structured memory is wrong. A record that sits
  beside the transcript and keeps descriptions should not reproduce the failure.
- **Metric** — takes, and whether objects named once are ever named again.
- **Precondition** — the transcript stays. This is additive or it is not run.

### Experiment 4 — Semantic duplicate recognition

Deferred, and the reason is written down: it is a more sophisticated `nudge`, and
`nudge` was ignored 69% of the time. Recognising that `push door` and `shove
door` are the same failed idea is only useful if the model acts on being told so,
and the evidence says it does not. Revisit only if Experiment 1 shows this model
acts on *any* interruption.

### Experiment 5 — Goal and strategy layer

Deferred. `agenda` is its nearest relative and lost. Revisit only after selection
works, and then as a persistent objective supplied *to the selector*, which is a
different mechanism from an objective printed in the prompt.

## Contamination rules

The memorisation audit is the research subject; anything that blurs it costs more
than it buys. Established this week: **opus recites Zork from training** — it
typed `echo` before ever trying the platinum bar, named treasures in rooms it had
not visited, and wrote "better: give to thief later" before banking the egg
anyway. **Haiku genuinely explores** — it found the grating by taking the leaves.
That contrast is only measurable because the arms are clean.

1. **No frontier model as the normal affordance generator.** Asking a model that
   has read Zork what one might do with a grating returns the walkthrough in
   costume, and because affordances sit upstream of selection, the audit can no
   longer see it. The concern is laundering recall, not using large models.
2. **Escalation output is a redirection, never a command.** Experiment 1's whole
   defensibility rests on this. A capable model that names objects or moves has
   played the game.
3. **Candidate order is neutral.** An ordered list is a hint channel. Order must
   be deterministic and seed-shuffled, and that belongs in a test beside the
   existing hygiene ones.
4. **`CLEAN_VERBS` is asserted, not argued.** The generic verb vocabulary
   (examine, take, drop, open, close, push, pull, move, read, attack, enter,
   climb) must be checked against `WORLD_NOUNS` in `tests/test_prompt_hygiene.py`,
   so the clean arm's action space is provably uncontaminated.
5. **`coached` remains the deliberately contaminated control**, and nothing below
   it foreshadows the game.

### The affordance arms

The withdrawn idea returns later as an experiment rather than a component,
because *how much world knowledge does an agent need before it can discover an
interactive fiction game's action space* is a better question than any single
implementation of affordances. Planned arms:

- **A** — observed vocabulary and ground truth only
- **B** — generic English action ontology (`CLEAN_VERBS`)
- **C** — lexical knowledge (WordNet-like), game-independent
- **D** — cold local model
- **E** — frontier model, as the contaminated control

Run after Experiment 2 has established whether selection helps at all.

## Standing constraints

- Prompts below `coached` contain no world nouns and no strategy; enforced by
  `tests/test_prompt_hygiene.py`.
- `roms/` and `assets/` are copyrighted and are never committed.
- Commits are authored as Mike Broadfoot and co-authored by the model that wrote
  them.
- The dashboard is not a side effect of the agent; the Observatory's product is
  the explanation of a command, not the command.

## Open questions

- Does this model act on *any* interruption? Experiment 1 answers it, and the
  answer governs Experiments 4 and 5.
- Is the 62.6% wasted-turn rate a property of the model or of the task? Untested
  against a larger local model.
- Does a closed set suppress the invention a naive player needs, and can that be
  measured rather than argued?
- What does the transcript actually do that structured memory did not? The sweep
  says descriptions staying in view for thirty turns. That is a testable claim
  about window content, not structure, and it has not been tested directly.
