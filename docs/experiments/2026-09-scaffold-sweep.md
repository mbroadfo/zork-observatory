# The scaffold sweep, September 2026

Twelve configurations of the same 14B model on the same game, 350 turns each,
nine hours of GPU. It was run to answer one question: of everything built to
help a local model play Zork, what actually helps?

The answer was none of it. The bare model — no coaching, no structured memory,
no intervention — scored 2.7× the best scaffolded arm, and each capability
added made the run worse than the one below it.

This document exists so the same things are not built again. The code is
recoverable at the tag `scaffolds-2026-09`; the traces are in `traces/sweep/`.

## Conditions

qwen3:14b through Ollama, thinking off, 8k context, seed 12345, 3 lives,
`jericho` on `zork1.z5`, 350 turns. Temperature 0.7 unless stated. One run per
arm — see *What this does not establish*.

## Results

```
arm                turns score rooms  obj die  futile  wasted distinct  nudge  ign  skip  s/turn
bare                 350    40    27   29   1   20.4%   34.3%       58      0    0     0     2.6
temp03               350    15    15   23   0    4.3%   61.7%      156     52    4   337     5.2
agenda               350    10    17   19   0    2.3%   62.6%      170     35    3   343     5.2
coached              350    10    17   22   0   10.0%   22.9%       55      0    0     0     2.9
candidates           350    10    14   18   0   11.7%   63.4%      132     56   11   389     4.7
nudge                350    10    11   14   1   45.7%   68.0%       91    259  179     0     3.3
cold-agenda          350    10    10   17   0   31.4%   80.3%      197    189  113   797    35.8
coached-episodic     350    10     7   13   0   45.7%   49.1%       25      0    0     0     3.0
floor (random)       350     0    15    8   0   23.7%   72.3%      123      0    0     0     0.0
episodic             350     0    12    6   0   23.7%   29.7%       16      0    0     0     2.8
journal-run1/2        58     0     8   13   4    4.8%   31.0%       53     32    7   108     4.0
```

Rooms explored fall monotonically as scaffolding accumulates: 27, 17, 14, 11, 7.

## What decided every run: one window

The only way into the house is a window at Behind House, ajar in the first
description of that room. Points are indoors. The turn each arm got inside:

| arm | enters house | takes | opens | looks |
|---|---|---|---|---|
| bare | **124** | 43 | 16 | 102 |
| cold-agenda | 80 | 8 | 5 | 73 |
| agenda | 190 | 7 | 13 | 123 |
| candidates | 206 | 5 | 34 | 56 |
| nudge | 239 | 5 | 45 | 110 |
| coached | **317** | 9 | 5 | 158 |
| episodic | **never** | 2 | 1 | 116 |

`bare` named the window in four commands across nine visits. `coached` once in
eight visits. `episodic` never, in two visits.

## Why the record failed, mechanically

The episodic record stored *what was typed and what came back*. A move was
filed as `e → heading Behind House`; the room description that came with it was
discarded as scenery.

The consequence, with recall set to one exchange: **the model saw each object
exactly once, on the turn it arrived, and never again.** Its record of Behind
House said a direction led there and nothing about a window. Two takes in 350
turns follows directly — you cannot take what you cannot see.

The transcript, which the record replaced, kept every description in view for
thirty turns. That is why the bare model kept coming back to the window.

This is a finding about *this implementation*, not about memory. The transcript
is memory, and it is the thing that worked. A structured memory that **adds to**
the transcript rather than replacing it, and that keeps descriptions, has not
been tested and is not refuted here.

## What each removed feature actually did

**nudge** — re-ask once when the model picks a command its own record shows
doing nothing here. 259 nudges, 179 ignored (69%), futility 45.7% against
20.4% unnudged. Telling this model it is repeating itself did not stop it.

**candidates** — ask for three ranked commands, play the first not known to be
inert. 389 commands skipped over 350 turns. Score 10, latency up 80%. The model
had the same idea three ways.

**agenda + vocabulary** — untried directions here, unused words from this
room's text, and what the parser has accepted or refused. It worked as designed:
futility 2.3% (the lowest of any arm) and 170 distinct commands (the most of any
model arm). It still wasted 62.6% of its turns. Novelty was not the missing
ingredient.

**episodic recall** — see above. The worst arm of the sweep.

**journal** — never tested. `cli.py` evaluated `fresh = (mode == "new")` per run,
so run 2 archived run 1's journal and began empty. The two runs came out
identical in every metric, which does confirm the setup is deterministic at a
fixed seed and temperature.

## What survived

- **`--think off`.** The largest single effect measured: ~10× faster (2.6 s/turn
  against 30+) and better play. qwen3 left to its own default spends thousands
  of characters of thinking per turn and plays worse for it.
- **`--temperature`.** The only dial that moved anything the right way: 0.7 → 0.3
  within the same scaffolded configuration took score 10 → 15, takes 7 → 13,
  and entry to the house from turn 190 → 18.
- **`--num-ctx`.** What keeps a 14B model entirely on a 12GB card. At 16k it
  spills to the CPU; gpt-oss:20b spills regardless and took 5h16m for one arm.
- **Local-server handling** — prose fallback when a server cannot carry a reply
  made under a schema, one retry before a failed call costs a turn, the
  output-ceiling retry, the near-the-window warning.
- **The prompts ladder**, with two lines cut from `coached`: "work each new
  description…", which the model satisfied by looking (158 looks, 9 takes), and
  the deposit hint, which sent it hunting a trophy case before it owned a
  treasure.
- **Lessons and the notebook.** Untested by this sweep, and the project's actual
  subject: a memory that is a belief rather than a fact. One run carried the
  note *"look for a light source before moving into the dark"* into a forest
  that has none and spent twenty turns hunting a lamp — memory steering a run
  wrongly, recorded in the agent's own words.

## Postscript: the same traces, read structurally

*Added 24 September 2026, after `world/frontier.py` was written. Nothing was
re-run; these are the sweep's own traces measured with something the sweep did
not have. `python tools/structure.py`.*

```text
arm                turns score rooms radius frag reach revisit dirs/rm once walls1  dead
episodic             350     0    12      8    1     8     91%    3.58    4      6   296
bare                 350    40    27      7    3     4     79%    2.48   10     11   101
agenda               350    10    17      7    1    16     76%    6.06    1     63    86
coached              350    10    17      7    1    16     86%     3.0    1     10    27
floor                350     0    15      6    1    15     81%    7.93    0     38    48
nudge                350    10    11      5    1     8     67%    1.91    5      2   105
temp03               350    15    15      4    1    15     81%    7.87    2     65   126
candidates           350    10    14      4    1    14     84%    3.43    2     11   136
coached-episodic     350    10     7      4    1     1     40%    1.43    4      1   327
cold-agenda          350    10    10      3    2     2     53%     2.4    3      6   267
```

Three things fall out, and the first one corrects the write-up above.

**The episodic arm was not a slow explorer.** It has the *deepest* map in the
sweep — radius 8, further from the front door than `bare` ever got — and it got
there fast. Sampled every 35 turns, it reached twelve rooms and radius 8 by
turn 70 and then did not find another room for 280 turns while its revisit
ratio climbed 62% → 91%. The failure was not sluggishness. It explored, and
then it locked, and the run was decidedly over at turn 105 in a way the score
(zero, throughout) could not distinguish from bad luck.

**`bare` won by breadth, not by depth.** Its radius plateaued at 7 by turn 140,
but its room count kept climbing to turn 280 — it was filling in the map
sideways while the others circled. Its fragment count going 1 → 2 → 3 around
turn 210 is the one-way descent underground, which is also where its 40 points
came from.

**`dead` — turns elapsed since the last new room — is the measure the sweep
should have been stopped on.** Nine hours of GPU bought roughly two thousand
turns that came after the map had stopped growing. Wiring it into the live event
stream means the next sweep can halt an arm that has been dead for a hundred
turns instead of paying for the rest of it.

One honest caution about `dirs/rm`: the random floor scores 7.93 on it, the
highest in the table, because trying every direction everywhere is exactly what
random does. High is not good. It measures thoroughness, and thoroughness
without direction is what the floor is.

## What this does not establish

One run per arm. Zork's thief and combat are random, and a single 350-turn run
is an anecdote with good instrumentation. The *direction* of the big effects is
safe — a 350-turn behavioural difference like never naming the window is not a
lucky roll — but "40 versus 10" is not a measurement to quote. Before any of
this is repeated as a number, the top configurations want three seeds each.

Nor does it say anything about larger models, about thinking left on, or about
a memory built the other way round.

## The shape of the mistake

Every scaffold was measured against the thing it was designed to fix, and every
one fixed it. Futility fell. Novelty rose. Repeats stopped.

And the player got worse, every time.

The metrics we chose to optimise were symptoms of a run going badly, not causes,
and optimising them directly produced runs that failed more tidily. The only
measures that tracked actual play were the dense behavioural ones — the turn it
got inside the house, and how often it took something.
