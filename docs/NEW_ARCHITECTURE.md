Absolutely. Below is the architecture summary I would hand directly to the Claude session. I’ve deliberately framed it as an **architectural design document rather than an implementation prescription**. The important idea is that Zork play is decomposed into specialized cognitive functions, with different model types used only where their characteristics are appropriate.

# Zork Observatory — Multi-Level Cognitive Architecture

## Purpose

The goal of Zork Observatory should no longer be simply to ask an LLM to play Zork by repeatedly generating the next command.

Instead, the Observatory should become an experimental **multi-level cognitive architecture** in which several specialized components collaborate to perceive the game, maintain reliable state, generate possible actions, choose among alternatives, reason about puzzles, discover goals, learn from failed attempts, and execute commands.

The architecture should deliberately distinguish between:

- what the game has explicitly told the player;
- what the system has inferred;
- what it currently believes;
- what it is attempting to accomplish;
- what actions are possible;
- which actions have already failed;
- why a particular action was selected;
- when fast decision-making is sufficient;
- and when deeper reasoning is required.

Different technologies — deterministic software, conventional LLMs, reasoning LLMs, Jev/System-One-style models, CLM/dual-encoder approaches, embeddings/RAG, and potentially specialized classifiers — should each perform the functions they are best suited to perform.

The overall architecture should therefore resemble a **cognitive system with multiple loops operating at different speeds**, rather than a collection of interchangeable AI agents.

---

# 1. Foundational principle: separate truth, belief, reasoning, and action

The most important architectural boundary is between **ground truth** and **AI interpretation**.

Zork provides certain information directly to the player. Examples include:

- the current room description;
- objects explicitly visible in the room;
- inventory contents;
- exits described in the text;
- game responses to commands;
- score changes;
- death;
- changes to visible objects;
- potentially other directly observable state.

These observations should be maintained in a structured **Ground Truth World State**.

The AI should never be required to remember facts that can instead be retrieved reliably from this store.

Alongside ground truth should exist separate stores for:

**Beliefs** — conclusions that appear likely but have not been directly demonstrated.

**Hypotheses** — explanations the reasoning system wants to test.

**Goals** — things the system currently wants to accomplish.

**Action history** — everything it has tried and what resulted.

**Semantic memory** — lessons, clues, previous reasoning, analogous situations and other information that is useful but cannot be represented simply as current state.

This distinction is essential to making the system observable and debuggable.

---

# 2. The Ground Truth State Layer

The Ground Truth State Layer is primarily conventional software rather than AI.

It represents the game's currently known reality from the player's perspective.

It should include concepts such as:

- current location;
- current inventory;
- visible objects;
- known exits;
- known rooms;
- known connections between rooms;
- object locations;
- known object properties;
- current score;
- current turn;
- alive/dead status;
- lighting conditions;
- known resource states;
- significant environmental changes.

The state should be versioned so that the system can determine whether an action that failed previously might now deserve reconsideration because relevant circumstances have changed.

This state store becomes the authoritative source for questions such as:

- Where am I?
- What am I carrying?
- What can I currently see?
- Have I seen this object before?
- Has this room changed?
- What happened the last time I tried something with this object?

Models should query this state rather than reconstructing it from their context window.

---

# 3. Observation and Perception Layer

Zork's text output must first be interpreted into changes to ground truth.

This layer determines things such as:

- a new room has been entered;
- an object has appeared;
- an object has disappeared;
- an item was acquired;
- an item was dropped;
- a passage opened;
- a command had no effect;
- a command was rejected by the parser;
- an action changed the world;
- an action caused damage or death.

This layer should favor deterministic parsing where possible, supplemented by a lightweight semantic model when the game's natural-language response is ambiguous.

The output is not a plan or an action.

Its job is simply:

> **What did Zork just tell us happened?**

---

# 4. Semantic Memory and RAG Layer

Not everything belongs in structured state.

The system also needs longer-term semantic memory for questions such as:

- Have we encountered clues mentioning this object?
- What happened previously around the grating?
- Have we encountered a similar locked barrier?
- What theories have we formed about this puzzle?
- Did some earlier room contain something that now seems relevant?
- What previously unsuccessful strategy might become relevant again?

This is the appropriate role for embeddings and RAG.

The architecture should therefore distinguish:

**Exact state retrieval** for factual current information.

**Semantic retrieval** for fuzzy historical or conceptual information.

RAG should augment ground truth, not replace it.

---

# 5. Goal Discovery Layer

One of the most interesting aspects of Zork is that a player initially may not know the real objective.

The game does not simply begin with:

> Collect these specific treasures and place them in the trophy case.

A capable agent should therefore be able to **infer the purpose of the game from evidence**.

Initially it might possess only very generic standing objectives:

- survive;
- explore;
- understand the environment;
- acquire potentially useful objects;
- increase knowledge;
- avoid wasting scarce resources.

Over time it may observe:

- score changes;
- unusual objects;
- rewards associated with particular actions;
- the importance of the trophy case;
- patterns in what constitutes progress.

From this it can develop hypotheses about higher-level goals.

Eventually the agent might infer that finding treasures and returning them to the trophy case is central to completing the game.

This layer should be relatively slow and should probably use a capable reasoning LLM.

It represents something closer to:

> **What is this game asking me to accomplish?**

rather than:

> What command should I type next?

---

# 6. Strategic Planning Layer

Once goals exist, the strategic planner decides what broad objective should currently receive attention.

Examples might include:

- explore an unmapped region;
- investigate the mysterious grating;
- acquire a light source;
- find a way through a locked barrier;
- recover an object left elsewhere;
- solve a particular puzzle;
- transport a treasure back to the trophy case;
- escape an immediate dangerous area.

Strategic reasoning should occur much less frequently than individual commands.

A strategic objective may govern many tactical actions.

The strategic planner should be capable of abandoning or suspending a goal when evidence indicates that it currently cannot be completed.

For instance:

> Open the living-room door

may eventually become:

> This barrier cannot currently be overcome.

which then becomes:

> Search elsewhere for information, tools, keys or another route.

That distinction prevents the fast action loop from obsessively attacking an unsolvable local problem.

---

# 7. Exploration Controller

Exploration is sufficiently important that it should be treated as its own specialized function.

Its question is:

> **Where should attention be directed next?**

It evaluates things such as:

- unexplored exits;
- partially explored rooms;
- unresolved objects;
- suspicious environmental features;
- dead ends;
- locations related to current hypotheses;
- locations likely to contain useful resources.

A Jev/System-One-style model could be particularly effective here because the problem often consists of ranking a finite set of candidate destinations or investigative opportunities.

The explorer need not understand every puzzle.

Its job is to continuously identify **where information or progress is likely to exist**.

---

# 8. Puzzle and Hypothesis Reasoning Layer

This is where the most powerful reasoning model belongs.

The puzzle reasoner should activate when fast local strategies stop producing progress.

For example, repeated attempts involving a grating may establish that:

- the grating is real and interactable;
- several obvious actions have failed;
- the parser understands some relevant verbs;
- local manipulation is not changing the state.

At this point the system should stop generating slight variations of the same command.

The puzzle reasoner instead develops hypotheses such as:

- the grating may be locked;
- it may require an object found elsewhere;
- a remote mechanism may control it;
- it may need to be approached from another direction;
- an earlier clue may describe its solution.

Its output should principally be **hypotheses and investigative objectives**, not raw Zork commands.

Those hypotheses flow back into strategic and exploration layers.

This is the primary System-Two function in the architecture.

---

# 9. Affordance Generation Layer

The existence of a noun does not directly tell the player which verbs matter.

This is a core Zork challenge.

A human sees:

- rug;
- mailbox;
- sword;
- rope;
- grating;
- bottle;
- door;

and automatically imagines actions that could apply to those things.

An AI architecture needs an explicit equivalent.

The Affordance Generator answers:

> **What kinds of actions could plausibly be performed on this object?**

This is a candidate-generation function.

A reasoning or generative LLM can produce plausible action families based on:

- the object;
- its description;
- the surrounding room;
- inventory;
- current goal;
- general-world knowledge.

It should not necessarily be run every turn.

Affordances can become part of a reusable semantic ontology.

For example, knowledge about doors, containers, ropes, weapons and movable objects can be reused whenever similar objects are encountered.

The system can therefore gradually build its own learned **Zork affordance vocabulary**.

---

# 10. Parser Vocabulary and Command Knowledge

Human affordances and Zork's parser vocabulary are not identical.

The architecture should explicitly learn the difference between:

> A reasonable action in the world.

and:

> An action that the Zork parser knows how to express.

This layer learns what the game understands.

A parser rejection teaches something different from an unsuccessful physical action.

For example:

**Unknown verb**

means the attempted linguistic representation is invalid.

**Recognized command with no effect**

means the parser accepted the action, but the current world state prevented progress.

That distinction should become persistent knowledge.

Over time the system develops a library mapping semantic intentions onto vocabulary that Zork actually accepts.

---

# 11. Candidate Action Construction

At this point several layers converge.

The system possesses:

- a current strategic objective;
- current ground truth;
- known nouns;
- candidate affordances;
- parser knowledge;
- historical results.

From these it can construct a finite candidate set of **plausible next intentions/actions**.

This is an important architectural transition.

The system stops asking an LLM:

> Invent the next Zork command.

Instead it asks:

> Of these plausible actions, which best advances the current objective?

This is where Jev/System-One architecture becomes especially compelling.

---

# 12. Tactical System-One Controller

The Tactical Controller operates frequently and quickly.

It chooses among the currently viable actions.

It should ideally perform **selection rather than generation**.

Inputs may include:

- current objective;
- current state;
- candidate affordances;
- known action outcomes;
- current risk;
- resource constraints;
- recent history.

The tactical controller may use:

- Jev;
- a Jev-like constrained classification architecture;
- a CLM;
- a dual encoder;
- a locally hosted open model operating over constrained outputs;
- or other low-latency discriminative models.

Its purpose is:

> **Given what we're trying to accomplish, which currently valid action is best?**

This is analogous to the tactical Jev used in TypeSafe's Doom example.

---

# 13. Novelty and Fruitless-Action Layer

Zork agents should explicitly remember what they have already tried.

This layer prevents the system from repeatedly performing actions that have already proven useless.

However, simple command string deduplication is insufficient.

Different phrases may represent the same semantic attempt.

The architecture therefore needs to identify:

- exact repeated commands;
- semantically equivalent actions;
- variants of previously failed strategies;
- actions worth retrying because relevant state has changed.

This could combine deterministic history with a Jev/CLM-like semantic classifier.

The critical concept is:

> **Fruitless under the current state — not fruitless forever.**

If the world changes, previously unsuccessful actions may become useful.

This allows the system to learn without becoming permanently rigid.

It also creates an excellent Observatory metric: how often does an agent repeat an action or semantic strategy despite evidence that it is not working?

---

# 14. Risk and Survival Layer

"Don't die" should not be merely an implicit instruction in the strategic prompt.

Survival should be an independent, persistent control function.

A Risk Controller can evaluate proposed actions in terms of:

- possible death;
- resource consumption;
- irreversible consequences;
- entering darkness;
- potential combat;
- loss of important inventory;
- triggering timers;
- wasting finite consumables;
- uncertain hazards.

The risk system may allow some experimentation while preventing obviously reckless behavior.

Risk tolerance itself could become configurable, allowing experiments involving conservative versus adventurous agents.

---

# 15. Information-Gathering Actions

The architecture should understand that sometimes the best action is not progress but **reducing uncertainty**.

Examples include:

- LOOK;
- INVENTORY;
- examining an object;
- revisiting a room;
- checking whether an earlier state still holds.

Models should not rely on uncertain memory when the game itself can cheaply provide authoritative information.

The system should therefore have an explicit concept of **verification**.

If confidence in an important state assumption becomes low, the agent can deliberately obtain new ground truth.

This creates a very human-like distinction between:

> I remember having the lamp.

and:

> I should check whether I still have the lamp.

---

# 16. Command Realization Layer

The final layer converts a selected semantic action into a command that Zork understands.

Its job is not strategic reasoning.

It is analogous to a compiler lowering an abstract intention into executable syntax.

The architecture can therefore be thought of as progressively transforming:

**Goal**

into

**Strategy**

into

**Intent**

into

**Affordance**

into

**Selected Action**

into

**Zork Command**

This keeps parser syntax separate from higher-order reasoning.

---

# 17. Execution and Feedback Loop

Every executed command produces new observations.

Those observations flow immediately back into:

- the ground-truth store;
- action history;
- parser knowledge;
- object knowledge;
- semantic memory;
- exploration state;
- current hypotheses;
- confidence levels.

Each command therefore becomes an experiment.

The architecture does not merely act.

It learns from the consequences of acting.

---

# 18. Escalation between cognitive levels

A central design principle should be **escalation rather than constant deep reasoning**.

Most game turns should not require an expensive reasoning model.

The hierarchy could conceptually behave as follows:

**Fast tactical model has a clear choice**

→ execute.

**Tactical choices are weak or exhausted**

→ invoke puzzle reasoning.

**Puzzle cannot currently be solved**

→ invoke strategy/exploration.

**Strategy encounters fundamentally new evidence**

→ reconsider broader goals.

This creates several timescales:

### Immediate loop

"What do I do right now?"

System-One / Jev / CLM.

### Local problem loop

"How do I accomplish the current objective?"

Tactical planning plus puzzle reasoning.

### Exploration loop

"Where should I go next to find progress?"

Explorer.

### Strategic loop

"What should I be working on?"

Strategic reasoning LLM.

### Meta-goal loop

"What does progress in this world actually mean?"

Goal-discovery reasoning.

This resembles a genuine cognitive hierarchy much more than a traditional LLM tool loop.

---

# 19. Multiple model technologies should compete experimentally

Zork Observatory should avoid hard-wiring Jev as the answer to every problem.

Part of the experiment should be determining which model architecture performs each task best.

For example:

| Cognitive task | Candidate technology |
|---|---|
| Ground truth | Deterministic software |
| Observation interpretation | Parser / small model |
| Semantic memory | Embeddings + RAG |
| Affordance generation | Generative/reasoning LLM |
| Affordance ranking | Jev / CLM / classifier |
| Exploration ranking | Jev / dual encoder |
| Tactical selection | Jev / System-One model |
| Semantic duplicate detection | CLM / embeddings |
| Risk classification | Jev / classifier |
| Puzzle solving | Reasoning LLM |
| Strategic planning | Reasoning LLM |
| Goal discovery | Reasoning LLM |
| Command realization | Deterministic mapping + learned parser vocabulary |

The Observatory can eventually compare multiple implementations for the same cognitive function.

That makes it an architecture research platform rather than just an application.

---

# 20. The Dashboard becomes as important as the agent

The Observatory dashboard should evolve from showing gameplay to showing the **cognitive machinery behind gameplay**.

At any moment an observer should be able to understand:

> What does the system know?

> What does it believe?

> What is it trying to accomplish?

> Which subsystem is currently active?

> What did that subsystem receive?

> What did it decide?

> What alternatives did it reject?

> How confident was it?

> Why was deeper reasoning invoked?

> What changed after the resulting command?

The dashboard should therefore show the system as a **live hierarchy of interacting cognitive components**.

---

# 21. Real-time model telemetry

Each model or cognitive service should have its own observable panel.

For every invocation, capture at least:

**Identity**

- model/component;
- model technology;
- version/configuration;
- role in the architecture.

**Input**

- state supplied;
- candidate choices;
- retrieved memory;
- current objective;
- relevant context.

**Output**

- selected decision;
- ranked alternatives;
- confidence/probability where available;
- hypotheses produced;
- strategic goal changes;
- escalations requested.

**Performance**

- latency;
- tokens where applicable;
- inference cost;
- GPU/CPU usage where available;
- context size.

**Outcome**

- command ultimately generated;
- Zork response;
- world-state change;
- whether the action advanced the current objective;
- whether the choice later proved useful or fruitless.

This creates an end-to-end causal trace from perception through decision to consequence.

---

# 22. A live "cognitive stack" view

One dashboard view should show the current state of every layer simultaneously.

Conceptually:

### Goal Discovery
Current inferred game objective and confidence.

### Strategic Planner
Current major objective.

### Explorer
Top candidate regions or unresolved locations.

### Puzzle Reasoner
Current active hypotheses.

### Tactical Controller
Current tactical intention.

### Affordance Layer
Possible actions for the relevant objects.

### Novelty Filter
Actions suppressed because they duplicate previous failures.

### Risk Layer
Current risk assessment.

### Command Layer
Chosen executable command.

### Ground Truth
What the game currently tells us is true.

This would allow a person watching the agent to understand **how the next command emerged from the hierarchy**.

---

# 23. Decision lineage

Every Zork command should have a complete lineage.

The system should be able to reconstruct:

> We are attempting to open the grating because the strategic planner selected "investigate underground access"; the explorer identified the grating as the highest-value unresolved location; the affordance model proposed MOVE, OPEN and LIFT; action history eliminated MOVE because it had already failed in the same state; tactical Jev ranked OPEN highest; risk was low; command realization emitted OPEN GRATING.

That lineage is perhaps the single most important Observatory capability.

Without it, we merely watch an AI play.

With it, we can understand **why the architecture behaved as it did**.

---

# 24. Failure attribution

The dashboard should also help identify which layer caused a poor action.

An unsuccessful move could result from:

- incorrect ground truth;
- poor observation extraction;
- missing memory;
- incorrect strategic priority;
- incorrect puzzle hypothesis;
- missing affordance;
- poor tactical ranking;
- failure to recognize duplication;
- poor risk estimation;
- incorrect parser translation.

This turns failures such as "the agent is stuck at the grating" into measurable architectural diagnoses.

That is precisely what makes the Observatory useful for improving agents.

---

# 25. Confidence and uncertainty should be first-class concepts

Where possible, discriminative models should expose meaningful probabilities or relative scores.

This enables the architecture to recognize uncertainty.

For example:

- very high tactical confidence → act immediately;
- moderate tactical confidence → obtain more information;
- low tactical confidence → invoke System-Two reasoning;
- conflicting models → escalate or experiment.

Uncertainty should therefore control when the architecture spends additional computation.

This is one of the most important potential benefits of a Jev-like layer.

---

# 26. Allow strange actions — deliberately

The architecture should not eliminate every unusual action.

Some exploration should remain.

An affordance model might plausibly propose odd actions involving a mailbox, sword, rug or door.

Those experiments could be entertaining, but they are also scientifically useful.

They reveal:

- model priors;
- affordance errors;
- parser vocabulary;
- unexpected game mechanics;
- exploration behavior;
- novelty-seeking tendencies.

The important distinction is between **creative experimentation** and **mindless repetition**.

The first can discover something.

The second should be detected and suppressed.

The Observatory should make that distinction measurable.

---

# 27. Model replacement should be easy

Every cognitive function should be behind a clear architectural boundary.

That enables experiments such as:

**Tactical control**

Jev versus Qwen logits versus CLM versus conventional LLM.

**Exploration**

dual encoder versus Jev.

**Duplicate detection**

embeddings versus CLM.

**Puzzle reasoning**

Claude versus Qwen versus another reasoning model.

The game run becomes a benchmark for model architecture choices.

---

# 28. Long-term experimental value

Eventually the Observatory should make it possible to compare agent architectures using metrics far richer than simply the final Zork score.

Potential measurements include:

- rooms discovered;
- unique objects investigated;
- puzzles solved;
- treasures discovered;
- treasures deposited;
- deaths;
- turns;
- commands;
- invalid parser commands;
- exact repeated commands;
- semantic repetitions;
- fruitless action rate;
- strategic goal changes;
- puzzle escalations;
- System-One versus System-Two invocation ratio;
- reasoning latency;
- total generated tokens;
- average decision latency;
- confidence calibration;
- resource waste;
- successful hypothesis rate;
- map coverage;
- information-gathering actions;
- unnecessary verification;
- recovery from failed strategies.

This allows a meaningful comparison between:

- pure LLM agents;
- scripted agents;
- System-One agents;
- hybrid System-One/System-Two architectures;
- multi-model cognitive architectures.

---

# Architectural summary

The refined Zork Observatory architecture can ultimately be viewed as five broad levels:

### Level 1 — Reality

Zork itself, observation parsing and authoritative ground-truth state.

### Level 2 — Memory and knowledge

World model, action history, parser knowledge, semantic memory and RAG.

### Level 3 — Fast cognition

Affordance ranking, exploration ranking, tactical choice, novelty detection, risk and other Jev/CLM/System-One functions.

### Level 4 — Deliberative cognition

Puzzle reasoning, hypothesis formation, strategic planning and goal discovery using more capable reasoning LLMs.

### Level 5 — Observatory

A real-time instrumentation layer exposing the inputs, outputs, confidence, latency, state transitions, decisions, escalations and consequences of every cognitive component.

The key architectural idea is:

> **Do not ask one model to play Zork. Build a system in which specialized cognitive processes collectively play Zork.**

And the key research idea is:

> **Make every stage observable enough that when the system succeeds or fails, we can identify which cognitive process was responsible.**

That turns Zork Observatory from an interesting agent demonstration into something much closer to an **experimental workbench for System-One/System-Two AI architecture** — exactly the sort of project where Jev, CLMs, local open models, reasoning LLMs, RAG, structured state and conventional software can all be compared in a controlled environment.