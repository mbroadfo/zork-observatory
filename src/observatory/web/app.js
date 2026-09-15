/* Zork Observatory — front end.
 *
 * The browser is a pure consumer of the event stream. It holds no game logic:
 * every pixel here is derived from events the session emitted, which is why a
 * recorded trace and a live run render identically.
 */

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ *
 * Map layout
 *
 * Rooms are placed by compass direction rather than by a force-directed
 * layout. A text adventure map has real geometry — north is up — and a
 * force layout throws that away, producing a blob that shuffles every time
 * a room is added. Positions here are sticky: once a room is placed it
 * never moves, so the map grows outward instead of rearranging itself
 * under the viewer.
 * ------------------------------------------------------------------ */

/* Rooms live on an integer lattice, not in free pixel space.
 *
 * Occupancy has to be checked at the same resolution rooms are drawn at.
 * Placing on a continuous plane and rounding for collision lets two rooms sit
 * twelve pixels apart, pass the "different cell" test, and overlap on screen.
 * Integer cells make "is this taken" exact, and the pixel conversion happens
 * once, at render. */
const CELL_W = 152;   // > node width (96), so horizontal neighbours never touch
const CELL_H = 96;    // > node height (40)

const VEC = {
  north: [0, -1], south: [0, 1], east: [1, 0], west: [-1, 0],
  northeast: [1, -1], northwest: [-1, -1],
  southeast: [1, 1], southwest: [-1, 1],
  // Vertical moves borrow a diagonal cell. The dashed cyan edge and the level
  // tint carry the real meaning; this only keeps the two rooms apart.
  up: [1, -1], down: [1, 1],
  in: [1, 1], out: [-1, -1],
};

const LEVEL_DELTA = { up: 1, down: -1 };
const VERTICAL = new Set(["up", "down", "in", "out"]);

// Rings of lattice offsets, nearest first — where a room goes when the cell
// its direction points at is already taken. Interior/exterior rooms overlap
// constantly in these games, so this path is well travelled.
const RINGS = (() => {
  const out = [];
  for (let r = 1; r <= 8; r++) {
    const ring = [];
    for (let dx = -r; dx <= r; dx++) {
      for (let dy = -r; dy <= r; dy++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) === r) ring.push([dx, dy]);
      }
    }
    ring.sort((a, b) => a[0] ** 2 + a[1] ** 2 - (b[0] ** 2 + b[1] ** 2));
    out.push(...ring);
  }
  return out;
})();

const state = {
  rooms: new Map(),        // id -> {id, name, dark, deaths, visits, x, y, level, placed}
  edges: new Map(),        // key -> {src, dst, direction, reciprocal}
  blocked: new Map(),      // key -> {src, direction, message}
  checkpoints: new Map(),  // id -> {id, label, turn, score, auto}
  discoveries: new Map(),  // key -> {key, label, reveals, turn, evidence}
  discoveriesSeen: new Set(),
  memory: [],              // lessons the agent kept across rollbacks
  occupied: new Set(),     // "col,row" lattice cells, so rooms never stack
  current: null,
  session: null,
  showThoughts: true,
  pendingThought: null,
};

/** Claim the lattice cell at (col,row), or the nearest free one. */
function claim(col, row) {
  const key = (c, r) => `${c},${r}`;
  if (!state.occupied.has(key(col, row))) {
    state.occupied.add(key(col, row));
    return [col, row];
  }
  for (const [dx, dy] of RINGS) {
    if (!state.occupied.has(key(col + dx, row + dy))) {
      state.occupied.add(key(col + dx, row + dy));
      return [col + dx, row + dy];
    }
  }
  state.occupied.add(key(col, row));
  return [col, row];
}

function assign(room, col, row, level) {
  const [c, r] = claim(col, row);
  room.col = c;
  room.row = r;
  room.x = c * CELL_W;
  room.y = r * CELL_H;
  room.level = level;
  room.placed = true;
}

function placeRoom(id) {
  const room = state.rooms.get(id);
  if (!room || room.placed) return;

  // Prefer placing relative to a neighbour whose position we already know.
  for (const edge of state.edges.values()) {
    if (edge.dst !== id) continue;
    const src = state.rooms.get(edge.src);
    if (!src || !src.placed) continue;
    const v = VEC[edge.direction] || [1, 0];
    assign(room, src.col + v[0], src.row + v[1], src.level + (LEVEL_DELTA[edge.direction] || 0));
    return;
  }
  // Also try the reverse: we may have walked *out* of this room.
  for (const edge of state.edges.values()) {
    if (edge.src !== id) continue;
    const dst = state.rooms.get(edge.dst);
    if (!dst || !dst.placed) continue;
    const v = VEC[edge.direction] || [1, 0];
    assign(room, dst.col - v[0], dst.row - v[1], dst.level - (LEVEL_DELTA[edge.direction] || 0));
    return;
  }
  // First room, or one reached by something that wasn't a compass move.
  assign(room, 0, 0, 0);
}

function placeAll() {
  // Several passes, because a room can only be placed once a neighbour is.
  for (let pass = 0; pass < 4; pass++) {
    for (const id of state.rooms.keys()) placeRoom(id);
  }
}

/* ------------------------------------------------------------------ *
 * Cytoscape
 * ------------------------------------------------------------------ */

const cy = cytoscape({
  container: $("map"),
  wheelSensitivity: 0.25,
  minZoom: 0.15,
  maxZoom: 3,
  style: [
    {
      selector: "node.room",
      style: {
        "background-color": "#161c29",
        "border-width": 1.5,
        "border-color": "#3a4a68",
        shape: "round-rectangle",
        width: 96,
        height: 40,
        label: "data(label)",
        color: "#b9c5da",
        "font-family": "ui-monospace, monospace",
        "font-size": 9.5,
        "text-wrap": "wrap",
        "text-max-width": 86,
        "text-valign": "center",
        "text-halign": "center",
      },
    },
    { selector: "node.room[?dark]", style: { "border-color": "#9d84e8", "background-color": "#14111f" } },
    { selector: "node.room[?deaths]", style: { "border-color": "#e05c6a", "border-width": 2 } },
    {
      selector: "node.room.current",
      style: {
        "border-color": "#e8b339",
        "border-width": 2.5,
        "background-color": "#231b0c",
        color: "#e8b339",
        "z-index": 20,
      },
    },
    {
      selector: "node.blocked",
      style: {
        "background-color": "#e05c6a",
        shape: "ellipse",
        width: 7,
        height: 7,
        label: "",
        "border-width": 0,
        opacity: 0.75,
      },
    },
    {
      selector: "edge",
      style: {
        width: 1.4,
        "line-color": "#33415c",
        "curve-style": "straight",
        "target-arrow-shape": "none",
      },
    },
    {
      selector: "edge.oneway",
      style: {
        "target-arrow-shape": "triangle",
        "target-arrow-color": "#4a5b7d",
        "arrow-scale": 0.7,
        "line-color": "#3d4d6b",
      },
    },
    {
      selector: "edge.vertical",
      style: { "line-color": "#4fd1c5", "line-style": "dashed", opacity: 0.8 },
    },
    {
      // An edge spanning more than one lattice cell would otherwise be drawn
      // straight through whatever room sits between its endpoints, reading as
      // two connections where there is one. Arc it clear instead.
      selector: "edge.long",
      style: {
        "curve-style": "unbundled-bezier",
        "control-point-distances": [42],
        "control-point-weights": [0.5],
      },
    },
    {
      selector: "edge.blocked-edge",
      style: { "line-color": "#6b2b33", "line-style": "dotted", width: 1.2 },
    },
  ],
});

$("fit").onclick = () => cy.animate({ fit: { padding: 50 }, duration: 250 });

cy.on("tap", "node.room", (evt) => {
  const d = evt.target.data();
  const blocks = [...state.blocked.values()].filter((b) => b.src === d.id);
  const exits = [...state.edges.values()].filter((e) => e.src === d.id);
  toast(
    `${d.label} — level ${d.level}, ${d.visits} visit(s)` +
      (exits.length ? ` · exits: ${exits.map((e) => e.direction).join(", ")}` : "") +
      (blocks.length ? ` · blocked: ${blocks.map((b) => b.direction).join(", ")}` : ""),
    5000
  );
});

cy.on("tap", "node.blocked", (evt) => toast(evt.target.data("message"), 5000));

function renderMap() {
  placeAll();
  const seen = new Set();

  for (const room of state.rooms.values()) {
    seen.add(room.id);
    const data = {
      id: room.id,
      label: room.name,
      dark: room.dark ? 1 : 0,
      deaths: room.deaths || 0,
      visits: room.visits || 1,
      level: room.level || 0,
    };
    const node = cy.$id(room.id);
    if (node.length) {
      node.data(data);
      node.position({ x: room.x, y: room.y });
    } else {
      cy.add({ group: "nodes", classes: "room", data, position: { x: room.x, y: room.y } });
    }
  }

  for (const [key, edge] of state.edges) {
    const id = `e:${key}`;
    if (cy.$id(id).length) {
      cy.$id(id).toggleClass("oneway", !edge.reciprocal);
      continue;
    }
    if (!seen.has(edge.src) || !seen.has(edge.dst)) continue;
    const classes = [];
    if (!edge.reciprocal) classes.push("oneway");
    if (VERTICAL.has(edge.direction)) classes.push("vertical");

    const a = state.rooms.get(edge.src);
    const b = state.rooms.get(edge.dst);
    const span = Math.max(Math.abs(a.col - b.col), Math.abs(a.row - b.row));
    if (span > 1) classes.push("long");

    cy.add({
      group: "edges",
      classes: classes.join(" "),
      data: { id, source: edge.src, target: edge.dst, direction: edge.direction },
    });
  }

  // A blocked exit is a fact about the map, so it gets drawn: a short stub
  // ending in a red tick, carrying the game's own refusal message.
  for (const [key, blk] of state.blocked) {
    const nodeId = `b:${key}`;
    if (cy.$id(nodeId).length) continue;
    const src = state.rooms.get(blk.src);
    if (!src || !src.placed) continue;
    const v = VEC[blk.direction] || [1, 0];
    cy.add({
      group: "nodes",
      classes: "blocked",
      data: { id: nodeId, message: `${blk.direction}: ${blk.message}` },
      position: { x: src.x + v[0] * CELL_W * 0.42, y: src.y + v[1] * CELL_H * 0.5 },
    });
    cy.add({
      group: "edges",
      classes: "blocked-edge",
      data: { id: `be:${key}`, source: blk.src, target: nodeId },
    });
  }

  cy.$(".room").removeClass("current");
  if (state.current) cy.$id(state.current).addClass("current");
  $("r-rooms").textContent = state.rooms.size;
}

let fitted = false;
function maybeFit() {
  if (state.rooms.size <= 1) return;
  if (!fitted || state.rooms.size < 8) {
    cy.fit(undefined, 60);
    fitted = true;
  }
}

/* ------------------------------------------------------------------ *
 * Transcript
 * ------------------------------------------------------------------ */

const transcript = $("transcript");
let entryCount = 0;

function atBottom() {
  return transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 80;
}

function append(node) {
  const stick = atBottom();
  transcript.appendChild(node);
  if (stick) transcript.scrollTop = transcript.scrollHeight;
}

function addThought(text, meta) {
  const el = document.createElement("div");
  el.className = "thought";
  el.textContent = text;
  if (meta && (meta.latency_ms || meta.cost_usd)) {
    const m = document.createElement("span");
    m.className = "meta";
    const bits = [];
    if (meta.model) bits.push(meta.model);
    if (meta.latency_ms) bits.push(`${meta.latency_ms} ms`);
    if (meta.cost_usd) bits.push(`$${meta.cost_usd.toFixed(4)}`);
    if (meta.cache_read_tokens) bits.push(`${meta.cache_read_tokens} cached`);
    m.textContent = bits.join("  ·  ");
    el.appendChild(m);
  }
  if (!state.showThoughts) el.style.display = "none";
  append(el);
}

const OUTCOME_LABEL = {
  progress: "",
  blocked: "no way through",
  unknown: "not a word it knows",
  absent: "not here",
  inert: "nothing happened",
  meta: "",
  futile: "↻ tried here before",
};

function addEntry(command, response, kind, outcome) {
  const el = document.createElement("div");
  el.className = "entry";
  if (outcome) {
    el.classList.add(
      outcome.outcome === "futile" ? "futile" : outcome.wasted ? "wasted" : "progress"
    );
  }
  if (command) {
    const c = document.createElement("div");
    c.className = "cmd";
    c.textContent = `> ${command}`;
    if (outcome && OUTCOME_LABEL[outcome.outcome]) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent =
        outcome.outcome === "futile" && outcome.repeat_count > 1
          ? `${OUTCOME_LABEL.futile} ×${outcome.repeat_count}`
          : OUTCOME_LABEL[outcome.outcome];
      c.appendChild(tag);
    }
    el.appendChild(c);
  }
  const r = document.createElement("div");
  r.className = "resp" + (kind ? ` ${kind}` : "");
  r.textContent = response;
  el.appendChild(r);
  append(el);
  $("t-count").textContent = `${++entryCount} exchanges`;
}

function note(text, isError) {
  const el = document.createElement("div");
  el.className = "event-note" + (isError ? " err" : "");
  el.textContent = text;
  append(el);
}

$("toggle-thoughts").onclick = () => {
  state.showThoughts = !state.showThoughts;
  $("toggle-thoughts").textContent = state.showThoughts ? "hide reasoning" : "show reasoning";
  document.querySelectorAll(".thought").forEach((n) => {
    n.style.display = state.showThoughts ? "" : "none";
  });
};

/* ------------------------------------------------------------------ *
 * State pane
 * ------------------------------------------------------------------ */

function renderTree(nodes, depth = 0) {
  if (!nodes || !nodes.length) return "";
  return nodes
    .map((n) => {
      const label = `${escapeHtml(n.name)}<span class="num">#${n.num}</span>`;
      if (!n.children || !n.children.length) return `<div class="leaf">${label}</div>`;
      const open = depth < 1 ? " open" : "";
      return `<details${open}><summary>${label}</summary>${renderTree(n.children, depth + 1)}</details>`;
    })
    .join("");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

/* Every object in the world that is holding something else.
 *
 * Derived from the object tree rather than named explicitly, so this works on
 * any game the engine can load — the trophy case is not special-cased, it just
 * happens to be a container with things in it. Rooms are skipped (everything
 * is "in" a room) and so is the player, whose contents are the Carrying list. */
function collectContainers(nodes, depth = 0, room = "") {
  const out = [];
  for (const node of nodes || []) {
    const here = depth === 0 ? node.name : room;
    const isRoom = depth === 0;
    const isPlayer = node.name === "you";
    if (!isRoom && !isPlayer && node.children && node.children.length) {
      out.push({
        name: node.name,
        num: node.num,
        room: here,
        contents: node.children.map((c) => c.name),
      });
    }
    if (node.children && node.children.length) {
      out.push(...collectContainers(node.children, depth + 1, here));
    }
  }
  return out;
}

function renderContainers(tree) {
  const found = collectContainers(tree);
  const el = $("s-containers");
  $("c-count").textContent = found.length ? `${found.length}` : "";
  if (!found.length) {
    el.innerHTML = '<li class="empty">none seen holding anything</li>';
    return;
  }
  el.innerHTML = found
    .map(
      (c) =>
        `<li><span class="holder">${escapeHtml(c.name)}</span> ` +
        `<span class="where">${escapeHtml(c.room)}</span><br>` +
        `<span class="arrow">└ </span><span class="held">${c.contents
          .map(escapeHtml)
          .join(", ")}</span></li>`
    )
    .join("");
}

function renderSnapshot(p) {
  $("s-room").textContent = `${p.location_name} (#${p.location_id})`;
  $("s-hash").textContent = p.state_hash ? p.state_hash.slice(0, 16) : "—";
  $("s-objects").textContent = p.object_count;

  const inv = $("s-inventory");
  $("inv-count").textContent = p.inventory.length ? `${p.inventory.length}` : "";
  inv.innerHTML = p.inventory.length
    ? p.inventory.map((i) => `<li>${escapeHtml(i)}</li>`).join("")
    : '<li class="empty">empty-handed</li>';

  renderContainers(p.tree);
  $("s-tree").innerHTML = renderTree(p.tree) || '<div class="hint">no objects reported</div>';

  $("r-score").textContent = p.max_score ? `${p.score} / ${p.max_score}` : p.score;
  $("r-gauge").style.width = p.max_score ? `${(p.score / p.max_score) * 100}%` : "0%";
}

function renderDelta(changes) {
  const el = $("s-delta");
  if (!changes || !changes.length) {
    el.innerHTML = '<div class="hint">nothing moved</div>';
    return;
  }
  el.innerHTML = changes
    .slice(0, 14)
    .map((c) => {
      const name = `<span class="name">${escapeHtml(c.name)}</span>`;
      if (c.kind === "moved") {
        return `<div class="moved">${name} <span class="arrow">${escapeHtml(
          c.before_name || c.before
        )} → ${escapeHtml(c.after_name || c.after)}</span></div>`;
      }
      return `<div class="${c.kind}">${name} <span class="arrow">${c.kind}</span></div>`;
    })
    .join("");
}

/* The ledger is rendered in full from the first frame, with undiscovered rows
 * greyed rather than absent. An empty row is data: "never established that
 * objects can be carried" is a finding, and hiding it until it happens would
 * make the pane look like a growing list of successes instead of a checklist
 * the run is being measured against. */
function renderDiscoveries(manifest) {
  const el = $("s-discoveries");
  if (!manifest || !manifest.length) {
    el.innerHTML = '<li class="empty">no session</li>';
    $("d-count").textContent = "";
    return;
  }
  const found = manifest.filter((d) => d.turn > 0).length;
  $("d-count").textContent = `${found} / ${manifest.length}`;

  el.innerHTML = manifest
    .map((d) => {
      const isNew = d.turn > 0 && !state.discoveriesSeen.has(d.key);
      if (d.turn > 0) state.discoveriesSeen.add(d.key);
      const cls = ["", d.turn > 0 ? "found" : "", isNew ? "fresh" : ""].join(" ").trim();
      const turn = d.turn > 0 ? d.turn : "·";
      const detail = d.evidence ? ` — ${escapeHtml(d.evidence)}` : "";
      return (
        `<li class="${cls}" title="${escapeHtml(d.reveals)}${detail}">` +
        `<span class="turn">${turn}</span>` +
        `<span class="what">${escapeHtml(d.label)}</span></li>`
      );
    })
    .join("");
}

const QUALITY_ORDER = [
  ["progress", "productive"],
  ["blocked", "no way through"],
  ["unknown", "word unknown"],
  ["absent", "not here"],
  ["inert", "no effect"],
  ["meta", "meta"],
  ["futile", "futile"],
];

function renderQuality(q) {
  if (!q || !q.steps) return;

  // Waste is the price of exploring. Futility is the price of not listening —
  // so it gets the loud number, and only it turns red.
  $("q-headline").innerHTML =
    `<b>${q.wasted_pct}%</b> wasted &nbsp;·&nbsp; ` +
    `<b class="${q.futile_pct > 10 ? "bad" : ""}">${q.futile_pct}%</b> futile`;

  const bar = $("q-bar");
  bar.innerHTML = QUALITY_ORDER.map(([key]) => {
    const n = q.counts[key] || 0;
    if (!n) return "";
    return `<span class="q-${key}" style="width:${(n / q.steps) * 100}%" title="${key}: ${n}"></span>`;
  }).join("");

  $("q-key").innerHTML = QUALITY_ORDER.filter(([key]) => q.counts[key])
    .map(([key, label]) => `<span><i class="q-${key}"></i>${label} ${q.counts[key]}</span>`)
    .join("");

  $("q-distinct").textContent = q.distinct_commands;
  $("q-deadends").textContent = q.known_dead_ends;
}

function renderMemory(lessons) {
  const el = $("s-memory");
  $("m-count").textContent = lessons && lessons.length ? `${lessons.length}` : "";
  if (!lessons || !lessons.length) {
    el.innerHTML = '<li class="empty">nothing kept yet</li>';
    return;
  }
  el.innerHTML = lessons
    .map((l) => {
      const where = [
        `life ${l.life}`,
        `turn ${l.turn}`,
        l.location || null,
      ].filter(Boolean).join(" · ");
      return `<li><span class="where">${escapeHtml(where)}</span>${escapeHtml(l.text)}</li>`;
    })
    .join("");
}

function renderCheckpoints(list) {
  const el = $("s-checkpoints");
  if (!list || !list.length) {
    el.innerHTML = '<span class="hint">none yet — auto-marked every 10 turns</span>';
    return;
  }
  el.innerHTML = "";
  for (const cp of [...list].sort((a, b) => a.turn - b.turn)) {
    const b = document.createElement("button");
    b.className = "chip" + (cp.auto ? "" : " manual");
    b.textContent = `${cp.label} · ${cp.score}pt`;
    b.title = `Rewind the world to turn ${cp.turn} (${cp.location}). The map keeps what it learned.`;
    b.onclick = async () => {
      const r = await post("/api/rewind", { id: cp.id });
      if (r && r.session) {
        note(`↺ rewound to ${cp.label} — world restored, map retained`);
        applySummary(r.session);
      }
    };
    el.appendChild(b);
  }
}

function renderTraces(list) {
  const el = $("s-traces");
  if (!list || !list.length) {
    el.innerHTML = '<span class="hint">none recorded</span>';
    return;
  }
  el.innerHTML = "";
  for (const t of list.slice(0, 8)) {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = t.name.replace(/\.jsonl$/, "");
    b.title = `Replay ${t.path} (${(t.size / 1024).toFixed(0)} KB)`;
    b.onclick = async () => {
      resetView();
      const r = await post("/api/replay", { path: t.path, speed: 10 });
      if (r && r.events) note(`↻ replaying ${t.name} — ${r.events} events`);
    };
    el.appendChild(b);
  }
}

/* ------------------------------------------------------------------ *
 * Event reduction
 * ------------------------------------------------------------------ */

function handle(event) {
  const p = event.payload || {};
  switch (event.type) {
    case "session.started":
      resetView();
      note(`▸ ${p.game} · ${p.agent} · max score ${p.max_score}`);
      break;

    case "agent.thought":
      addThought(p.text, p.meta);
      if (p.meta && p.meta.cost_usd) accrueCost(p.meta.cost_usd);
      break;

    case "command.issued":
      state.pendingThought = p.command;
      $("r-turn").textContent = p.turn;
      break;

    case "observation": {
      const kind = p.lost ? "death" : p.won ? "win" : "";
      addEntry(state.pendingThought, p.text, kind, p.outcome);
      state.pendingThought = null;
      if (p.quality) renderQuality(p.quality);
      break;
    }

    case "state.snapshot":
      state.current = p.room_id;
      renderSnapshot(p);
      if (state.rooms.has(p.room_id)) {
        cy.$(".room").removeClass("current");
        cy.$id(p.room_id).addClass("current");
      }
      break;

    case "object.delta":
      renderDelta(p.changes);
      break;

    case "checkpoint.created":
      state.checkpoints.set(p.id, p);
      renderCheckpoints([...state.checkpoints.values()]);
      break;

    case "discovery.made":
      state.discoveries.set(p.key, p);
      renderDiscoveries([...state.discoveries.values()]);
      note(`◆ ${p.label} — ${p.reveals}`);
      break;

    case "lesson.learned":
      state.memory.push(p);
      renderMemory(state.memory);
      note(`✎ kept: ${p.text}`);
      break;

    case "run.restored":
      $("r-deaths").textContent = p.deaths;
      note(
        p.by_agent
          ? `↺ the agent restored its own save — world back to turn ${p.to_turn}`
          : `↺ life ${p.life} — world back to turn ${p.to_turn}, ` +
            `carrying ${p.carried} note(s). ${p.lives_left} live(s) left`
      );
      break;

    case "map.update": {
      if (p.new_room) {
        state.rooms.set(p.new_room.id, { ...p.new_room, placed: false });
      }
      if (p.new_edge) {
        state.edges.set(p.new_edge.key, p.new_edge);
        // A newly reciprocal edge means its partner changed too.
        for (const e of state.edges.values()) {
          if (e.src === p.new_edge.dst && e.dst === p.new_edge.src) {
            e.reciprocal = true;
            p.new_edge.reciprocal = true;
          }
        }
      }
      if (p.new_blocked) state.blocked.set(p.new_blocked.key, p.new_blocked);
      state.current = p.current_room;
      renderMap();
      maybeFit();
      break;
    }

    case "session.ended": {
      const u = p.usage || {};
      note(
        `■ ${p.reason} — ${p.final_score}/${p.max_score} in ${p.turns} turns · ` +
          `${p.deaths || 0} death(s) · ` +
          `${p.map.rooms} rooms, ${p.map.edges} edges, ${p.map.blocked} blocked` +
          (u.calls ? ` · $${u.cost_usd} over ${u.calls} calls` : "")
      );
      if (p.censored) {
        // Not an outcome. Saying so here stops the run being read as a failure.
        note("  ⚠ stopped by the harness, not the game — this run is censored, not finished", true);
      }
      setRunning(false, true);
      refresh();
      break;
    }

    case "error":
      note(`✕ ${p.where}: ${p.message}`, true);
      break;
  }
}

let costTotal = 0;
function accrueCost(amount) {
  costTotal += amount;
  $("r-cost").textContent = `$${costTotal.toFixed(costTotal < 1 ? 4 : 2)}`;
}

function resetView() {
  state.rooms.clear();
  state.edges.clear();
  state.blocked.clear();
  state.checkpoints.clear();
  state.discoveries.clear();
  state.discoveriesSeen.clear();
  state.memory = [];
  state.occupied.clear();
  renderCheckpoints([]);
  renderDiscoveries([]);
  renderMemory([]);
  $("r-deaths").textContent = "0";
  $("q-headline").innerHTML = "<b>—</b> wasted &nbsp;·&nbsp; <b>—</b> futile";
  $("q-bar").innerHTML = "";
  $("q-key").innerHTML = "";
  $("q-distinct").textContent = "0";
  $("q-deadends").textContent = "0";
  $("s-containers").innerHTML = '<li class="empty">none seen</li>';
  $("c-count").textContent = "";
  $("inv-count").textContent = "";
  state.current = null;
  costTotal = 0;
  entryCount = 0;
  fitted = false;
  cy.elements().remove();
  transcript.innerHTML = "";
  $("r-cost").textContent = "$0.00";
  $("r-turn").textContent = "0";
  $("r-rooms").textContent = "0";
  $("s-delta").innerHTML = '<div class="hint">nothing yet</div>';
}

/* ------------------------------------------------------------------ *
 * Transport
 * ------------------------------------------------------------------ */

let socket = null;

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${location.host}/ws`);

  socket.onopen = () => {
    $("r-link").textContent = "live";
    $("r-link").className = "value cyan";
  };

  socket.onmessage = (msg) => {
    const data = JSON.parse(msg.data);
    if (data.type === "hello") {
      // Catch up on everything that happened before this tab opened.
      for (const event of data.payload.backlog) handle(event);
      return;
    }
    handle(data);
  };

  socket.onclose = () => {
    $("r-link").textContent = "offline";
    $("r-link").className = "value";
    setTimeout(connect, 1500);
  };
}

async function post(path, body) {
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json();
    if (!res.ok) {
      toast(data.error || `${res.status} ${res.statusText}`, 7000);
      return null;
    }
    return data;
  } catch (err) {
    toast(String(err), 7000);
    return null;
  }
}

async function refresh() {
  const res = await fetch("/api/state");
  const data = await res.json();
  renderTraces(data.traces);
  if (data.session) applySummary(data.session);
}

function applySummary(s) {
  state.session = s;
  $("r-turn").textContent = s.turn;
  for (const cp of s.checkpoints || []) state.checkpoints.set(cp.id, cp);
  renderCheckpoints([...state.checkpoints.values()]);

  // The summary carries the full ledger including rows not yet discovered.
  // Events and this snapshot can arrive in either order, so a row that is
  // already found here never loses to an empty one from the manifest.
  for (const d of s.discoveries || []) {
    const existing = state.discoveries.get(d.key);
    if (!existing || !existing.turn) state.discoveries.set(d.key, d);
  }
  renderDiscoveries([...state.discoveries.values()]);

  if ((s.memory || []).length >= state.memory.length) state.memory = s.memory || [];
  renderMemory(state.memory);
  renderQuality(s.quality);
  $("r-deaths").textContent = s.deaths || 0;
  if (s.usage && s.usage.cost_usd) {
    costTotal = s.usage.cost_usd;
    $("r-cost").textContent = `$${costTotal.toFixed(costTotal < 1 ? 4 : 2)}`;
  }
  setRunning(!s.paused && !s.finished, s.finished);
}

/* ------------------------------------------------------------------ *
 * Controls
 * ------------------------------------------------------------------ */

function setRunning(running, finished) {
  $("run").disabled = running || finished;
  $("pause").disabled = !running;
  $("step").disabled = running || finished;
  $("mark").disabled = !state.session || finished;
}

function syncAgentControls() {
  const agent = $("agent").value;
  const llm = agent === "claude";
  $("model").style.display = llm ? "" : "none";
  $("effort").style.display = llm ? "" : "none";
  $("info").style.display = llm ? "" : "none";
  $("input-row").classList.toggle("on", agent === "human");
}

function syncEngineControls() {
  $("rom").style.display = $("engine").value === "jericho" ? "" : "none";
}

$("agent").onchange = syncAgentControls;
$("engine").onchange = syncEngineControls;

$("new-run").onclick = async () => {
  resetView();
  const body = {
    engine: $("engine").value,
    rom: $("rom").value || null,
    agent: $("agent").value,
    model: $("model").value,
    effort: $("effort").value,
    info_level: $("info").value,
    max_turns: parseInt($("turns").value, 10) || 200,
    lives: parseInt($("lives").value, 10) || 0,
    delay: 0.35,
  };
  const data = await post("/api/session", body);
  if (data && data.session) {
    applySummary(data.session);
    if (data.trace) note(`● recording to ${data.trace}`);
  }
};

$("run").onclick = async () => {
  setRunning(true, false);
  const r = await post("/api/control", { action: "run" });
  if (r && r.session) applySummary(r.session);
};

$("pause").onclick = async () => {
  const r = await post("/api/control", { action: "pause" });
  if (r && r.session) applySummary(r.session);
};

$("step").onclick = async () => {
  const r = await post("/api/control", { action: "step" });
  if (r && r.session) applySummary(r.session);
};

$("mark").onclick = async () => {
  const r = await post("/api/checkpoint", { label: "" });
  if (r && r.session) {
    note(`⚑ checkpoint ${r.label} — the world can be restored here`);
    applySummary(r.session);
  }
};

async function sendCommand() {
  const input = $("human-cmd");
  const value = input.value.trim();
  if (!value) return;
  input.value = "";
  await post("/api/command", { command: value });
}

$("send").onclick = sendCommand;
$("human-cmd").addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendCommand();
});

let toastTimer = null;
function toast(message, ms = 4000) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("on"), ms);
}

syncAgentControls();
syncEngineControls();
connect();
refresh();
