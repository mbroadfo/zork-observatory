/* Chart view — the scanned map, uncovered as the run explores it.
 *
 * The graph view draws the world from what the agent has seen. This one draws
 * it from a period map, but only the parts the run has been to: the rest sits
 * under fog, and each room visited burns a soft-edged hole through it. The map
 * is the observer's, never the agent's — nothing here reaches a prompt.
 *
 * Coordinates are "atlas units": pixels of the original scan. The world layer
 * is sized in those units and moved with one CSS transform, so the image, the
 * fog and the vector marks can never drift apart.
 *
 * Room identity comes from the event stream (`r<object number>`), and the
 * atlas is keyed by object number for one exact story build. State is recorded
 * even before the atlas loads, so a backlog replayed on connect still reveals.
 */

const Chart = (() => {
  const $ = (id) => document.getElementById(id);
  const stage = $("atlas");
  const world = $("atlas-world");
  const img = $("atlas-img");
  const fog = $("atlas-fog");
  const fogCtx = fog.getContext("2d");
  const svg = $("atlas-marks");
  const mini = $("atlas-mini");
  const miniCtx = mini.getContext("2d");
  const plate = $("atlas-plate");
  const tip = $("atlas-tip");
  const NS = "http://www.w3.org/2000/svg";

  const FOG_RES = 1 / 5;          // fog pixels per atlas unit; upscaling softens it
  const MAX_K = 1.6;              // deepest zoom, in screen px per scan px
  const CORRIDOR_MAX = 720;       // longer hops are page jumps on the scan, not passages
  const TRAIL_LEN = 40;
  const REVEAL_MS = 1100;

  let A = null;                   // atlas data
  let W = 1, H = 1;
  let loadToken = 0;
  const view = { x: 0, y: 0, k: 0.1 };
  let follow = true;
  let placed = false;             // has the view been positioned for this atlas

  const revealed = new Map();     // num -> t0
  const corridors = new Map();    // "a|b" -> {a, b, t0}
  const deaths = new Map();       // num -> count
  const trail = [];               // nums, oldest first
  let current = null;             // num
  let currentName = "";
  let fogBusy = false;
  let fogDirty = true;
  let miniThumb = null;

  const num = (id) => (typeof id === "string" && id[0] === "r" ? id.slice(1) : String(id));
  const room = (n) => (A && A.rooms[n]) || null;
  const centre = (r) => [(r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2];
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const ease = (t) => 1 - Math.pow(1 - t, 3);

  /* ---------------- loading ---------------- */

  async function load(story) {
    const token = ++loadToken;
    A = null;
    stage.classList.remove("ready");
    if (!story) return false;
    let data = null;
    try {
      const res = await fetch(`/api/atlas?story=${encodeURIComponent(story)}`);
      data = (await res.json()).atlas;
    } catch (_) {
      return false;
    }
    if (!data || token !== loadToken) return false;

    await new Promise((resolve) => {
      img.onload = resolve;
      img.onerror = resolve;
      img.src = data.image_url;
    });
    if (token !== loadToken || !img.naturalWidth) return false;

    A = data;
    W = A.width;
    H = A.height;
    world.style.width = `${W}px`;
    world.style.height = `${H}px`;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    fog.width = Math.round(W * FOG_RES);
    fog.height = Math.round(H * FOG_RES);
    $("atlas-credit").textContent = A.credit || "";
    buildThumb();
    placed = false;
    fogDirty = true;
    stage.classList.add("ready");
    return true;
  }

  /* ---------------- fog ---------------- */

  // Tileable value noise, so the fog has body instead of being a flat fill.
  const mist = (() => {
    const size = 256;
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const ctx = c.getContext("2d");
    const out = ctx.createImageData(size, size);
    const lattice = (period, seed) => {
      const g = new Float32Array(period * period);
      let s = seed;
      for (let i = 0; i < g.length; i++) {
        s = (s * 16807) % 2147483647;
        g[i] = s / 2147483647;
      }
      return (x, y) => g[((y % period) + period) % period * period + (((x % period) + period) % period)];
    };
    const octaves = [[4, 0.5, 11], [8, 0.27, 23], [16, 0.15, 37], [32, 0.08, 51]];
    const fns = octaves.map(([p, , seed]) => lattice(p, seed));
    const smooth = (t) => t * t * (3 - 2 * t);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let v = 0;
        octaves.forEach(([p, amp], i) => {
          const fx = (x / size) * p, fy = (y / size) * p;
          const x0 = Math.floor(fx), y0 = Math.floor(fy);
          const tx = smooth(fx - x0), ty = smooth(fy - y0);
          const f = fns[i];
          const a = f(x0, y0) + (f(x0 + 1, y0) - f(x0, y0)) * tx;
          const b = f(x0, y0 + 1) + (f(x0 + 1, y0 + 1) - f(x0, y0 + 1)) * tx;
          v += (a + (b - a) * ty) * amp;
        });
        const o = (y * size + x) * 4;
        const lum = Math.pow(clamp(v, 0, 1), 1.6);
        out.data[o] = 70 + lum * 60;
        out.data[o + 1] = 78 + lum * 64;
        out.data[o + 2] = 104 + lum * 70;
        out.data[o + 3] = Math.round(lum * 120);
      }
    }
    ctx.putImageData(out, 0, 0);
    return c;
  })();

  // The same mist, drifting slowly over the whole stage.
  $("atlas-drift").style.backgroundImage = `url(${mist.toDataURL()})`;
  const mistPattern = fogCtx.createPattern(mist, "repeat");

  /* The line the cartographers drew between two rooms, oriented from `a`.
   * Traced passages come from the atlas (tools/trace_paths.py); rooms the map
   * joins only by a "(to …)" stub fall back to a short straight reveal, or to
   * nothing if they sit far apart on the scan. */
  function passage(a, b) {
    const ra = room(a), rb = room(b);
    if (!ra || !rb) return null;
    const ca = centre(ra), cb = centre(rb);
    const traced = A.paths && A.paths[[a, b].sort().join("|")];
    if (traced && traced.length > 1) {
      const head = traced[0], tail = traced[traced.length - 1];
      const flip = Math.hypot(tail[0] - ca[0], tail[1] - ca[1]) < Math.hypot(head[0] - ca[0], head[1] - ca[1]);
      return { drawn: true, pts: flip ? [...traced].reverse() : traced };
    }
    if (Math.hypot(cb[0] - ca[0], cb[1] - ca[1]) > CORRIDOR_MAX) return null;
    return { drawn: false, pts: [ca, cb] };
  }

  /** The first `p` (0..1) of a polyline, by length. */
  function partial(pts, p) {
    if (p >= 1) return pts;
    const seg = [];
    let total = 0;
    for (let i = 1; i < pts.length; i++) {
      const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
      seg.push(d);
      total += d;
    }
    let left = total * p;
    const out = [pts[0]];
    for (let i = 1; i < pts.length; i++) {
      if (left >= seg[i - 1]) {
        out.push(pts[i]);
        left -= seg[i - 1];
        continue;
      }
      const t = seg[i - 1] ? left / seg[i - 1] : 0;
      out.push([pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * t, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * t]);
      break;
    }
    return out;
  }

  function rng(seed) {
    let s = (parseInt(seed, 10) * 9301 + 49297) % 233280 || 1;
    return () => (s = (s * 9301 + 49297) % 233280) / 233280;
  }

  function hole(ctx, x, y, r, a) {
    if (r <= 0) return;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(0,0,0,${a})`);
    g.addColorStop(0.45, `rgba(0,0,0,${a})`);
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawFog(now) {
    const c = fogCtx;
    const s = FOG_RES;
    let busy = false;
    const progress = (t0) => {
      const t = (now - t0) / REVEAL_MS;
      if (t < 1) busy = true;
      return ease(clamp(t, 0, 1));
    };

    c.globalCompositeOperation = "source-over";
    c.fillStyle = "#06080d";
    c.fillRect(0, 0, fog.width, fog.height);
    c.save();
    c.scale(3, 3);                     // mist features larger than the tile
    c.fillStyle = mistPattern;
    c.fillRect(0, 0, fog.width / 3, fog.height / 3);
    c.restore();

    c.globalCompositeOperation = "destination-out";
    c.lineCap = "round";

    for (const cor of corridors.values()) {
      if (!revealed.has(cor.a) || !revealed.has(cor.b)) continue;
      const route = passage(cor.a, cor.b);
      if (!route) continue;
      // Uncovered from the end the run walked out of, like a lamp going ahead.
      const pts = partial(route.pts, progress(cor.t0));
      const bands = route.drawn
        ? [[110, 0.3], [70, 0.5], [34, 0.9]]
        : [[120, 0.35], [80, 0.45], [44, 0.8]];
      for (const [w, a] of bands) {
        c.strokeStyle = `rgba(0,0,0,${a})`;
        c.lineWidth = w * s;
        c.lineJoin = "round";
        c.beginPath();
        pts.forEach(([x, y], i) => (i ? c.lineTo(x * s, y * s) : c.moveTo(x * s, y * s)));
        c.stroke();
      }
    }

    for (const [n, t0] of revealed) {
      const r = room(n);
      if (!r) continue;
      const p = progress(t0);
      const [x0, y0, x1, y1] = r.box;
      const rw = (x1 - x0) / 2, rh = (y1 - y0) / 2;
      const [cx, cy] = centre(r);
      const short = Math.min(rw, rh);
      const r0 = (short * 1.9 + 40) * p;

      // Tall or wide rooms get a row of holes along their long axis.
      const span = Math.max(rw, rh) - short;
      const steps = Math.max(1, Math.ceil(span / (short * 0.6)));
      for (let i = 0; i <= steps; i++) {
        const t = steps === 0 ? 0 : -span + (2 * span * i) / steps;
        const hx = rw >= rh ? cx + t : cx;
        const hy = rw >= rh ? cy : cy + t;
        hole(c, hx * s, hy * s, r0 * s, 1);
      }

      // Ragged edge: satellites at stable pseudo-random spots.
      const rand = rng(n);
      for (let i = 0; i < 7; i++) {
        const ang = (i / 7) * Math.PI * 2 + rand() * 0.8;
        const reach = 0.75 + rand() * 0.35;
        const sx = cx + Math.cos(ang) * (rw + short * 0.5) * reach * p;
        const sy = cy + Math.sin(ang) * (rh + short * 0.5) * reach * p;
        hole(c, sx * s, sy * s, r0 * (0.35 + rand() * 0.25) * s, 0.6);
      }
    }

    c.globalCompositeOperation = "source-over";
    return busy;
  }

  /* ---------------- vector marks ---------------- */

  const layer = (id) => {
    const g = document.createElementNS(NS, "g");
    g.id = id;
    svg.appendChild(g);
    return g;
  };
  const gVisited = layer("mk-visited");
  const gTrail = layer("mk-trail");
  const gDeaths = layer("mk-deaths");
  const gYou = layer("mk-you");
  gYou.innerHTML = `
    <defs>
      <radialGradient id="you-halo">
        <stop offset="0" stop-color="#ffd36b" stop-opacity=".55"/>
        <stop offset=".5" stop-color="#e8b339" stop-opacity=".18"/>
        <stop offset="1" stop-color="#e8b339" stop-opacity="0"/>
      </radialGradient>
    </defs>
    <circle class="you-halo" r="1" fill="url(#you-halo)"/>
    <circle class="you-ring" r="1"/>
    <circle class="you-ring late" r="1"/>
    <rect class="you-box"/>
    <circle class="you-dot" r="1"/>`;
  const el = (sel) => gYou.querySelector(sel);

  function el2(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function drawMarks() {
    gVisited.replaceChildren();
    gTrail.replaceChildren();
    gDeaths.replaceChildren();
    if (!A) return;

    for (const n of revealed.keys()) {
      const r = room(n);
      if (!r || n === current) continue;
      const [x0, y0, x1, y1] = r.box;
      gVisited.appendChild(el2("rect", {
        x: x0 - 6, y: y0 - 6, width: x1 - x0 + 12, height: y1 - y0 + 12, rx: 8,
        class: "visited",
      }));
    }

    // The recent path, along the drawn passages. Only steps down a walked
    // passage count, so a restore's jump leaves a gap instead of a line.
    let d = "";
    for (let i = 1; i < trail.length; i++) {
      const a = trail[i - 1], b = trail[i];
      if (a === b || !corridors.has([a, b].sort().join("|"))) continue;
      const route = passage(a, b);
      if (!route) continue;
      const [ca, cb] = [centre(room(a)), centre(room(b))];
      const pts = route.drawn ? [ca, ...route.pts, cb] : route.pts;
      d += pts.map(([x, y], j) => `${j ? "L" : "M"}${x},${y}`).join("");
    }
    if (d) gTrail.appendChild(el2("path", { d, class: "trail" }));

    for (const [n, count] of deaths) {
      const r = room(n);
      if (!r) continue;
      const x = r.box[2] - 4, y = r.box[1] + 4, k = 26;
      const g = el2("g", { class: "death", transform: `translate(${x},${y})` });
      g.appendChild(el2("circle", { r: 40 }));
      g.appendChild(el2("path", { d: `M${-k},${-k}L${k},${k}M${k},${-k}L${-k},${k}` }));
      if (count > 1) {
        const t = el2("text", { x: 48, y: 14 });
        t.textContent = `×${count}`;
        g.appendChild(t);
      }
      gDeaths.appendChild(g);
    }
    placeYou();
  }

  let youAt = null;
  function placeYou() {
    const r = current && room(current);
    gYou.style.display = r ? "" : "none";
    if (!r) return;
    const [cx, cy] = centre(r);
    const [x0, y0, x1, y1] = r.box;
    const rw = (x1 - x0) / 2, rh = (y1 - y0) / 2;
    // Never smaller than a legible dot, whatever the zoom.
    const ring = Math.max(Math.hypot(rw, rh) + 30, 22 / view.k);
    if (!youAt || youAt[0] !== cx || youAt[1] !== cy) {
      // A page jump gets a longer glide, in step with the camera's flight.
      const far = youAt ? Math.hypot(cx - youAt[0], cy - youAt[1]) : -1;
      gYou.style.transitionDuration = `${far < 0 ? 0 : far > 1500 ? 1.4 : far > 600 ? 0.9 : 0.55}s`;
      gYou.style.transform = `translate(${cx}px, ${cy}px)`;
      youAt = [cx, cy];
    }
    el(".you-halo").setAttribute("r", ring * 1.9);
    el(".you-ring").setAttribute("r", ring);
    el(".you-ring.late").setAttribute("r", ring);
    el(".you-dot").setAttribute("r", Math.max(14, 5 / view.k));
    const box = el(".you-box");
    box.setAttribute("x", -rw - 8);
    box.setAttribute("y", -rh - 8);
    box.setAttribute("width", 2 * rw + 16);
    box.setAttribute("height", 2 * rh + 16);
    box.setAttribute("rx", 10);
  }

  /* ---------------- view ---------------- */

  const size = () => [stage.clientWidth || 1, stage.clientHeight || 1];
  const fitK = () => {
    const [sw, sh] = size();
    return Math.min(sw / W, sh / H);
  };

  function clampView() {
    const [sw, sh] = size();
    view.k = clamp(view.k, fitK() * 0.6, MAX_K);
    view.x = clamp(view.x, sw / 2 - W * view.k, sw / 2);
    view.y = clamp(view.y, sh / 2 - H * view.k, sh / 2);
  }

  let settleTimer = null;
  function apply() {
    clampView();
    world.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.k})`;
    // Promote to a layer only while moving; at rest the browser re-rasterises
    // at the true scale, so zoomed-in text stays sharp.
    world.classList.add("moving");
    clearTimeout(settleTimer);
    settleTimer = setTimeout(() => world.classList.remove("moving"), 160);
    stage.style.setProperty("--k", view.k);
    placeYou();
    requestFrame();
  }

  function zoomAt(sx, sy, factor) {
    chaseTarget = null;    // the viewer is driving now
    const wx = (sx - view.x) / view.k, wy = (sy - view.y) / view.k;
    const k = clamp(view.k * factor, fitK() * 0.6, MAX_K);
    view.x = sx - wx * k;
    view.y = sy - wy * k;
    view.k = k;
    apply();
  }

  /* Following the run.
   *
   * A fixed-length tween per move can't keep up: the run moves every few
   * hundred milliseconds, each new tween restarts from rest, and a jump across
   * the scan's page break never finishes before the next move. Instead the
   * camera chases the latest room continuously, so a new move just moves the
   * target. While the target is far away, the camera zooms out to keep both
   * ends in view, then settles back in as it closes — a page jump reads as a
   * flight rather than a smear. */
  let chaseTarget = null;       // {x, y, k}
  let chaseLast = 0;
  const CHASE_TAU = 260;        // ms to close ~63% of the gap
  const ZOOM_TAU = 320;

  function chase(wx, wy) {
    flight = null;
    inertia = null;
    const k = chaseTarget ? chaseTarget.k : view.k;
    chaseTarget = { x: wx, y: wy, k };
    if (!chaseLast) {
      chaseLast = performance.now();
      requestAnimationFrame(chaseStep);
    }
  }

  function chaseStep(now) {
    if (!chaseTarget) {
      chaseLast = 0;
      return;
    }
    const dt = Math.min(64, now - chaseLast);
    chaseLast = now;
    const [sw, sh] = size();
    const cx = (sw / 2 - view.x) / view.k, cy = (sh / 2 - view.y) / view.k;
    const dx = chaseTarget.x - cx, dy = chaseTarget.y - cy;
    const dist = Math.hypot(dx, dy);

    // Far away: pull back until the whole remaining trip fits on screen.
    const roomy = (Math.min(sw, sh) * 0.7) / Math.max(dist, 1);
    const wantK = Math.min(chaseTarget.k, Math.max(roomy, fitKForChase()));
    const kk = view.k * Math.pow(wantK / view.k, 1 - Math.exp(-dt / ZOOM_TAU));
    const step = 1 - Math.exp(-dt / CHASE_TAU);
    const nx = cx + dx * step, ny = cy + dy * step;

    view.k = kk;
    view.x = sw / 2 - nx * kk;
    view.y = sh / 2 - ny * kk;
    apply();

    if (dist < 0.5 && Math.abs(kk / chaseTarget.k - 1) < 0.002) {
      chaseTarget = null;
      chaseLast = 0;
      return;
    }
    requestAnimationFrame(chaseStep);
  }

  const fitKForChase = () => fitK() * 0.9;

  let flight = null;
  function flyTo(wx, wy, k, ms = 700) {
    chaseTarget = null;
    const [sw, sh] = size();
    const k0 = view.k;
    const c0 = [(sw / 2 - view.x) / k0, (sh / 2 - view.y) / k0];
    const k1 = clamp(k, fitK() * 0.6, MAX_K);
    const t0 = performance.now();
    const token = {};
    flight = token;
    const step = (now) => {
      if (flight !== token) return;
      const t = clamp((now - t0) / ms, 0, 1);
      const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      const kk = k0 * Math.pow(k1 / k0, e);
      const cx = c0[0] + (wx - c0[0]) * e, cy = c0[1] + (wy - c0[1]) * e;
      view.k = kk;
      view.x = sw / 2 - cx * kk;
      view.y = sh / 2 - cy * kk;
      apply();
      if (t < 1) requestAnimationFrame(step);
      else flight = null;
    };
    requestAnimationFrame(step);
  }

  function jumpTo(wx, wy, k) {
    chaseTarget = null;
    const [sw, sh] = size();
    view.k = clamp(k, fitK() * 0.6, MAX_K);
    view.x = sw / 2 - wx * view.k;
    view.y = sh / 2 - wy * view.k;
    apply();
  }

  const homeK = () => clamp(size()[0] / 1500, fitK(), 0.9);

  function locate(animate = true) {
    const r = current && room(current);
    if (!r) return fit();
    const [cx, cy] = centre(r);
    const k = Math.max(view.k, homeK() * 0.7);
    animate ? flyTo(cx, cy, k) : jumpTo(cx, cy, k);
  }

  function fit() {
    flyTo(W / 2, H / 2, fitK() * 0.96);
  }

  function setFollow(on) {
    follow = on;
    $("atlas-follow").classList.toggle("on", on);
    if (on) locate();
  }

  function ensurePlaced() {
    if (placed || !A || !stage.clientWidth) return;
    placed = true;
    const r = current && room(current);
    if (r) {
      const [cx, cy] = centre(r);
      jumpTo(cx, cy, homeK());
    } else {
      jumpTo(W / 2, H / 2, fitK());
    }
  }

  /* ---------------- frame loop ---------------- */

  let framePending = false;
  function requestFrame() {
    if (framePending) return;
    framePending = true;
    requestAnimationFrame((now) => {
      framePending = false;
      if (!A) return;
      ensurePlaced();
      if (fogDirty || fogBusy) {
        fogBusy = drawFog(now);
        fogDirty = false;
      }
      drawMini();
      if (fogBusy) requestFrame();
    });
  }

  /* ---------------- minimap ---------------- */

  function buildThumb() {
    const dpr = window.devicePixelRatio || 1;
    const w = 210;
    const h = Math.round((w * H) / W);
    mini.style.width = `${w}px`;
    mini.style.height = `${h}px`;
    mini.width = Math.round(w * dpr);
    mini.height = Math.round(h * dpr);
    miniThumb = document.createElement("canvas");
    miniThumb.width = mini.width;
    miniThumb.height = mini.height;
    const t = miniThumb.getContext("2d");
    t.filter = "sepia(.4) brightness(.8)";
    t.drawImage(img, 0, 0, mini.width, mini.height);
  }

  function drawMini() {
    if (!miniThumb) return;
    const c = miniCtx;
    const mw = mini.width, mh = mini.height;
    c.clearRect(0, 0, mw, mh);
    c.drawImage(miniThumb, 0, 0);
    c.drawImage(fog, 0, 0, mw, mh);
    const sx = mw / W, sy = mh / H;
    const [sw, sh] = size();
    const vx = (-view.x / view.k) * sx, vy = (-view.y / view.k) * sy;
    const vw = (sw / view.k) * sx, vh = (sh / view.k) * sy;
    c.strokeStyle = "rgba(232,179,57,.9)";
    c.lineWidth = Math.max(1, mw / 210);
    c.strokeRect(vx, vy, vw, vh);
    const r = current && room(current);
    if (r) {
      const [cx, cy] = centre(r);
      c.fillStyle = "#ffd36b";
      c.shadowColor = "#e8b339";
      c.shadowBlur = 8;
      c.beginPath();
      c.arc(cx * sx, cy * sy, Math.max(2.5, mw / 80), 0, Math.PI * 2);
      c.fill();
      c.shadowBlur = 0;
    }
  }

  function miniJump(e) {
    const rect = mini.getBoundingClientRect();
    const wx = ((e.clientX - rect.left) / rect.width) * W;
    const wy = ((e.clientY - rect.top) / rect.height) * H;
    setFollow(false);
    flight = null;
    jumpTo(wx, wy, view.k);
  }

  mini.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    mini.setPointerCapture(e.pointerId);
    miniJump(e);
  });
  mini.addEventListener("pointermove", (e) => {
    if (e.buttons) miniJump(e);
  });

  /* ---------------- interaction ---------------- */

  const pointers = new Map();
  let drag = null;
  let inertia = null;

  function local(e) {
    const r = stage.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  function hit(sx, sy) {
    if (!A) return null;
    const wx = (sx - view.x) / view.k, wy = (sy - view.y) / view.k;
    const pad = 10 / view.k;
    for (const n of revealed.keys()) {
      const r = room(n);
      if (!r) continue;
      const [x0, y0, x1, y1] = r.box;
      if (wx >= x0 - pad && wx <= x1 + pad && wy >= y0 - pad && wy <= y1 + pad) return n;
    }
    return null;
  }

  stage.addEventListener("wheel", (e) => {
    if (!A) return;
    e.preventDefault();
    flight = null;
    inertia = null;
    const [sx, sy] = local(e);
    const scale = e.deltaMode === 1 ? 40 : 1;
    zoomAt(sx, sy, Math.exp(-e.deltaY * scale * (e.ctrlKey ? 0.01 : 0.0015)));
  }, { passive: false });

  stage.addEventListener("pointerdown", (e) => {
    if (!A || e.target.closest(".atlas-ui")) return;
    stage.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, local(e));
    flight = null;
    inertia = null;
    chaseTarget = null;
    drag = { moved: 0, last: local(e), t: performance.now(), vx: 0, vy: 0 };
    stage.classList.add("grabbing");
  });

  stage.addEventListener("pointermove", (e) => {
    if (!A) return;
    const p = local(e);
    if (!pointers.has(e.pointerId)) {
      hover(p, e);
      return;
    }
    const prev = pointers.get(e.pointerId);
    pointers.set(e.pointerId, p);

    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const other = [...pointers.entries()].find(([id]) => id !== e.pointerId)[1];
      const before = Math.hypot(prev[0] - other[0], prev[1] - other[1]);
      const after = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (before > 0) zoomAt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, after / before);
      drag.moved += 10;
      return;
    }

    const dx = p[0] - prev[0], dy = p[1] - prev[1];
    drag.moved += Math.abs(dx) + Math.abs(dy);
    if (drag.moved > 4 && follow) setFollow(false);
    view.x += dx;
    view.y += dy;
    const now = performance.now();
    const dt = Math.max(1, now - drag.t);
    drag.vx = 0.8 * (dx / dt) + 0.2 * drag.vx;
    drag.vy = 0.8 * (dy / dt) + 0.2 * drag.vy;
    drag.t = now;
    tip.hidden = true;
    apply();
  });

  function release(e) {
    if (!pointers.has(e.pointerId)) return;
    pointers.delete(e.pointerId);
    if (pointers.size) return;
    stage.classList.remove("grabbing");
    if (drag && drag.moved <= 4) {
      const n = hit(...local(e));
      if (n) flyTo(...centre(room(n)), Math.max(view.k, homeK()), 500);
    } else if (drag && performance.now() - drag.t < 80) {
      coast(drag.vx, drag.vy);
    }
    drag = null;
  }
  stage.addEventListener("pointerup", release);
  stage.addEventListener("pointercancel", release);
  stage.addEventListener("pointerleave", () => (tip.hidden = true));

  function coast(vx, vy) {
    const token = {};
    inertia = token;
    let last = performance.now();
    const step = (now) => {
      if (inertia !== token) return;
      const dt = now - last;
      last = now;
      view.x += vx * dt;
      view.y += vy * dt;
      const decay = Math.pow(0.994, dt);
      vx *= decay;
      vy *= decay;
      apply();
      if (Math.hypot(vx, vy) > 0.02) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  stage.addEventListener("dblclick", (e) => {
    if (!A || e.target.closest(".atlas-ui")) return;
    const [sx, sy] = local(e);
    flyTo((sx - view.x) / view.k, (sy - view.y) / view.k, view.k * 2, 450);
  });

  function hover(p, e) {
    const n = hit(...p);
    stage.classList.toggle("over-room", !!n);
    if (!n) {
      tip.hidden = true;
      return;
    }
    const info = api.describe ? api.describe(`r${n}`) : null;
    const r = room(n);
    // The chart's names, not the story file's: Jericho reads the object table,
    // where Zork abbreviates ("CanyView", "Living ").
    const lines = [`<b>${esc(r.name)}</b>`];
    if (info) {
      lines.push(`${info.visits} visit${info.visits === 1 ? "" : "s"}` +
        (deaths.get(n) ? ` · <span class="bad">${deaths.get(n)} death${deaths.get(n) > 1 ? "s" : ""}</span>` : ""));
      if (info.exits.length) lines.push(`exits taken: ${esc(info.exits.join(", "))}`);
      if (info.blocked.length) lines.push(`<span class="dim">walls: ${esc(info.blocked.join(", "))}</span>`);
    }
    tip.innerHTML = lines.join("<br>");
    tip.hidden = false;
    const [sw] = size();
    tip.style.left = `${Math.min(p[0] + 16, sw - tip.offsetWidth - 8)}px`;
    tip.style.top = `${p[1] + 16}px`;
  }

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  $("atlas-in").onclick = () => zoomAt(size()[0] / 2, size()[1] / 2, 1.5);
  $("atlas-out").onclick = () => zoomAt(size()[0] / 2, size()[1] / 2, 1 / 1.5);
  $("atlas-fit").onclick = () => {
    setFollow(false);
    fit();
  };
  $("atlas-locate").onclick = () => setFollow(true);
  $("atlas-follow").onclick = () => setFollow(!follow);

  window.addEventListener("keydown", (e) => {
    if (!A || stage.offsetParent === null) return;
    if (e.target.closest("input, select, textarea") || e.ctrlKey || e.metaKey || e.altKey) return;
    const [sw, sh] = size();
    if (e.key === "+" || e.key === "=") zoomAt(sw / 2, sh / 2, 1.4);
    else if (e.key === "-" || e.key === "_") zoomAt(sw / 2, sh / 2, 1 / 1.4);
    else if (e.key === "0") $("atlas-fit").onclick();
    else if (e.key === "." || e.key === "c") setFollow(true);
    else return;
    e.preventDefault();
  });

  const visible = () => stage.clientWidth > 0;

  new ResizeObserver(() => {
    if (!A || !visible()) return;
    ensurePlaced();
    apply();
  }).observe(stage);

  /* ---------------- plate ---------------- */

  function renderPlate() {
    const r = current && room(current);
    const charted = [...revealed.keys()].filter((n) => room(n)).length;
    const total = A ? Object.keys(A.rooms).length : 0;
    plate.querySelector(".where").textContent = (r && r.name) || currentName || "—";
    plate.querySelector(".sub").textContent = !A
      ? ""
      : r
        ? `${charted} of ${total} charted`
        : `off the chart · ${charted} of ${total} charted`;
  }

  /* ---------------- public ---------------- */

  const api = {
    describe: null,
    load,
    get active() {
      return !!A;
    },

    reset() {
      revealed.clear();
      corridors.clear();
      deaths.clear();
      trail.length = 0;
      current = null;
      currentName = "";
      placed = false;
      youAt = null;
      chaseTarget = null;
      follow = true;
      $("atlas-follow").classList.add("on");
      fogDirty = true;
      drawMarks();
      renderPlate();
      requestFrame();
    },

    /** The run is now in this room. */
    arrive(id, name) {
      const n = num(id);
      currentName = name || currentName;
      if (!revealed.has(n)) {
        revealed.set(n, performance.now());
        fogDirty = true;
      }
      if (n !== current) {
        current = n;
        trail.push(n);
        if (trail.length > TRAIL_LEN) trail.shift();
        drawMarks();
        if (A && placed && follow && room(n) && visible()) chase(...centre(room(n)));
      }
      renderPlate();
      requestFrame();
    },

    /** A passage was walked. Taken from map edges rather than consecutive
     * rooms, so a rollback or a restore never draws a corridor that isn't. */
    link(src, dst) {
      const a = num(src), b = num(dst);
      const key = [a, b].sort().join("|");
      if (a === b || corridors.has(key)) return;
      corridors.set(key, { a, b, t0: performance.now() });
      fogDirty = true;
      drawMarks();     // the trail step that just became a known passage
      requestFrame();
    },

    death(id) {
      const n = num(id || `r${current}`);
      deaths.set(n, (deaths.get(n) || 0) + 1);
      drawMarks();
    },

    shown() {
      if (!A || !visible()) return;
      ensurePlaced();
      if (follow) locate(false);
      apply();
      drawMarks();
      renderPlate();
    },
  };
  return api;
})();
