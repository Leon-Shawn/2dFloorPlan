/* =============================================================================
 * 2D 户型全覆盖清扫仿真 — 前端
 *
 * 职责：
 *   1. 户型选择（缩略图来自服务端栅格化结果）
 *   2. 解码服务端下发的紧凑回放数据并重放机器人清扫路径
 *   3. Canvas 逐格渲染：墙体 / 待清扫 / 已清扫 / 不可达 / 死角 / 热度
 *   4. 覆盖率看板与效率指标
 *
 * 回放数据之所以只有几 KB：服务端只下发"每步一个方向码 + 起点"，
 * 浏览器按同样的整数方形足迹规则重新叠印清扫范围，因此动画与看板数字始终一致。
 * ========================================================================== */
'use strict';

/* 方向码 → 偏移，与 cleaner_sim/grid.py 的 DIR_CODES 一一对应 */
const DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0], [1, -1], [1, 1], [-1, 1], [-1, -1]];

const $ = (id) => document.getElementById(id);

const state = {
  floorplans: [],
  selected: null,
  data: null,
  w: 0, h: 0, n: 0, radius: 2,
  wall: null, cleanable: null,
  cleanedAt: null, visits: null, maxVisits: 1,
  cells: null, phaseOfStep: null,
  phases: [],
  step: 0, playing: false, rate: 1, baseRate: 60, lastT: 0,
  rafId: 0, reqSeq: 0,
  off: null, octx: null, img: null,
};

/* ------------------------------------------------------------------ 工具 */
function toast(msg, ms = 3600) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), ms);
}

function b64Bytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function decodeMask(b64, n) {
  const bytes = b64Bytes(b64);
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = (bytes[i >> 3] >> (i & 7)) & 1;
  return out;
}

function fmt(n, d = 0) {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  return Number(n).toFixed(d);
}
function pct(x, d = 1) { return (x * 100).toFixed(d) + '%'; }
function clockFmt(sec) {
  sec = Math.round(sec);
  if (sec < 60) return sec + 's';
  const m = Math.floor(sec / 60), s = sec % 60;
  if (m < 60) return m + '分' + String(s).padStart(2, '0') + '秒';
  return Math.floor(m / 60) + '小时' + String(m % 60).padStart(2, '0') + '分';
}

/* =============================================================== 户型列表 */
async function loadFloorplans() {
  await refreshFloorplans();
}

function renderPicker() {
  const q = $('search').value.trim().toLowerCase();
  const tier = $('tier').value;
  const box = $('picker');
  box.innerHTML = '';
  let shown = 0;

  for (const p of state.floorplans) {
    const custom = !!p.custom;
    if (tier === 'custom' && !custom) continue;
    if (tier && tier !== 'custom' && p.tier !== tier) continue;
    if (q) {
      const hay = (p.id + ' ' + (p.name || '') + ' ' + (p.categories || []).join(' ')).toLowerCase();
      if (!hay.includes(q)) continue;
    }
    shown++;
    const btn = document.createElement('div');   // div[role=button]：卡片内还有删除按钮，避免嵌套 <button>
    btn.className = 'card' + (state.selected === p.id ? ' active' : '');
    btn.setAttribute('role', 'button');
    btn.tabIndex = 0;
    btn.title = `${p.id}\n${p.room_num} 房间 · ${p.size_x_m}×${p.size_y_m} m · ${p.area_m2} m²`
      + (custom ? '\n（用户上传）' : '');
    const label = (p.name || p.id).slice(0, 18);
    btn.innerHTML = `
      ${custom ? '<span class="badge">上传</span>' : ''}
      ${custom ? '<button class="del" title="删除">×</button>' : ''}
      <img loading="lazy" src="/api/thumb?id=${encodeURIComponent(p.id)}&w=168" alt="">
      <div class="cmeta"><span><i>${p.room_num || '—'}</i> 房</span><span>${p.size_x_m}×${p.size_y_m}m</span></div>
      <div class="cname" title="${p.id}">${label}</div>`;
    btn.onclick = (ev) => {
      if (ev.target.classList.contains('del')) return;
      selectFloorplan(p.id);
    };
    btn.onkeydown = (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); selectFloorplan(p.id); }
    };
    const del = btn.querySelector('.del');
    if (del) del.onclick = (ev) => { ev.stopPropagation(); deleteUpload(p.id); };
    box.appendChild(btn);
  }
  if (!shown) box.innerHTML = '<p class="tip">没有匹配的户型。</p>';
}

function selectFloorplan(id) {
  state.selected = id;
  renderPicker();
  const p = state.floorplans.find((x) => x.id === id);
  if (p) {
    $('stageTitle').textContent = `${p.room_num} 房间 · ${p.size_x_m} × ${p.size_y_m} m · ${p.area_m2} m²`;
    $('stageSub').textContent =
      `${p.id} ｜ 轮廓顶点 ${p.n_verts} 个 ｜ 房间类别：${(p.categories || []).join('、') || '—'}`;
    $('runHint').textContent = '参数可在左侧调整，然后点击「开始仿真」。';
  }
  $('runBtn').disabled = false;
}

/* =============================================================== 运行仿真 */
async function runSimulation() {
  if (!state.selected) return;
  const seq = ++state.reqSeq;             // 请求序号：迟到的旧响应一律丢弃
  const params = new URLSearchParams({
    id: state.selected,
    cell_cm: $('cellCm').value,
    robot_cm: $('robotCm').value,
    target_cov: (Number($('targetCov').value) / 100).toFixed(4),
    speed_mps: (Number($('speedMps').value) / 100).toFixed(3),
    turn_time_s: '0.6',
    max_retreat: '4',
    spacing_mode: $('spacingMode').value,
  });

  const btn = $('runBtn');
  btn.disabled = true;
  btn.textContent = '仿真计算中…';
  $('metaStatus').className = 'chip chip-warn';
  $('metaStatus').textContent = '计算中';

  try {
    const res = await fetch('/api/run?' + params.toString());
    const json = await res.json();
    if (seq !== state.reqSeq) return;     // 已有更新的请求（或删除操作），丢弃本次结果
    if (!res.ok || json.error) throw new Error(json.error || ('HTTP ' + res.status));
    prepare(json);
    $('metaStatus').className = 'chip chip-ok';
    $('metaStatus').textContent = '就绪';
  } catch (err) {
    if (seq !== state.reqSeq) return;
    toast('仿真失败：' + err.message, 6000);
    $('metaStatus').className = 'chip chip-err';
    $('metaStatus').textContent = '失败';
  } finally {
    if (seq === state.reqSeq) {
      btn.disabled = false;
      btn.textContent = '开始仿真';
    }
  }
}

/* 解码 + 烘焙（一次 O(步数 × 足迹)），之后任意步数都能瞬时定位 */
function prepare(data) {
  state.data = data;
  const g = data.grid;
  const w = state.w = g.w, h = state.h = g.h, n = state.n = w * h;
  state.radius = g.radius_cells;
  state.wall = decodeMask(g.wall_b64, n);
  state.cleanable = decodeMask(g.cleanable_b64, n);
  state.separated = decodeMask(g.separated_b64, n);
  state.tight = decodeMask(g.tight_b64, n);

  const start = data.replay.start;
  const dirs = b64Bytes(data.replay.dirs_b64);
  const cells = new Int32Array(dirs.length + 1);
  let x = start[0], y = start[1];
  cells[0] = y * w + x;
  for (let i = 0; i < dirs.length; i++) {
    const d = DIRS[dirs[i]];
    if (!d) { cells[i + 1] = cells[i]; continue; }
    x += d[0]; y += d[1];
    cells[i + 1] = y * w + x;
  }
  state.cells = cells;
  state.phases = data.phases || [];

  bake();

  /* 自检：浏览器按方向码重放出的覆盖率，必须与服务端算出的值一致
     （两边用完全相同的方形足迹规则，这也是动画与看板数字不会打架的保证） */
  let hit = 0, all = 0;
  for (let i = 0; i < n; i++) {
    if (state.cleanable[i]) { all++; if (state.cleanedAt[i] >= 0) hit++; }
  }
  const clientCov = all ? hit / all : 0;
  const serverCov = data.coverage.coverage_reachable;
  state.selfCheck = { clientCov, serverCov, diff: Math.abs(clientCov - serverCov) };
  if (state.selfCheck.diff > 0.002) {
    console.warn('重放覆盖率与服务端不一致', state.selfCheck);
  }

  /* 每一步属于哪个规划阶段（用于分区着色）
     -1 哨兵 + 两遍填充：先让细分相位（单元 / 死区回退）占位，主扫只补剩余步。
     否则主扫相位会“先到先得”占满整个主扫区间，单元相位永远拿不到自己的颜色。 */
  const phaseOfStep = new Int16Array(cells.length).fill(-1);
  const claim = (p, idx) => {
    const from = Math.max(0, p.start_step);
    const to = Math.min(p.end_step, phaseOfStep.length - 1);
    for (let s = from; s <= to; s++) {
      if (phaseOfStep[s] === -1) phaseOfStep[s] = idx + 1;
    }
  };
  state.phases.forEach((p, idx) => { if (p.kind !== 'sweep') claim(p, idx); });
  state.phases.forEach((p, idx) => { if (p.kind === 'sweep') claim(p, idx); });
  state.phaseOfStep = phaseOfStep;

  const off = state.off || (state.off = document.createElement('canvas'));
  off.width = w; off.height = h;
  state.octx = off.getContext('2d');
  state.img = state.octx.createImageData(w, h);

  state.step = 0;
  state.baseRate = Math.max(20, cells.length / 18);   // 1× 约 18 秒跑完
  $('scrub').max = cells.length - 1;
  $('scrub').value = 0;
  $('scrub').disabled = false;
  $('playBtn').disabled = false;
  $('resetBtn').disabled = false;
  $('canvasEmpty').style.display = 'none';

  renderDashboard();
  fitCanvas();
  render();
  play(true);
}

/* 沿路径逐格叠印方形足迹，记录每格的首次清扫步与访问次数 */
function bake() {
  const { w, h, n, radius: r } = state;
  const wall = state.wall, cells = state.cells;
  const cleanedAt = new Int32Array(n).fill(-1);
  const visits = new Uint8Array(n);
  let maxVisits = 1;
  let walkable = 0;
  for (let i = 0; i < n; i++) if (!wall[i]) walkable++;
  state.walkableCount = walkable;

  for (let s = 0; s < cells.length; s++) {
    const c = cells[s];
    const cy = (c / w) | 0, cx = c - cy * w;
    for (let dy = -r; dy <= r; dy++) {
      const ny = cy + dy;
      if (ny < 0 || ny >= h) continue;
      const base = ny * w;
      for (let dx = -r; dx <= r; dx++) {
        const nx = cx + dx;
        if (nx < 0 || nx >= w) continue;
        const j = base + nx;
        if (wall[j]) continue;
        if (cleanedAt[j] < 0) cleanedAt[j] = s;
        if (visits[j] < 255) {
          visits[j]++;
          if (visits[j] > maxVisits) maxVisits = visits[j];
        }
      }
    }
  }
  state.cleanedAt = cleanedAt;
  state.visits = visits;
  state.maxVisits = maxVisits;
}

/* ================================================================== 渲染 */
function fitCanvas() {
  const cv = $('view');
  const wrap = cv.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const availW = wrap.clientWidth - 4, availH = wrap.clientHeight - 4;
  const scale = Math.min(availW / state.w, availH / state.h);
  const cssW = Math.max(80, Math.round(state.w * scale));
  const cssH = Math.max(80, Math.round(state.h * scale));
  cv.width = Math.round(cssW * dpr);        // backing store 按 dpr 放大，HiDPI 下不模糊
  cv.height = Math.round(cssH * dpr);
  cv.style.width = cssW + 'px';
  cv.style.height = cssH + 'px';
}

function phaseColor(idx) {
  const hue = (idx * 67) % 360;
  return hsl2rgb(hue, 0.62, 0.55);
}
function hsl2rgb(h, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = h / 60, xx = c * (1 - Math.abs((hp % 2) - 1));
  let r = 0, g = 0, b = 0;
  if (hp < 1) [r, g, b] = [c, xx, 0];
  else if (hp < 2) [r, g, b] = [xx, c, 0];
  else if (hp < 3) [r, g, b] = [0, c, xx];
  else if (hp < 4) [r, g, b] = [0, xx, c];
  else if (hp < 5) [r, g, b] = [xx, 0, c];
  else [r, g, b] = [c, 0, xx];
  const m = l - c / 2;
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}

const PALETTE = {
  wall: [42, 50, 65],
  floor: [214, 224, 238],
  clean: [34, 195, 166],
  sep: [107, 75, 143],
  tight: [240, 169, 43],
};

function render() {
  if (!state.data) return;
  const { w, h, n } = state;
  const S = Math.floor(state.step);
  const d = state.img.data;
  const wall = state.wall, cleanable = state.cleanable;
  const separated = state.separated, tight = state.tight;
  const cleanedAt = state.cleanedAt, visits = state.visits;
  const heat = $('showHeat').checked;
  const showDead = $('showDead').checked;
  const phaseOn = $('showPhase').checked;
  const phaseOfStep = state.phaseOfStep;
  const maxV = Math.max(2, state.maxVisits);
  const phaseColors = state.phases.map((_, i) => phaseColor(i));

  for (let i = 0; i < n; i++) {
    let r, g, b;
    if (wall[i]) {
      [r, g, b] = PALETTE.wall;
    } else {
      const done = cleanedAt[i] >= 0 && cleanedAt[i] <= S;
      if (!cleanable[i] && !done && showDead) {
        [r, g, b] = separated[i] ? PALETTE.sep : PALETTE.tight;   // 不可达 / 死角
      } else if (done) {
        if (heat) {
          const t = (visits[i] - 1) / (maxV - 1);      // 0 = 只扫过一遍
          r = 34 + 215 * t; g = 195 - 150 * t; b = 166 - 110 * t;
        } else if (phaseOn && phaseOfStep) {
          const ph = phaseOfStep[cleanedAt[i]];
          const c = (ph > 0 && phaseColors[ph - 1]) || [34, 195, 166];
          r = c[0]; g = c[1]; b = c[2];
        } else {
          [r, g, b] = PALETTE.clean;
        }
      } else {
        [r, g, b] = PALETTE.floor;
      }
    }
    const o = i * 4;
    d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
  }
  state.octx.putImageData(state.img, 0, 0);

  const cv = $('view');
  const ctx = cv.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(state.off, 0, 0, cv.width, cv.height);

  const sx = cv.width / w, sy = cv.height / h;

  /* 已走过的轨迹（最近一段，避免万级折线拖慢绘制） */
  if ($('showPath').checked && state.cells) {
    const from = Math.max(0, S - 260);
    ctx.strokeStyle = 'rgba(255,255,255,.55)';
    ctx.lineWidth = Math.max(1, Math.min(sx, sy) * 0.28);
    ctx.beginPath();
    for (let s = from; s <= S; s++) {
      const c = state.cells[s];
      const cy = (c / w) | 0, cx = c - cy * w;
      const px = (cx + 0.5) * sx, py = (cy + 0.5) * sy;
      if (s === from) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  /* 机器人：方形足迹 + 朝向 */
  const c = state.cells[Math.min(S, state.cells.length - 1)];
  const cy = (c / w) | 0, cx = c - cy * w;
  const r = state.radius;
  const fx = (cx - r) * sx, fy = (cy - r) * sy;
  const fw = (2 * r + 1) * sx, fh = (2 * r + 1) * sy;

  ctx.fillStyle = 'rgba(79,124,255,.20)';
  ctx.fillRect(fx, fy, fw, fh);
  ctx.strokeStyle = 'rgba(122,162,255,.85)';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(fx + 0.5, fy + 0.5, fw - 1, fh - 1);

  /* 机器人：方形足迹 + 朝向箭头
     朝向以前进方向为准；静止在起点时与仿真模型的初始朝向一致（+x 方向）。
     设计：圆形机身 + 从机身伸出的"箭杆 + 实心三角箭头"，带描边阴影，
     保证在深色墙体、浅色地面、青色已清扫区域上都有足够对比度。 */
  const prev = state.cells[Math.max(0, Math.min(S, state.cells.length - 1) - 1)];
  const py = (prev / w) | 0, px = prev - py * w;
  let hx = cx - px, hy = cy - py;
  if (hx === 0 && hy === 0) { hx = 1; hy = 0; }    // 起点：与 simulate.py 初始朝向（+x）一致
  const hl = Math.hypot(hx, hy) || 1;
  const ux = hx / hl, uy = hy / hl;
  const cxp = (cx + 0.5) * sx, cyp = (cy + 0.5) * sy;

  const bodyR = Math.max(3.5, Math.min(sx, sy) * 0.7);
  const arrowLen = Math.max(bodyR * 2.2, Math.min(fw, fh) * 0.78);
  const headLen = Math.max(5, arrowLen * 0.42);
  const headHalf = Math.max(3.2, arrowLen * 0.30);
  const tipX = cxp + ux * arrowLen, tipY = cyp + uy * arrowLen;
  const baseX = tipX - ux * headLen, baseY = tipY - uy * headLen;
  const nx = -uy, ny = ux;                         // 箭头底边的垂直方向

  ctx.save();
  ctx.shadowColor = 'rgba(8,10,16,.75)';           // 阴影描边：任何底色上都看得清
  ctx.shadowBlur = 3;

  ctx.beginPath();                                 // 箭杆
  ctx.moveTo(cxp, cyp);
  ctx.lineTo(baseX, baseY);
  ctx.strokeStyle = '#ffd166';
  ctx.lineWidth = Math.max(2.2, bodyR * 0.6);
  ctx.lineCap = 'round';
  ctx.stroke();

  ctx.beginPath();                                 // 实心三角箭头
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(baseX + nx * headHalf, baseY + ny * headHalf);
  ctx.lineTo(baseX - nx * headHalf, baseY - ny * headHalf);
  ctx.closePath();
  ctx.fillStyle = '#ffd166';
  ctx.fill();
  ctx.restore();

  ctx.beginPath();                                 // 机身圆（压在箭杆起点之上）
  ctx.arc(cxp, cyp, bodyR, 0, Math.PI * 2);
  ctx.fillStyle = '#4f7cff';
  ctx.fill();
  ctx.strokeStyle = '#dfe8ff';
  ctx.lineWidth = 1.4;
  ctx.stroke();

  /* 高亮当前所处阶段（仅在切换时改 DOM，避免每帧重排） */
  const activeIdx = state.phases.findIndex((p) => S >= p.start_step && S <= p.end_step);
  if (activeIdx !== state._activePhase) {
    state._activePhase = activeIdx;
    const rows = $('phases').querySelectorAll('.row');
    rows.forEach((el, i) => el.style.borderColor = (i === activeIdx) ? 'var(--brand)' : '');
  }
}

/* ================================================================ 看板 */
function renderDashboard() {
  const cov = state.data.coverage;
  const m = state.data.metrics;
  const w = state.data.grid.w, h = state.data.grid.h;
  const area = state.data.pipeline.occupancy_stats;

  $('covBig').textContent = pct(cov.coverage_walkable, 1);
  $('covSub').textContent = `可达区域 ${pct(cov.coverage_reachable, 1)}`;

  const C = 2 * Math.PI * 52, C2 = 2 * Math.PI * 40;
  $('donutWalk').setAttribute('stroke-dasharray', `${C * cov.coverage_walkable} ${C}`);
  $('donutReach').setAttribute('stroke-dasharray', `${C2 * cov.coverage_reachable} ${C2}`);

  const cells = [
    ['清扫面积', fmt(cov.covered_area_m2, 1) + ' m²', 'good', `可清扫 ${fmt(cov.cleanable_area_m2, 1)} m² / 全屋 ${fmt(cov.walkable_area_m2, 1)} m²`],
    ['覆盖率(全屋)', pct(cov.coverage_walkable, 2), cov.coverage_walkable > 0.95 ? 'good' : 'warn', `含不可达区域`],
    ['覆盖率(可达)', pct(cov.coverage_reachable, 2), cov.coverage_reachable > 0.99 ? 'good' : 'warn', `目标 ${pct(state.data.params.target_cov, 1)}`],
    ['路径效率', pct(m.path_efficiency, 1), m.path_efficiency > 0.7 ? 'good' : 'warn',
      `相对理论下限 ${fmt(m.ideal_steps, 0)} 步`],
    ['总步数', m.steps, '', `${fmt(m.steps_per_m2, 2)} 步/m²`],
    ['路径长度', fmt(m.path_length_m, 1) + ' m', '', `单位面积 ${fmt(m.area_per_meter, 3)} m²/m`],
    ['重定位占比', m.reposition_ratio == null ? '—' : pct(m.reposition_ratio, 1), '',
      '非清扫移动占全程比例'],
    ['扫描间距', (m.sweep_spacing_cells || '—') + ' 格', '',
      `BCD 单元 ${m.cells == null ? '—' : m.cells} 个 · ${m.decomposition_axis || '-'} 向分解`],
    ['足迹重复度', pct(m.repeat_rate, 0), '',
      `每个格子平均被足迹覆盖 ${fmt(1 + m.repeat_rate, 1)} 次（受足迹宽度决定）`],
    ['预计耗时', m.duration_text, '', `${fmt(m.time_per_m2_s, 1)} s/m² · ${fmt(m.area_per_hour_m2, 0)} m²/h`],
    ['转向次数', m.turns, '', `转向耗时 ${fmt(m.turn_seconds, 0)} s`],
    ['不可达区域', pct(cov.separated_ratio, 1), cov.separated_ratio > 0.02 ? 'bad' : 'good', `${fmt(cov.separated_area_m2, 1)} m² · 需分区多次清扫`],
    ['死角', pct(cov.tight_ratio, 1), cov.tight_ratio > 0.02 ? 'warn' : 'good', `${fmt(cov.tight_area_m2, 1)} m² · 任何位姿都够不到`],
    ['死区回退', m.retreat_passes, '', `算法：${m.algorithm || '-'} · 目标 ${pct(state.data.params.target_cov, 1)}`],
    ['栅格', `${w}×${h}`, '', `${state.data.grid.cell_cm} cm/格 · 足迹 ${area.footprint_cm} cm`],
  ];
  $('metrics').innerHTML = cells.map(([k, v, cls, sub]) => `
    <div class="metric ${cls}">
      <b>${v}</b><span>${k}</span>
      <span class="rsub" style="display:block;font-size:10.5px;color:var(--ink-mute)">${sub}</span>
    </div>`).join('');

  /* 区域列表 */
  const regions = cov.regions || [];
  $('regions').innerHTML = regions.map((r) => `
    <div class="row">
      <span class="rname">区域 #${r.region} ${r.robot_can_enter ? '' : '（机器人无法进入）'}</span>
      <span class="rval">${pct(r.coverage, 1)}</span>
      <span class="rsub">${fmt(r.area_m2, 1)} m² · 已清扫 ${fmt(r.covered_cells * area.cell_area_m2, 1)} m²</span>
    </div>`).join('') || '<p class="tip">单一连通区域。</p>';

  const clusters = cov.dead_clusters || [];
  if (clusters.length) {
    $('regions').innerHTML += clusters.map((c) => {
      const tag = c.reason === 'separated' ? 'sep' : c.reason === 'mixed' ? 'mixed' : 'tight';
      const name = c.reason === 'separated' ? '不可达区' : c.reason === 'mixed' ? '混合死区' : '死角';
      return `
      <div class="row tag-${tag}">
        <span class="rname">${name}</span>
        <span class="rval">${fmt(c.area_m2, 2)} m²</span>
        <span class="rsub">${c.reason_text}</span>
      </div>`;
    }).join('');
  }

  /* 阶段列表 */
  $('phases').innerHTML = state.phases.map((p, i) => `
    <div class="row clickable" data-step="${p.start_step}">
      <span class="rname">${p.name}</span>
      <span class="rval">${p.start_step}–${p.end_step}</span>
      <span class="rsub">该阶段结束时覆盖 ${pct(p.coverage, 1)}</span>
    </div>`).join('') || '<p class="tip">—</p>';
  $('phases').querySelectorAll('[data-step]').forEach((el) => {
    el.tabIndex = 0;
    el.setAttribute('role', 'button');
    const jump = () => { pause(); state.step = Number(el.dataset.step); syncScrub(); render(); renderChart(); };
    el.onclick = jump;
    el.onkeydown = (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); jump(); }
    };
  });

  /* 栅格化管线 */
  const stages = state.data.pipeline.stages || [];
  const sc = state.selfCheck;
  $('stages').innerHTML = stages.map((s) => `
    <div class="row">
      <span class="rname">${s.name}</span>
      <span class="rval">${s.ms} ms</span>
      <span class="rsub">${s.detail}</span>
    </div>`).join('') +
    `<div class="row"><span class="rname">合计</span><span class="rval">${state.data.pipeline.total_ms} ms</span>
     <span class="rsub">纯 Python 标准库，无第三方依赖</span></div>` +
    (sc ? `<div class="row${sc.diff > 0.002 ? ' tag-tight' : ''}">
      <span class="rname">浏览器重放自检</span>
      <span class="rval">${sc.diff < 0.0005 ? '一致' : '差异 ' + pct(sc.diff, 2)}</span>
      <span class="rsub">前端重放 ${pct(sc.clientCov, 3)} vs 服务端 ${pct(sc.serverCov, 3)}</span>
    </div>` : '');

  renderChart();
}

/* 覆盖率曲线 */
function renderChart() {
  const cv = $('chart');
  const dpr = window.devicePixelRatio || 1;
  const wantW = Math.max(120, Math.round(cv.clientWidth * dpr));
  const wantH = Math.max(90, Math.round((cv.clientHeight || 120) * dpr));
  if (cv.width !== wantW) cv.width = wantW;      // 仅在尺寸变化时重建，避免每帧重置画布
  if (cv.height !== wantH) cv.height = wantH;

  const w = cv.width;
  const h = cv.height;
  const ctx = cv.getContext('2d');
  const tl = state.data ? state.data.timeline : null;
  ctx.clearRect(0, 0, w, h);
  if (!tl || !tl.length) return;

  const padL = 34 * dpr, padB = 18 * dpr, padT = 10 * dpr, padR = 8 * dpr;
  const total = Math.max(1, state.cells.length - 1);
  const X = (s) => padL + (w - padL - padR) * (s / total);
  const Y = (v) => padT + (h - padT - padB) * (1 - v);

  ctx.strokeStyle = '#232c3a';
  ctx.lineWidth = 1;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(w - padR, Y(v)); ctx.stroke();
  }
  ctx.fillStyle = '#6f7c92';
  ctx.font = `${10 * (window.devicePixelRatio || 1)}px monospace`;
  ctx.fillText('100%', 4, Y(1) + 3);
  ctx.fillText('50%', 10, Y(0.5) + 3);
  ctx.fillText('0', 22, Y(0) + 3);

  ctx.beginPath();
  tl.forEach(([s, v], i) => { const px = X(s), py = Y(v); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
  ctx.strokeStyle = '#39435a';
  ctx.lineWidth = 1.5 * (window.devicePixelRatio || 1);
  ctx.stroke();

  const S = Math.floor(state.step);
  ctx.beginPath();
  let started = false;
  for (const [s, v] of tl) {
    if (s > S) break;
    const px = X(s), py = Y(v);
    started ? ctx.lineTo(px, py) : (ctx.moveTo(px, py), started = true);
  }
  ctx.strokeStyle = '#22c3a6';
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
  ctx.stroke();

  const cx = X(S);
  ctx.beginPath();
  ctx.moveTo(cx, padT); ctx.lineTo(cx, h - padB);
  ctx.strokeStyle = 'rgba(255,209,102,.85)';
  ctx.lineWidth = 1.4 * (window.devicePixelRatio || 1);
  ctx.stroke();

  ctx.fillStyle = '#6f7c92';
  ctx.fillText('0', padL, h - 5);
  ctx.fillText(String(total), w - padR - 30, h - 5);
}

/* ============================================================== 播放控制 */
function play(force) {
  if (state.playing && !force) return;
  if (state.rafId) cancelAnimationFrame(state.rafId);   // 先取消旧循环，避免双链 2× 播放
  state.rafId = 0;
  state.playing = true;
  $('playBtn').textContent = '❚❚';
  state.lastT = performance.now();
  state.rafId = requestAnimationFrame(tick);
}

function pause() {
  state.playing = false;
  if (state.rafId) { cancelAnimationFrame(state.rafId); state.rafId = 0; }
  $('playBtn').textContent = '▶';
}

function syncScrub() {
  $('scrub').value = Math.floor(state.step);
  $('stepNow').textContent = Math.floor(state.step);
  $('stepTotal').textContent = state.cells ? state.cells.length - 1 : 0;
}

function tick(now) {
  state.rafId = 0;
  if (!state.playing) return;
  const dt = Math.min(0.1, (now - state.lastT) / 1000);
  state.lastT = now;
  const last = state.cells.length - 1;
  state.step += state.baseRate * state.rate * dt;
  if (state.step >= last) {
    state.step = last;
    pause();
  }
  syncScrub();
  render();
  renderChart();
  if (state.playing) state.rafId = requestAnimationFrame(tick);
}

/* ============================================================ 户型图上传 */
const IMG_MAX_SIDE = 560;          // 上传图像的降采样上限：越大越精确，Python 解析越慢

const upload = { gray: null, width: 0, height: 0, name: '', url: '' };

function setHint(text, cls) {
  const el = $('uploadHint');
  el.textContent = text;
  el.style.color = cls === 'err' ? 'var(--rose)' : cls === 'ok' ? 'var(--teal)' : '';
}

async function handleFiles(files) {
  const file = files && files[0];
  if (!file) return;
  const name = file.name || 'upload';
  const isJson = /\.json$/i.test(name) || file.type === 'application/json';
  const isImage = /^image\//.test(file.type) || /\.(png|jpe?g|webp|gif|bmp)$/i.test(name);

  try {
    if (isJson) {
      setHint('正在上传 HouseExpo JSON…');
      const text = await file.text();
      const res = await fetch('/api/upload/houseexpo', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, text }),
      });
      const json = await res.json();
      if (!res.ok || json.error) throw new Error(json.error || ('HTTP ' + res.status));
      await refreshFloorplans();
      selectFloorplan(json.id);
      setHint(`已加入：${name}（${json.meta.room_num} 房，${json.meta.size_x_m}×${json.meta.size_y_m} m）`, 'ok');
      toast('户型已加入列表，可直接仿真');
    } else if (isImage) {
      setHint('正在解码图片…');
      await decodeImage(file);
      openImageModal(name);
    } else {
      throw new Error('不支持的文件类型，请选择 .json 或图片');
    }
  } catch (err) {
    setHint('上传失败：' + err.message, 'err');
    toast('上传失败：' + err.message, 6000);
  }
}

/* 浏览器负责解码 PNG/JPG，然后只把灰度缓冲发给服务端 */
async function decodeImage(file) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, IMG_MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(8, Math.round(bitmap.width * scale));
  const h = Math.max(8, Math.round(bitmap.height * scale));

  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;

  const gray = new Uint8Array(w * h);
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    gray[i] = (data[j] * 299 + data[j + 1] * 587 + data[j + 2] * 114) / 1000 | 0;
  }
  upload.gray = gray;
  upload.width = w;
  upload.height = h;
  upload.name = file.name || 'floorplan';
  if (upload.url) URL.revokeObjectURL(upload.url);
  upload.url = URL.createObjectURL(file);
  $('srcPreview').src = upload.url;
  $('imgName').textContent = `${file.name} · ${w}×${h} px`;
  setHint(`已解码 ${w}×${h} 灰度图`, 'ok');
}

function imageBody(extra = {}) {
  return Object.assign({
    name: upload.name,
    width: upload.width,
    height: upload.height,
    gray_b64: bytesToB64(upload.gray),
    threshold: $('thrAuto').checked ? null : Number($('thr').value),
    invert: $('invert').checked,
    seal_px: Number($('seal').value),
    real_width_m: Number($('realW').value),
    scale_axis: 'x',
  }, extra);
}

function bytesToB64(bytes) {
  let s = '';
  const CH = 0x8000;
  for (let i = 0; i < bytes.length; i += CH) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
  }
  return btoa(s);
}

let modalReturnFocus = null;

function openImageModal(name) {
  modalReturnFocus = document.activeElement;
  $('imgModal').hidden = false;
  $('imgName').textContent = `${name || upload.name} · ${upload.width}×${upload.height} px`;
  const box = $('imgModal').querySelector('.modal-box');
  box.tabIndex = -1;
  box.focus();
  requestPreview();
}

function closeImageModal() {
  $('imgModal').hidden = true;
  if (modalReturnFocus && modalReturnFocus.focus) modalReturnFocus.focus();
  modalReturnFocus = null;
}

async function requestPreview() {
  const btn = $('imgReparse');
  btn.disabled = true;
  try {
    const res = await fetch('/api/parse/preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(imageBody()),
    });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || ('HTTP ' + res.status));
    $('parsePreview').src = 'data:image/png;base64,' + json.preview;
    const d = json.diagnostics || {};
    const warn = (d.warnings || []).map((w) => `<div class="warn-line">⚠ ${w}</div>`).join('');
    $('imgStats').innerHTML = `
      <div><b>${d.threshold}</b><span>阈值${d.threshold_auto ? (d.threshold_auto_adjusted ? '（自动优化）' : '（Otsu 自动）') : ''}</span></div>
      <div><b>${d.floor_area_m2} m²</b><span>识别面积</span></div>
      <div><b>${d.crop_px[0]}×${d.crop_px[1]}</b><span>户型像素</span></div>
      <div><b>${(d.meters_per_px * 100).toFixed(1)} cm</b><span>每像素实际尺寸</span></div>
      <div><b>${(d.wall_fraction * 100).toFixed(1)}%</b><span>墙体占比</span></div>
      <div><b>${d.dropped_islands}</b><span>剔除的碎片</span></div>
      ${warn}`;
  } catch (err) {
    $('imgStats').innerHTML = `<div class="warn-line">解析失败：${err.message}</div>`;
  } finally {
    btn.disabled = false;
  }
}

async function saveImage() {
  const btn = $('imgSave');
  btn.disabled = true;
  btn.textContent = '解析中…';
  try {
    const res = await fetch('/api/upload/image', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(imageBody()),
    });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || ('HTTP ' + res.status));
    closeImageModal();
    await refreshFloorplans();
    selectFloorplan(json.id);
    setHint(`已加入：${upload.name}（${json.meta.size_x_m}×${json.meta.size_y_m} m）`, 'ok');
    toast('图片已解析并加入列表');
    runSimulation();
  } catch (err) {
    toast('保存失败：' + err.message, 6000);
  } finally {
    btn.disabled = false;
    btn.textContent = '加入列表并仿真';
  }
}

async function deleteUpload(id) {
  if (!confirm('从列表中删除这个上传的户型？')) return;
  try {
    const res = await fetch('/api/delete', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    });
    const json = await res.json();
    if (!res.ok || json.error) throw new Error(json.error || ('HTTP ' + res.status));
    if (state.selected === id) {
      state.selected = null;
      state.data = null;
      state.cells = null;
      state.phases = [];
      state.phaseOfStep = null;
      state.step = 0;
      state.reqSeq++;                       // 作废可能仍在途的仿真响应
      pause();
      $('scrub').max = 0; $('scrub').value = 0; $('scrub').disabled = true;
      $('playBtn').disabled = true;
      $('resetBtn').disabled = true;
      $('stepNow').textContent = '0';
      $('stepTotal').textContent = '0';
      $('metrics').innerHTML = '';
      $('regions').innerHTML = '';
      $('phases').innerHTML = '';
      $('stages').innerHTML = '';
      $('covBig').textContent = '—';
      $('covSub').textContent = '可达区域 —';
      $('donutWalk').setAttribute('stroke-dasharray', '0 400');
      $('donutReach').setAttribute('stroke-dasharray', '0 300');
      $('canvasEmpty').style.display = '';
      const cv = $('view');
      cv.getContext('2d').clearRect(0, 0, cv.width, cv.height);
      $('runBtn').disabled = true;
      renderChart();
    }
    await refreshFloorplans();
    toast('已删除');
  } catch (err) {
    toast('删除失败：' + err.message, 5000);
  }
}

async function refreshFloorplans() {
  const res = await fetch('/api/floorplans');
  const json = await res.json();
  state.floorplans = json.floorplans || [];
  const custom = state.floorplans.filter((p) => p.custom).length;
  const where = (json.data_dir || '').split(/[\\/]/).slice(-2).join('/');
  $('metaData').textContent = `${state.floorplans.length} 个户型`
    + (custom ? `（含 ${custom} 个上传）` : '') + (where ? ` · ${where}` : '');
  $('planCount').textContent = `${state.floorplans.length} 个`;
  renderPicker();
}

/* ================================================================== 绑定 */
function bind() {
  $('search').oninput = renderPicker;
  $('tier').onchange = renderPicker;

  const bindOut = (id, out, fmtFn) => {
    const el = $(id);
    const upd = () => { $(out).textContent = fmtFn(el.value); };
    el.oninput = upd; upd();
  };
  bindOut('cellCm', 'cellOut', (v) => `${v} cm / 格`);
  bindOut('robotCm', 'robotOut', (v) => `${v} cm`);
  bindOut('targetCov', 'covOut', (v) => `${Number(v).toFixed(1)}%`);
  bindOut('speedMps', 'speedOut', (v) => `${(v / 100).toFixed(2)} m/s`);

  $('runBtn').onclick = runSimulation;
  $('playBtn').onclick = () => (state.playing ? pause() : play());
  $('resetBtn').onclick = () => { pause(); state.step = 0; syncScrub(); render(); renderChart(); };
  $('scrub').oninput = (e) => { pause(); state.step = Number(e.target.value); syncScrub(); render(); renderChart(); };
  $('rate').onchange = (e) => { state.rate = Number(e.target.value); };
  ['showHeat', 'showPath', 'showDead', 'showPhase'].forEach((id) => {
    $(id).onchange = () => render();
  });

  /* ---- 上传 ---- */
  const drop = $('drop');
  const fileInput = $('file');
  drop.onclick = () => fileInput.click();
  drop.onkeydown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  };
  fileInput.onchange = () => { handleFiles(fileInput.files); fileInput.value = ''; };
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); e.stopPropagation(); drop.classList.add('over');
  }));
  ['dragleave'].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.remove('over');
  }));
  drop.addEventListener('drop', (e) => {
    e.preventDefault(); e.stopPropagation(); drop.classList.remove('over');
    handleFiles(e.dataTransfer.files);
  });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
  });

  /* ---- 图片解析弹层 ---- */
  let previewTimer = 0;
  const schedulePreview = () => {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(requestPreview, 260);
  };
  $('imgMask').onclick = closeImageModal;
  $('imgCancel').onclick = closeImageModal;
  $('imgReparse').onclick = requestPreview;
  $('imgSave').onclick = saveImage;

  /* 焦点陷阱：Tab 只在弹层内部循环 */
  $('imgModal').addEventListener('keydown', (e) => {
    if (e.key !== 'Tab') return;
    const list = Array.from(
      $('imgModal').querySelectorAll('button, [href], input, select, [tabindex]:not([tabindex="-1"])')
    ).filter((el) => !el.disabled && el.offsetParent !== null);
    if (!list.length) return;
    const first = list[0], last = list[list.length - 1];
    const idx = list.indexOf(document.activeElement);
    if (idx === -1) { e.preventDefault(); (e.shiftKey ? last : first).focus(); return; }
    if (e.shiftKey && idx === 0) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && idx === list.length - 1) { e.preventDefault(); first.focus(); }
  });

  const syncThr = () => {
    const auto = $('thrAuto').checked;
    $('thr').disabled = auto;
    $('thrOut').textContent = auto ? '自动' : $('thr').value;
  };
  $('thrAuto').onchange = () => { syncThr(); schedulePreview(); };
  $('thr').oninput = () => { syncThr(); schedulePreview(); };
  $('seal').oninput = () => { $('sealOut').textContent = $('seal').value + ' px'; schedulePreview(); };
  $('realW').oninput = () => { $('realWOut').textContent = Number($('realW').value).toFixed(1) + ' m'; };
  $('invert').onchange = schedulePreview;
  syncThr();

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !$('imgModal').hidden) closeImageModal();
  });

  window.addEventListener('resize', () => { if (state.data) { fitCanvas(); render(); renderChart(); } });
}

(async function init() {
  bind();
  try {
    await loadFloorplans();
    if (state.floorplans.length) selectFloorplan(state.floorplans[0].id);
  } catch (err) {
    toast('无法读取户型列表：' + err.message, 8000);
    $('metaStatus').className = 'chip chip-err';
    $('metaStatus').textContent = '离线';
  }
})();
