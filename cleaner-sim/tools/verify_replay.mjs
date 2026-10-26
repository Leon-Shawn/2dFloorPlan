/* End-to-end contract test for the replay payload.
 *
 * Re-implements web/app.js's decode + bake step (same direction codes, same square
 * footprint rule) and checks that the coverage it reproduces matches what the Python
 * server computed. If this passes, the animation and the dashboard cannot disagree.
 *
 * Usage: node tools/verify_replay.mjs [port]
 */
import http from 'node:http';

const PORT = Number(process.argv[2] || 8017);
const DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0], [1, -1], [1, 1], [-1, 1], [-1, -1]];

function get(path) {
  return new Promise((resolve, reject) => {
    const req = http.get({ host: '127.0.0.1', port: PORT, path, timeout: 240000 }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => resolve({ status: res.statusCode, body: Buffer.concat(chunks) }));
    });
    req.on('error', reject);
    req.on('timeout', () => req.destroy(new Error('timeout ' + path)));
  });
}

function decodeMask(b64, n) {
  const bytes = Buffer.from(b64, 'base64');
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = (bytes[i >> 3] >> (i & 7)) & 1;
  return out;
}

function replay(data) {
  const { w, h } = data.grid;
  const n = w * h;
  const r = data.grid.radius_cells;
  const wall = decodeMask(data.grid.wall_b64, n);
  const cleanable = decodeMask(data.grid.cleanable_b64, n);

  const dirs = Buffer.from(data.replay.dirs_b64, 'base64');
  let [x, y] = data.replay.start;
  const cells = new Int32Array(dirs.length + 1);
  cells[0] = y * w + x;
  for (let i = 0; i < dirs.length; i++) {
    const d = DIRS[dirs[i]];
    x += d[0]; y += d[1];
    cells[i + 1] = y * w + x;
  }

  // --- exactly what bake() does in the browser
  const cleanedAt = new Int32Array(n).fill(-1);
  let stampCount = 0;
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
        stampCount++;
      }
    }
  }

  let walkTotal = 0, walkHit = 0, cleanTotal = 0, cleanHit = 0;
  const cleanableNotDone = [];
  const doneButNotCleanable = [];
  for (let i = 0; i < n; i++) {
    const done = cleanedAt[i] >= 0;
    if (!wall[i]) { walkTotal++; if (done) walkHit++; }
    if (cleanable[i]) {
      cleanTotal++;
      if (done) cleanHit++;
      else cleanableNotDone.push([i % w, (i / w) | 0]);
    } else if (done) {
      doneButNotCleanable.push([i % w, (i / w) | 0]);
    }
  }
  return {
    cells: cells.length,
    coverageWalk: walkHit / Math.max(1, walkTotal),
    coverageReach: cleanHit / Math.max(1, cleanTotal),
    // steps whose footprint actually changed something = "new cells" per step sum
    cleanedCells: walkHit,
    cleanTotal,
    cleanHit,
    samples: cleanableNotDone.slice(0, 4),
    extra: doneButNotCleanable.slice(0, 4),
  };
}

const listing = await get('/api/floorplans');
if (listing.status !== 200) {
  console.error('GET /api/floorplans failed:', listing.status);
  process.exit(1);
}
const { floorplans } = JSON.parse(listing.body.toString('utf8'));

// cover the extremes: simplest, the biggest, and the ones with split regions
const picked = [];
const byRooms = [...floorplans].sort((a, b) => a.room_num - b.room_num);
picked.push(byRooms[0], byRooms[byRooms.length - 1]);
const byArea = [...floorplans].sort((a, b) => b.area_m2 - a.area_m2);
picked.push(byArea[0], byArea[1]);
for (const p of floorplans.slice(0, 40)) if (picked.length < 7) picked.push(p);

const unique = [...new Map(picked.map((p) => [p.id, p])).values()];
let failures = 0;
console.log(`verifying replay contract on ${unique.length} floorplans\n`);
console.log('rooms   steps  replay-cov   server-cov        diff  verdict');
console.log('(verdict requires the cleaned-cell counts to match exactly; the server');
console.log(' rounds its reported ratios to 4 decimals, hence the ratio tolerance)');

for (const meta of unique) {
  const res = await get(`/api/run?id=${encodeURIComponent(meta.id)}&cell_cm=10&robot_cm=34`);
  if (res.status !== 200) {
    if (meta.custom) {
      console.log(`${String(meta.room_num).padStart(5)}  HTTP ${res.status}  `
        + 'SKIP（用户上传，自身不可规划）');
      continue;
    }
    console.log(`${String(meta.room_num).padStart(5)}  HTTP ${res.status}`);
    failures++;
    continue;
  }
  const data = JSON.parse(res.body.toString('utf8'));
  const out = replay(data);
  // The exact cell counts are the real contract; the server rounds its reported
  // ratios to 4 decimals, so compare those with a matching tolerance.
  const sameCount = out.cleanHit === data.coverage.covered_cleanable_cells
    && out.cleanTotal === data.coverage.cleanable_cells;
  const diff = Math.abs(out.coverageReach - data.coverage.coverage_reachable);
  const ok = sameCount && diff < 1e-4;
  if (!ok) failures++;
  console.log(`${String(meta.room_num).padStart(5)} ${String(out.cells - 1).padStart(7)} `
    + `${out.coverageReach.toFixed(6).padStart(11)} `
    + `${data.coverage.coverage_reachable.toFixed(6).padStart(11)} `
    + `${diff.toExponential(2).padStart(9)}  ${ok ? 'OK' : 'MISMATCH'}`);
  if (!ok) {
    console.log(`        cleanable=${out.cleanTotal} (server ${data.coverage.cleanable_cells})`
      + `  covered=${out.cleanHit} (server ${data.coverage.covered_cleanable_cells})`
      + `  countMatch=${sameCount}`);
    console.log(`        cleanable-but-not-cleaned samples: ${JSON.stringify(out.samples)}`);
    console.log(`        cleaned-but-not-cleanable samples: ${JSON.stringify(out.extra)}`);
  }
}

console.log();
if (failures) {
  console.error(`FAILED: ${failures} mismatch(es) — browser replay would disagree with the server`);
  process.exit(1);
}
console.log('PASS: browser replay reproduces the server coverage exactly on every case');
