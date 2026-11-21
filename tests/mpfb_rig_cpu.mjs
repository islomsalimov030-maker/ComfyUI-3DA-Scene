// E10c, CPU-тест web/mpfbRig.js и привязки риг MPFB в poseHandles.js: настоящий файл рига web/mpfbRig.dat,
// синтетическая сцена (скелет из test_pose_ik, вершины рига в позе, как у GLB: углы продублированы, порядок вершин
// перемешан, у нескольких квадов другая диагональ). node tests/mpfb_rig_cpu.mjs <data.json: {K (оси Z вверх), conf}>
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { parseRig, retarget, skinMatrices, skinVertices, bindMesh, editedPositions, MPFB_TRIANGLES } from '../web/mpfbRig.js';
import { solve, kpBones, applyT } from '../web/poseRig.js';
import { poseHandles } from '../web/poseHandles.js';

const { K: KZ, conf } = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const rig = await parseRig(fs.readFileSync(new URL('../web/mpfbRig.dat', import.meta.url)));
const toY = p => [p[0], p[2], -p[1]], toZ = p => [p[0], -p[2], p[1]];
const K = KZ.map(toY);                       // оси окна: Y вверх

// ---- файл рига
assert.equal(rig.nV, 54078); assert.equal(rig.nT, MPFB_TRIANGLES); assert.equal(rig.nB, 163); assert.equal(rig.top, 8);
for (let v = 0; v < rig.nV; v++) {
  let s = 0;
  for (let k = 0; k < rig.top; k++) s += rig.wgt[rig.top * v + k];
  assert.ok(Math.abs(s - 1) < 2e-4, `веса вершины ${v} дают ${s}`);
}
assert.ok(rig.tris.every(i => i < rig.nV));
assert.ok(rig.parent.every((p, b) => p < b), 'родители идут раньше детей');
assert.match(rig.header.license, /CC0/);

// ---- сцена «как GLB»
const pose0 = retarget(rig, K.map(toZ));
const fit0 = skinVertices(rig, skinMatrices(rig, pose0));
const restY = new Float32Array(rig.nV * 3);
for (let v = 0; v < rig.nV; v++) restY.set(toY([fit0[3 * v], fit0[3 * v + 1], fit0[3 * v + 2]]), 3 * v);
let seed = 12345;
const rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
function glbLike({ offsetY = 0, flipQuads = 15 } = {}) {
  const tris = Uint32Array.from(rig.tris);
  for (let q = 0, done = 0; done < flipQuads; q += 997, done++) {   // диагональ другая: (a,b,c),(c,d,a) -> (a,b,d),(b,c,d)
    const t = 2 * q, A = [...tris.subarray(3 * t, 3 * t + 3)], B = [...tris.subarray(3 * t + 3, 3 * t + 6)];
    const shared = A.filter(x => B.includes(x));
    if (shared.length !== 2) continue;
    const r = A.find(x => !shared.includes(x)), s = B.find(x => !shared.includes(x));
    tris.set([r, s, shared[0], s, r, shared[1]], 3 * t);
  }
  const n = tris.length, order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const position = new Float32Array(3 * n), index = new Uint32Array(n);
  order.forEach((corner, slot) => {                          // каждый угол — своя вершина GLB в случайном месте буфера
    const r = tris[corner];
    position.set([restY[3 * r], restY[3 * r + 1] + offsetY, restY[3 * r + 2]], 3 * slot);
    index[corner] = slot;
  });
  return { position, index, tris };
}
const scene = glbLike({ offsetY: 0.007 });
const bound = bindMesh(rig, scene.position, scene.index, K);
assert.ok(bound.ok, bound.reason);
assert.ok(bound.stats.median_m < 1e-5 && bound.stats.p99_m < 1e-4, JSON.stringify(bound.stats));
assert.ok(Math.abs(bound.stats.offset_m[1] - 0.007) < 1e-5, 'сдвиг сцены относительно рига (контакт) учтён');

// ---- без правок меш не меняется; сдвиг сохраняется
const same = editedPositions(bound, K);
let worst = 0;
for (let i = 0; i < same.length; i++) worst = Math.max(worst, Math.abs(same[i] - scene.position[i]));
assert.ok(worst < 1e-5, `без правок сдвиг ${worst}`);

// ---- правки ручек: смещение = риг(после) − риг(до); грудь не едет от кисти
const bones = kpBones(K);
const editKp = edits => { const { T } = solve(K, edits, conf); return K.map((p, i) => applyT(T[bones[i]], p)); };
function expected(kp2) {
  const moved = skinVertices(rig, skinMatrices(rig, retarget(rig, kp2.map(toZ))));
  const out = Float32Array.from(scene.position), seen = new Set();
  for (let c = 0; c < scene.index.length; c++) {
    const g = scene.index[c], r = scene.tris[c];
    if (seen.has(g)) continue;
    seen.add(g);
    const d = toY([moved[3 * r] - fit0[3 * r], moved[3 * r + 1] - fit0[3 * r + 1], moved[3 * r + 2] - fit0[3 * r + 2]]);
    for (let k = 0; k < 3; k++) out[3 * g + k] += d[k];
  }
  return out;
}
const handEdit = { hand_r: [0.04, 0.12, -0.10] };
const kpHand = editKp(handEdit);
const moved = editedPositions(bound, kpHand);
const ref = expected(kpHand);
// угол c индекса дублирует вершину рига: для сравнения берём вершины, которых коснулась ссылка выше
let diff = 0, shift = 0;
for (let g = 0; g < moved.length / 3; g++) {
  for (let k = 0; k < 3; k++) {
    if (!Number.isFinite(moved[3 * g + k])) assert.fail('NaN в позициях');
    diff = Math.max(diff, Math.abs(moved[3 * g + k] - ref[3 * g + k]));
  }
  shift = Math.max(shift, Math.hypot(moved[3 * g] - scene.position[3 * g], moved[3 * g + 1] - scene.position[3 * g + 1], moved[3 * g + 2] - scene.position[3 * g + 2]));
}
assert.ok(diff < 1e-5, `правки: расхождение с прямым расчётом ${diff}`);
assert.ok(shift > 0.05, `кисть должна сдвинуться (макс. смещение ${shift})`);
// ядро груди (вес breast.L/R >= 0.5): кисть её почти не двигает (≤ 1 мм), ручка груди — двигает
const core = new Set();
for (const name of ['breast.L', 'breast.R']) {
  const b = rig.index.get(name);
  for (let v = 0; v < rig.nV; v++) for (let k = 0; k < rig.top; k++) if (rig.idx[rig.top * v + k] === b && rig.wgt[rig.top * v + k] > 0.5) core.add(v);
}
assert.ok(core.size > 400);
const coreMove = (a, b) => { let m = 0;
  for (let c = 0; c < scene.index.length; c++) { const r = scene.tris[c]; if (!core.has(r)) continue; const g = scene.index[c];
    m = Math.max(m, Math.hypot(a[3 * g] - b[3 * g], a[3 * g + 1] - b[3 * g + 1], a[3 * g + 2] - b[3 * g + 2])); }
  return m; };
assert.ok(coreMove(moved, scene.position) < 1e-3, `грудь от кисти: ${coreMove(moved, scene.position)}`);
assert.ok(coreMove(editedPositions(bound, editKp({ chest: [0, 0.06, -0.10] })), scene.position) > 0.02, 'ручка груди двигает грудь');
// длины костей рига не меняются от ручек (IK) — вершины не растягиваются: рёбра в пределах 2.5x у кисти
{
  const e = new Map(); let worstRatio = 0;
  for (let t = 0; t < rig.nT; t += 7) for (let j = 0; j < 3; j++) {
    const a = scene.tris[3 * t + j], b = scene.tris[3 * t + (j + 1) % 3], key = a * rig.nV + b;
    if (e.has(key)) continue; e.set(key, 1);
    const ga = scene.index[3 * t + j], gb = scene.index[3 * t + (j + 1) % 3];
    const l0 = Math.hypot(scene.position[3 * ga] - scene.position[3 * gb], scene.position[3 * ga + 1] - scene.position[3 * gb + 1], scene.position[3 * ga + 2] - scene.position[3 * gb + 2]);
    const l1 = Math.hypot(moved[3 * ga] - moved[3 * gb], moved[3 * ga + 1] - moved[3 * gb + 1], moved[3 * ga + 2] - moved[3 * gb + 2]);
    if (l0 > 1e-4) worstRatio = Math.max(worstRatio, l1 / l0);
  }
  assert.ok(worstRatio < 6, `растяжение ребра ${worstRatio}`);
}

// ---- отказы привязки: окно остаётся на приблизительных весах
assert.equal(bindMesh(rig, scene.position, scene.index.slice(0, 3 * (rig.nT - 1)), K).ok, false);
assert.match(bindMesh(rig, scene.position, scene.index.slice(0, 3 * (rig.nT - 1)), K).reason, /треугольников/);
{
  const noisy = Float32Array.from(scene.position).map(x => x + (rnd() - .5) * 0.04);   // ±2 см: это уже не тот меш
  const r = bindMesh(rig, noisy, scene.index, K);
  assert.equal(r.ok, false);
}
{
  const broken = Uint32Array.from(scene.index);
  for (let c = 0; c < broken.length; c += 40) broken[c] = broken[(c + 12345) % broken.length];   // 2.5% углов перепутано
  const r = bindMesh(rig, scene.position, broken, K);
  assert.equal(r.ok, false); assert.match(r.reason, /соответствие|уникальных/);
}
{
  const other = scene.position.slice(); other[0] += 1;                       // нет совпадающих координат у дублей
  assert.equal(bindMesh(rig, other, scene.index, K).ok, false);
  // Чужая постановка: риг не может сесть на меш, построенный по другим точкам
  const far = K.map(p => [p[0] + 0.0, p[1], p[2]]); far[5] = [far[5][0] + 0.3, far[5][1], far[5][2]]; far[7] = [far[7][0] + .3, far[7][1] + .3, far[7][2]];
  const r = bindMesh(rig, scene.position, scene.index, far);
  assert.equal(r.ok, false); assert.match(r.reason, /не сел/);
}

// ---- poseHandles.js с рисом MPFB: тот же результат, что editedPositions; сброс; запасной путь
class V { constructor(x = 0, y = 0, z = 0) { this.set(x, y, z); } set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
  clone() { return new V(this.x, this.y, this.z); } copy(v) { return this.set(v.x, v.y, v.z); }
  applyMatrix4(m) { const e = m.elements, { x, y, z } = this;
    return this.set(e[0] * x + e[4] * y + e[8] * z + e[12], e[1] * x + e[5] * y + e[9] * z + e[13], e[2] * x + e[6] * y + e[10] * z + e[14]); }
  setScalar(s) { return this.set(s, s, s); } }
class M { constructor() { this.elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]; } clone() { const m = new M(); m.elements = [...this.elements]; return m; }
  multiply() { return this; } invert() { return this; } }
class Attribute { constructor(array, itemSize) { this.array = array; this.itemSize = itemSize; } }
class Geometry { constructor() { this.attributes = {}; this.userData = {}; this.index = null; }
  setAttribute(n, v) { this.attributes[n] = v; } setIndex(v) { this.index = v; }
  clone() { const g = new Geometry(); for (const [n, a] of Object.entries(this.attributes)) g.setAttribute(n, new Attribute(a.array.slice(), a.itemSize));
    if (this.index) g.setIndex(new Attribute(this.index.array.slice(), 1)); return g; }
  computeVertexNormals() {} computeBoundingSphere() {} computeBoundingBox() {} dispose() {} }
class Material { constructor() { this.color = { setHex() {} }; this.emissive = { setHex() {} }; } dispose() {} }
class Mesh { constructor(geometry, material = new Material()) { this.geometry = geometry; this.material = material; this.isMesh = true;
  this.position = new V(); this.scale = new V(1, 1, 1); this.userData = {}; this.matrixWorld = new M(); } }
let nextId = 1;
const frames = new Map(), idles = new Map();
globalThis.requestAnimationFrame = f => { const id = nextId++; frames.set(id, f); return id; };
globalThis.cancelAnimationFrame = id => frames.delete(id);
globalThis.requestIdleCallback = f => { const id = nextId++; idles.set(id, f); return id; };
globalThis.cancelIdleCallback = id => idles.delete(id);
async function drain() {
  for (const q of [frames, frames, idles]) { const tasks = [...q.values()]; q.clear(); await Promise.all(tasks.map(f => f())); }
}
async function setupHandles(loadRig, saved) {
  const g = new Geometry();
  g.setAttribute('position', new Attribute(Float32Array.from(scene.position), 3));
  g.setIndex(new Attribute(Uint32Array.from(scene.index), 1));
  const mesh = new Mesh(g);
  const object = { position: new V(), matrixWorld: new M(), traverse(fn) { fn(mesh); }, updateMatrixWorld() {},
    localToWorld(v) { return v; }, worldToLocal(v) { return v; }, getWorldScale(v) { return v.set(1, 1, 1); } };
  const sc = { items: [], add(o) { this.items.push(o); }, remove(o) { this.items.splice(this.items.indexOf(o), 1); } };
  const node = { properties: { '3DA Pose Edits': structuredClone(saved) } };
  const states = [];
  const controller = poseHandles({ getSceneManager: () => ({ scene: sc }) }, { object, pose_rig: { kp_local: K, config: conf } }, node,
    undefined, (state, message) => states.push({ state, message }), { loadRig });
  await drain();
  return { mesh, controller, states, node };
}
{
  let loads = 0;
  const { mesh, controller, states } = await setupHandles(async () => { loads++; return rig; }, handEdit);
  assert.equal(loads, 1); assert.equal(states.at(-1).state, 'ready'); assert.equal(states.at(-1).message, 'Ручки позы готовы');
  assert.ok(mesh.geometry.userData.threedaSkin.mpfb, 'тело MPFB должно идти по настоящему ригу');
  const actual = mesh.geometry.attributes.position.array;
  let d = 0;
  for (let i = 0; i < actual.length; i++) d = Math.max(d, Math.abs(actual[i] - moved[i]));
  assert.ok(d < 1e-5, `poseHandles ↔ editedPositions: ${d}`);
  controller.reset();
  let r = 0;
  for (let i = 0; i < actual.length; i++) r = Math.max(r, Math.abs(actual[i] - scene.position[i]));
  assert.ok(r === 0, `сброс возвращает исходный меш (${r})`);
  controller.dispose();
}
{
  // файл рига не загрузился: окно работает на приблизительных весах и честно говорит об этом
  const { mesh, controller, states } = await setupHandles(async () => { throw new Error('HTTP 404'); }, handEdit);
  assert.equal(states.at(-1).state, 'ready');
  assert.match(states.at(-1).message, /приблизительные веса: не загрузился файл рига: HTTP 404/);
  assert.ok(mesh.geometry.userData.threedaSkin.rig);
  controller.dispose();
}
console.log('mpfb_rig_cpu: ok');
