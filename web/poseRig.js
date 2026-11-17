// E10: ручки манекена в 3D-окне — IK рук и ног, изгиб корпуса, веса LBS.
// Та же математика, что tools/scene3d/posev2/pose_ik.py (сервер применяет правки перед постановкой);
// здесь — только предпросмотр. Чистый JS без three.js; точки — массивы [x, y, z] в локальных осях тела.
export const HANDLES = ['hand_r', 'hand_l', 'foot_r', 'foot_l', 'chest'];
export const HANDLE_LABELS = { hand_r: 'правая кисть', hand_l: 'левая кисть', foot_r: 'правая стопа',
  foot_l: 'левая стопа', chest: 'грудь' };
export const BONES = ['pelvis', 'spine1', 'spine2', 'chest', 'head',
  'upperarm_l', 'forearm_l', 'hand_l', 'upperarm_r', 'forearm_r', 'hand_r',
  'thigh_l', 'shin_l', 'foot_l', 'thigh_r', 'shin_r', 'foot_r'];
const SEGMENTS = {
  pelvis: [['hip_l', 'hip_r', [.25, .5, .75]]],
  spine1: [['j0', 'j1', [.5, .75]]],
  spine2: [['j1', 'j2', [.25, .5, .75]]],
  chest: [['j2', 'neck', [.25, .5, .75]], ['neck', 'sh_l', [.5]], ['neck', 'sh_r', [.5]]],
  head: [['neck', 'head', [.5, 1]]],
};
for (const s of ['l', 'r']) {
  SEGMENTS['upperarm_' + s] = [['sh_' + s, 'el_' + s, [.25, .5, .75]]];
  SEGMENTS['forearm_' + s] = [['el_' + s, 'wr_' + s, [.25, .5, .75]]];
  SEGMENTS['hand_' + s] = [['wr_' + s, 'hd_' + s, [.5, 1]]];
  SEGMENTS['thigh_' + s] = [['hip_' + s, 'kn_' + s, [.25, .5, .75]]];
  SEGMENTS['shin_' + s] = [['kn_' + s, 'an_' + s, [.25, .5, .75]]];
  SEGMENTS['foot_' + s] = [['an_' + s, 'toe_' + s, [.3, .6, .9]], ['an_' + s, 'heel_' + s, [.5]]];
}
const KP_BONE = { 5: 'upperarm_l', 6: 'upperarm_r', 7: 'forearm_l', 8: 'forearm_r', 9: 'thigh_l', 10: 'thigh_r',
  11: 'shin_l', 12: 'shin_r', 13: 'foot_l', 14: 'foot_r', 69: 'chest' };
for (let i = 0; i < 5; i++) KP_BONE[i] = 'head';
for (const i of [15, 16, 17]) KP_BONE[i] = 'foot_l';
for (const i of [18, 19, 20]) KP_BONE[i] = 'foot_r';
for (let i = 21; i < 42; i++) KP_BONE[i] = 'hand_r';
for (let i = 42; i < 63; i++) KP_BONE[i] = 'hand_l';

// ---------------------------------------------------------------------------------------------- векторы
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => Math.hypot(a[0], a[1], a[2]);
const unit = a => { const n = norm(a); return n > 1e-12 ? mul(a, 1 / n) : null; };
const lerp = (a, b, t) => add(a, mul(sub(b, a), t));
const mean = pts => mul(pts.reduce(add, [0, 0, 0]), 1 / pts.length);
// 3×3 — массив из 9 чисел по строкам.
const I3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const mv = (R, p) => [R[0] * p[0] + R[1] * p[1] + R[2] * p[2], R[3] * p[0] + R[4] * p[1] + R[5] * p[2],
  R[6] * p[0] + R[7] * p[1] + R[8] * p[2]];
const mm = (A, B) => { const C = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
  return C; };
const tr = A => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
const frame = (e, n) => { const c = cross(e, n); return [e[0], n[0], c[0], e[1], n[1], c[1], e[2], n[2], c[2]]; };
export const IDENTITY = { R: I3, t: [0, 0, 0] };
const compose = (outer, inner) => ({ R: mm(outer.R, inner.R), t: add(mv(outer.R, inner.t), outer.t) });
const about = (R, p) => ({ R, t: sub(p, mv(R, p)) });
export const applyT = (T, p) => add(mv(T.R, p), T.t);
function rotAxis([x, y, z], angle) {
  const c = Math.cos(angle), s = Math.sin(angle), C = 1 - c;
  return [c + x * x * C, x * y * C - z * s, x * z * C + y * s,
    y * x * C + z * s, c + y * y * C, y * z * C - x * s,
    z * x * C - y * s, z * y * C + x * s, c + z * z * C];
}

// ---------------------------------------------------------------------------------------------- скелет
export function joints(K) {
  const pelvis = mul(add(K[9], K[10]), .5), neck = K[69];
  return { j0: pelvis, j1: lerp(pelvis, neck, 1 / 3), j2: lerp(pelvis, neck, 2 / 3), neck,
    head: mul(add(K[3], K[4]), .5), chest: mul(add(K[5], K[6]), .5),
    sh_l: K[5], sh_r: K[6], el_l: K[7], el_r: K[8], wr_l: K[62], wr_r: K[41],
    hd_l: mean([49, 53, 57, 61].map(i => K[i])), hd_r: mean([28, 32, 36, 40].map(i => K[i])),
    hip_l: K[9], hip_r: K[10], kn_l: K[11], kn_r: K[12], an_l: K[13], an_r: K[14],
    toe_l: mul(add(K[15], K[16]), .5), toe_r: mul(add(K[18], K[19]), .5), heel_l: K[17], heel_r: K[20] };
}
export function handlePoints(K) {
  const J = joints(K);
  return { hand_l: J.wr_l, hand_r: J.wr_r, foot_l: J.an_l, foot_r: J.an_r, chest: J.chest };
}
function forward(J) {
  const up = unit(sub(J.neck, J.j0));
  if (!up) return [1, 0, 0];
  const left = add(sub(J.hip_l, J.hip_r), sub(J.sh_l, J.sh_r));
  let f = cross(left, up);
  f = unit(sub(f, mul(up, dot(f, up))));
  return f ?? [1, 0, 0];
}
// Анатомическое направление сустава (= pose_ik.pole_hint): колено — к носку (пятка→носок), локоть — назад и вниз.
function poleHint(J, s, knee) {
  const fwd = forward(J);
  if (knee) return unit(sub(J['toe_' + s], J['heel_' + s])) ?? fwd;
  const up = unit(sub(J.neck, J.j0)) ?? [0, 0, 0];
  return sub(mul(fwd, -1), mul(up, .5));
}
function segmentDistance(p, a, b) {
  const ab = sub(b, a), t = Math.min(1, Math.max(0, dot(sub(p, a), ab) / Math.max(dot(ab, ab), 1e-12)));
  return norm(sub(p, add(a, mul(ab, t))));
}
function nearestBone(p, J) {
  let best = Infinity, out = 0;
  BONES.forEach((bone, bi) => { for (const [a, b] of SEGMENTS[bone]) {
    const d = segmentDistance(p, J[a], J[b]);
    if (d < best) { best = d; out = bi; } } });
  return out;
}

// ---------------------------------------------------------------------------------------------- IK
// Аналитическая двухзвенная IK: длины костей неизменны, локоть/колено — только в сторону исходного полюса.
// Полюс — исходное смещение сустава; подсказка poleHint заменяет его у почти прямой конечности (угол > straightDeg)
// и, при hintSide (колено), если исходный полюс смотрит против подсказки (переразгибание).
export function twoBone(a, b, c, target, poleHint, eps, straightDeg = 180, hintSide = false) {
  const l1 = norm(sub(b, a)), l2 = norm(sub(c, b));
  const u = unit(sub(c, a)) ?? unit(sub(b, a));
  const source = sub(sub(b, a), mul(u, dot(sub(b, a), u)));
  const hint = unit(sub(poleHint, mul(u, dot(poleHint, u))));
  const bendSrc = Math.acos(Math.max(-1, Math.min(1, dot(sub(a, b), sub(c, b)) / Math.max(l1 * l2, 1e-18)))) * 180 / Math.PI;
  const useHint = !!hint && (norm(source) < 1e-6 * Math.max(l1, 1e-9) || bendSrc > straightDeg ||
    (hintSide && dot(source, hint) <= 0));
  const pole = (useHint ? hint : unit(source)) ??
    (Math.abs(u[0]) < .9 ? unit(cross(u, [1, 0, 0])) : unit(cross(u, [0, 1, 0])));
  const normal = unit(cross(u, pole));
  const reach = sub(target, a), dist = norm(reach);
  const dir = dist > 1e-9 ? mul(reach, 1 / dist) : u;
  const dmin = Math.abs(l1 - l2) + eps * (l1 + l2), dmax = (l1 + l2) * (1 - eps);
  const d = Math.min(Math.max(dist, dmin), dmax);
  const side = unit(sub(pole, mul(dir, dot(pole, dir)))) ?? unit(cross(normal, dir));
  const cosA = Math.min(1, Math.max(-1, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)));
  const sinA = Math.sqrt(Math.max(0, 1 - cosA * cosA));
  const b2 = add(a, mul(add(mul(dir, cosA), mul(side, sinA)), l1));
  const c2 = add(a, mul(dir, d));
  const normal2 = unit(cross(dir, side));
  const oldFrame = e => frame(e, unit(sub(normal, mul(e, dot(normal, e)))) ?? normal2);
  const R1 = mm(frame(mul(sub(b2, a), 1 / l1), normal2), tr(oldFrame(mul(sub(b, a), 1 / l1))));
  const R2 = mm(frame(mul(sub(c2, b2), 1 / l2), normal2), tr(oldFrame(mul(sub(c, b), 1 / l2))));
  return { b2, c2, R1, R2, info: { reached: dist >= dmin && dist <= dmax, straightened: dist > dmax,
    pole: useHint ? 'hint' : 'source', bend_src_deg: bendSrc,
    error_m: norm(sub(c2, target)), lengths_m: [l1, l2], bend_deg: Math.acos(Math.max(-1, Math.min(1,
      dot(sub(a, b2), sub(c2, b2)) / (l1 * l2)))) * 180 / Math.PI } };
}

// Изгиб корпуса поровну в 3 позвонках вокруг одной оси, таз на месте, угол не больше spine_max_deg.
export function spineBend(J, target, conf) {
  const P = J.j0, C = J.chest, v0 = sub(C, P), v1 = sub(target, P);
  let axis = cross(v0, v1);
  const s = norm(axis), limit = conf.spine_max_deg * Math.PI / 180;
  if (s < 1e-9) return { T: { spine1: IDENTITY, spine2: IDENTITY, chest: IDENTITY },
    info: { bend_deg: 0, limited: false, error_m: norm(sub(C, target)) } };
  axis = mul(axis, 1 / s);
  let want = Math.atan2(s, dot(v0, v1));
  const limited = want > limit;
  want = Math.min(want, limit);
  const chain = theta => {
    const R = rotAxis(axis, theta / 3);
    const t1 = about(R, J.j0), t2 = compose(t1, about(R, J.j1)), t3 = compose(t2, about(R, J.j2));
    return { spine1: t1, spine2: t2, chest: t3 };
  };
  const achieved = T => { const v = sub(applyT(T.chest, C), P); return Math.atan2(dot(cross(v0, v), axis), dot(v0, v)); };
  let theta = want;
  for (let i = 0; i < conf.spine_iters; i++) {
    const got = achieved(chain(theta));
    if (Math.abs(got) < 1e-12) break;
    theta = Math.min(theta * want / got, limit);
  }
  const T = chain(theta);
  return { T, info: { bend_deg: theta * 180 / Math.PI, limited, error_m: norm(sub(applyT(T.chest, C), target)) } };
}

// edits: {ручка: [dx,dy,dz]} в осях K. Возвращает {T: {кость: {R,t}}, info}.
export function solve(K, edits, conf) {
  const J = joints(K);
  const T = Object.fromEntries(BONES.map(b => [b, IDENTITY]));
  const info = {};
  if (edits.chest) {
    const r = spineBend(J, add(J.chest, edits.chest), conf);
    Object.assign(T, r.T);
    info.chest = r.info;
    for (const b of ['head', 'upperarm_l', 'forearm_l', 'hand_l', 'upperarm_r', 'forearm_r', 'hand_r']) T[b] = T.chest;
  }
  for (const s of ['l', 'r']) {
    for (const [handle, parent, names, bones, knee] of [
      ['hand_' + s, 'chest', ['sh', 'el', 'wr'], ['upperarm', 'forearm', 'hand'], false],
      ['foot_' + s, 'pelvis', ['hip', 'kn', 'an'], ['thigh', 'shin', 'foot'], true]]) {
      if (!edits[handle]) continue;
      const Tp = T[parent];
      const [a, b, c] = names.map(n => applyT(Tp, J[n + '_' + s]));
      const r = twoBone(a, b, c, add(J[names[2] + '_' + s], edits[handle]), mv(Tp.R, poleHint(J, s, knee)),
        conf.reach_eps, conf.straight_deg ?? 180, knee);
      T[bones[0] + '_' + s] = compose({ R: r.R1, t: sub(a, mv(r.R1, a)) }, Tp);
      T[bones[1] + '_' + s] = compose({ R: r.R2, t: sub(r.b2, mv(r.R2, b)) }, Tp);
      T[bones[2] + '_' + s] = compose({ R: I3, t: sub(r.c2, c) }, Tp);
      info[handle] = r.info;
    }
  }
  return { T, info };
}
export function kpBones(K) {
  const J = joints(K);
  return K.map((p, i) => KP_BONE[i] ?? BONES[nearestBone(p, J)]);
}
// Положение ручек после правок (неправленые ручки едут вместе со своей костью).
export function posedHandles(K, T) {
  const H = handlePoints(K);   // имя ручки = имя её кости
  return Object.fromEntries(HANDLES.map(h => [h, applyT(T[h], H[h])]));
}

// ---------------------------------------------------------------------------------------------- веса LBS
// positions — Float32Array вершин буфера (у GLB вершины на швах/гранях продублированы), index — индексы или null.
// Дубли по координатам сливаются, граф рёбер — по треугольникам; метка кости — Дейкстра от затравок на осях
// костей, затем диффузия меток по рёбрам (ширина blend_m), до 4 костей на вершину.
export function buildWeights(positions, index, K, conf) {
  const n = positions.length / 3;
  const ids = new Map(), uniq = new Int32Array(n), P = [];
  for (let i = 0; i < n; i++) {
    const key = positions[3 * i] + ',' + positions[3 * i + 1] + ',' + positions[3 * i + 2];
    let u = ids.get(key);
    if (u === undefined) { u = P.length; ids.set(key, u); P.push([positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]]); }
    uniq[i] = u;
  }
  const nU = P.length, nT = (index ? index.length : n) / 3;
  const tri = new Int32Array(nT * 3);
  for (let k = 0; k < nT * 3; k++) tri[k] = uniq[index ? index[k] : k];
  const edgeSet = new Set();
  for (let f = 0; f < nT; f++) for (let k = 0; k < 3; k++) {
    let a = tri[3 * f + k], b = tri[3 * f + (k + 1) % 3];
    if (a === b) continue;
    if (a > b) [a, b] = [b, a];
    edgeSet.add(a * nU + b);
  }
  const deg = new Int32Array(nU);
  const edges = [...edgeSet].map(e => { const a = Math.floor(e / nU), b = e - a * nU; deg[a]++; deg[b]++; return [a, b]; });
  const start = new Int32Array(nU + 1);
  for (let u = 0; u < nU; u++) start[u + 1] = start[u] + deg[u];
  const nbr = new Int32Array(start[nU]), len = new Float64Array(start[nU]), fill = start.slice(0, nU);
  let total = 0;
  for (const [a, b] of edges) {
    const l = norm(sub(P[a], P[b])) + 1e-12;
    total += l;
    nbr[fill[a]] = b; len[fill[a]++] = l; nbr[fill[b]] = a; len[fill[b]++] = l;
  }
  // нормали вершин (сумма нормалей граней с весом площади)
  const N = P.map(() => [0, 0, 0]);
  for (let f = 0; f < nT; f++) {
    const [a, b, c] = [tri[3 * f], tri[3 * f + 1], tri[3 * f + 2]];
    const fn = cross(sub(P[b], P[a]), sub(P[c], P[a]));
    for (const v of [a, b, c]) { N[v][0] += fn[0]; N[v][1] += fn[1]; N[v][2] += fn[2]; }
  }
  const J = joints(K);
  // затравки: ближайшая вершина «снаружи» оси среди seed_candidates ближайших
  const seedBone = new Map();
  const dist = new Float64Array(nU);
  BONES.forEach((bone, bi) => { for (const [a, b, ts] of SEGMENTS[bone]) for (const t of ts) {
    const s = lerp(J[a], J[b], t);
    let near = -1, out = -1;
    for (let v = 0; v < nU; v++) {
      const d = norm(sub(P[v], s));
      dist[v] = d;
      if (near < 0 || d < dist[near]) near = v;
      if (dot(N[v], sub(P[v], s)) > 0 && (out < 0 || d < dist[out])) out = v;
    }
    let v = near;
    if (out >= 0) {
      let closer = 0;
      for (let w = 0; w < nU; w++) if (dist[w] < dist[out]) closer++;
      if (closer < Math.min(conf.seed_candidates, nU)) v = out;
    }
    if (!seedBone.has(v)) seedBone.set(v, bi);
  } });
  // Дейкстра от всех затравок сразу
  const best = new Float64Array(nU).fill(Infinity), label = new Int32Array(nU).fill(-1);
  const heap = new MinHeap();
  for (const [v, bi] of seedBone) { best[v] = 0; label[v] = bi; heap.push(0, v); }
  while (heap.size) {
    const [d, v] = heap.pop();
    if (d > best[v]) continue;
    for (let k = start[v]; k < start[v + 1]; k++) {
      const w = nbr[k], nd = d + len[k];
      if (nd < best[w]) { best[w] = nd; label[w] = label[v]; heap.push(nd, w); }
    }
  }
  for (let v = 0; v < nU; v++) if (label[v] < 0) label[v] = nearestBone(P[v], J);
  // диффузия меток
  const nB = BONES.length;
  let W = new Float32Array(nU * nB), W2 = new Float32Array(nU * nB);
  for (let v = 0; v < nU; v++) W[v * nB + label[v]] = 1;
  const meanEdge = edges.length ? total / edges.length : 1;
  const iters = Math.min(conf.blend_iters_max, Math.max(conf.blend_iters_min,
    Math.round(2 * (conf.blend_m / Math.max(meanEdge, 1e-6)) ** 2)));
  for (let it = 0; it < iters; it++) {
    for (let v = 0; v < nU; v++) {
      const dv = Math.max(start[v + 1] - start[v], 1), o = v * nB;
      for (let b = 0; b < nB; b++) W2[o + b] = .5 * W[o + b];
      for (let k = start[v]; k < start[v + 1]; k++) {
        const ow = nbr[k] * nB;
        for (let b = 0; b < nB; b++) W2[o + b] += .5 * W[ow + b] / dv;
      }
    }
    [W, W2] = [W2, W];
  }
  const idx = new Uint8Array(nU * 4), w = new Float32Array(nU * 4);
  for (let v = 0; v < nU; v++) {
    const o = v * nB;
    const order = [...Array(nB).keys()].sort((a, b) => W[o + b] - W[o + a] || a - b).slice(0, 4);
    let sum = 0;
    order.forEach((b, k) => { const x = W[o + b] >= conf.weight_min ? W[o + b] : 0; idx[4 * v + k] = b; w[4 * v + k] = x; sum += x; });
    for (let k = 0; k < 4; k++) w[4 * v + k] /= Math.max(sum, 1e-12);
  }
  return { n, nU, uniq, P, idx, w, iters, label };
}

// LBS: positions (буфер) <- деформированные вершины; T — трансформации костей из solve().
export function deform(rig, T, out) {
  const Ts = BONES.map(b => T[b]);
  const moved = new Float64Array(rig.nU * 3);
  for (let v = 0; v < rig.nU; v++) {
    const p = rig.P[v];
    let x = 0, y = 0, z = 0;
    for (let k = 0; k < 4; k++) {
      const wk = rig.w[4 * v + k];
      if (!wk) continue;
      const { R, t } = Ts[rig.idx[4 * v + k]];
      x += wk * (R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + t[0]);
      y += wk * (R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + t[1]);
      z += wk * (R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2]);
    }
    moved[3 * v] = x; moved[3 * v + 1] = y; moved[3 * v + 2] = z;
  }
  for (let i = 0; i < rig.n; i++) {
    const u = rig.uniq[i];
    out[3 * i] = moved[3 * u]; out[3 * i + 1] = moved[3 * u + 1]; out[3 * i + 2] = moved[3 * u + 2];
  }
  return out;
}

class MinHeap {
  constructor() { this.k = []; this.v = []; }
  get size() { return this.k.length; }
  push(key, val) {
    const k = this.k, v = this.v;
    let i = k.length;
    k.push(key); v.push(val);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p] <= key) break;
      k[i] = k[p]; v[i] = v[p]; i = p;
    }
    k[i] = key; v[i] = val;
  }
  pop() {
    const k = this.k, v = this.v, top = [k[0], v[0]];
    const lk = k.pop(), lv = v.pop();
    if (k.length) {
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i, mk = lk;
        if (l < k.length && k[l] < mk) { m = l; mk = k[l]; }
        if (r < k.length && k[r] < mk) { m = r; mk = k[r]; }
        if (m === i) break;
        k[i] = k[m]; v[i] = v[m]; i = m;
      }
      k[i] = lk; v[i] = lv;
    }
    return top;
  }
}
