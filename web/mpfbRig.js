// E10c: настоящий риг манекена MPFB в 3D-окне — те же кости и веса, что у сервера (Blender).
// Сервер ставит манекен так: pose_tools.pose_mannequin_to_mhr = retarget_world (кости наводятся на точки MHR70) +
// скиннинг рига MPFB. Здесь — порт retarget_world и LBS по весам рига (файл mpfbRig.dat, выгрузка
// tools/scene3d/mpfb_rig_dump.py), чтобы правки ручек двигали те же вершины, что и в итоговом рендере.
// Чистый JS без three.js. Оси — как у сервера: мир Z вверх; матрицы 4×4 — массивы из 16 чисел по строкам.

// ---------------------------------------------------------------------------------------------- матрицы 4×4
const I4 = () => Float64Array.of(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
function mm4(a, b, out = new Float64Array(16)) {
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[4 * i + k] * b[4 * k + j];
    out[4 * i + j] = s;
  }
  return out;
}
function inv4(m) {   // аффинная: [A t; 0 1]
  const [a, b, c, d, e, f, g, h, i] = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const A = e * i - f * h, B = f * g - d * i, C = d * h - e * g;
  const det = a * A + b * B + c * C;
  const r = [A, c * h - b * i, b * f - c * e, B, a * i - c * g, c * d - a * f, C, b * g - a * h, a * e - b * d].map(x => x / det);
  const t = [m[3], m[7], m[11]];
  return Float64Array.of(r[0], r[1], r[2], -(r[0] * t[0] + r[1] * t[1] + r[2] * t[2]),
    r[3], r[4], r[5], -(r[3] * t[0] + r[4] * t[1] + r[5] * t[2]),
    r[6], r[7], r[8], -(r[6] * t[0] + r[7] * t[1] + r[8] * t[2]), 0, 0, 0, 1);
}
const from3 = (R, t = [0, 0, 0]) => Float64Array.of(R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2], 0, 0, 0, 1);
const translation = t => from3([1, 0, 0, 0, 1, 0, 0, 0, 1], t);
const point = (m, p) => [m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3], m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
  m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11]];
const head = (P, b) => [P[16 * b + 3], P[16 * b + 7], P[16 * b + 11]];
const tailOf = (P, b, length) => point(P.subarray(16 * b, 16 * b + 16), [0, length, 0]);

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => Math.hypot(a[0], a[1], a[2]);
const unit = a => mul(a, 1 / norm(a));
const mean = pts => mul(pts.reduce(add, [0, 0, 0]), 1 / pts.length);

// ---------------------------------------------------------------------------------------------- повороты
// Кватернион (w, x, y, z) -> матрица 3×3 по строкам.
function quatMatrix([w, x, y, z]) {
  return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
}
// Кратчайший поворот вектора v на d (как mathutils Vector.rotation_difference).
function rotationDifference(v, d) {
  const a = unit(v), b = unit(d), c = dot(a, b);
  let axis = cross(a, b);
  let s = norm(axis);
  if (s < 1e-9) {
    if (c > 0) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
    axis = Math.abs(a[0]) < .9 ? cross(a, [1, 0, 0]) : cross(a, [0, 1, 0]);
    s = norm(axis);
    axis = mul(axis, 1 / s);
    return quatMatrix([0, axis[0], axis[1], axis[2]]);
  }
  axis = mul(axis, 1 / s);
  const half = Math.atan2(s, c) / 2, sn = Math.sin(half);
  return quatMatrix([Math.cos(half), axis[0] * sn, axis[1] * sn, axis[2] * sn]);
}
// Jacobi для симметричной 4×4: собственный вектор наибольшего значения.
function topEigenvector(N) {
  const A = N.map(r => [...r]), V = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) off += A[i][j] * A[i][j];
    if (off < 1e-30) break;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 4; q++) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 4; k++) { const x = A[k][p], y = A[k][q]; A[k][p] = c * x - s * y; A[k][q] = s * x + c * y; }
      for (let k = 0; k < 4; k++) { const x = A[p][k], y = A[q][k]; A[p][k] = c * x - s * y; A[q][k] = s * x + c * y; }
      for (let k = 0; k < 4; k++) { const x = V[k][p], y = V[k][q]; V[k][p] = c * x - s * y; V[k][q] = s * x + c * y; }
    }
  }
  let best = 0;
  for (let i = 1; i < 4; i++) if (A[i][i] > A[best][best]) best = i;
  return [V[0][best], V[1][best], V[2][best], V[3][best]];
}
// Поворот cur -> tgt по методу наименьших квадратов (Хорн = Кабш без отражения); как pose_tools._kabsch.
// Возвращает кватернион (w≥0), матрицу из него даёт quatMatrix.
function kabschQuat(cur, tgt) {
  const ca = mean(cur), cb = mean(tgt);
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  cur.forEach((p, n) => { const a = sub(p, ca), b = sub(tgt[n], cb);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) S[i][j] += a[i] * b[j]; });
  const [xx, xy, xz] = S[0], [yx, yy, yz] = S[1], [zx, zy, zz] = S[2];
  const q = topEigenvector([
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, -xx - yy + zz]]);
  return q[0] < 0 ? q.map(x => -x) : q;
}
const kabsch = (cur, tgt) => quatMatrix(kabschQuat(cur, tgt));
// Половина поворота (slerp от единицы на 0.5 по кратчайшей дуге).
function halfRotation(q) {
  const h = [1 + q[0], q[1], q[2], q[3]], n = Math.hypot(...h);
  return quatMatrix(h.map(x => x / n));
}
export function rotationAbout(axis, deg) {
  const a = deg * Math.PI / 180, [x, y, z] = axis, c = Math.cos(a), s = Math.sin(a), C = 1 - c;
  return from3([c + x * x * C, x * y * C - z * s, x * z * C + y * s,
    y * x * C + z * s, c + y * y * C, y * z * C - x * s,
    z * x * C - y * s, z * y * C + x * s, c + z * z * C]);
}

// ---------------------------------------------------------------------------------------------- файл рига
// Номера точек MHR70 по именам (pose_tools.K) и блоки пальцев рук.
const K = { nose: 0, l_eye: 1, r_eye: 2, l_ear: 3, r_ear: 4, l_sh: 5, r_sh: 6, l_el: 7, r_el: 8, l_hip: 9, r_hip: 10,
  l_kn: 11, r_kn: 12, l_an: 13, r_an: 14, l_btoe: 15, l_stoe: 16, l_heel: 17, r_btoe: 18, r_stoe: 19, r_heel: 20,
  r_wr: 41, l_wr: 62, neck: 69 };
const FINGER_BASE = { R: 21, L: 42 };
const SCALE_LIMITS = [0.85, 1.15];

const DTYPES = { '<f8': Float64Array, '<f4': Float32Array, '<i4': Int32Array, '|u1': Uint8Array, '<u2': Uint16Array, '<u4': Uint32Array };

// bytes — ArrayBuffer/Uint8Array содержимого mpfbRig.dat (gzip или уже распакованное).
export async function parseRig(bytes) {
  let u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (u8[0] === 0x1f && u8[1] === 0x8b) {
    const stream = new Blob([u8]).stream().pipeThrough(new DecompressionStream('gzip'));
    u8 = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const hl = view.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(4, 4 + hl)));
  if (header.format !== '3da-mpfb-rig-1') throw new Error('неизвестный формат рига: ' + header.format);
  const base = 4 + hl, S = {};
  for (const [name, [offset, dtype, count]] of Object.entries(header.sections)) {
    const T = DTYPES[dtype];
    S[name] = new T(u8.slice(base + offset, base + offset + count * T.BYTES_PER_ELEMENT).buffer);
  }
  return buildRig(header, S);
}

function buildRig(header, S) {
  const nB = header.bones.length, index = new Map(header.bones.map((n, i) => [n, i]));
  const wgt = new Float32Array(S.wgt.length);
  for (let i = 0; i < wgt.length; i++) wgt[i] = S.wgt[i] / 65535;
  const rest = S.rest, restInv = new Float64Array(16 * nB), relRest = new Float64Array(16 * nB);
  for (let b = 0; b < nB; b++) {
    restInv.set(inv4(rest.subarray(16 * b, 16 * b + 16)), 16 * b);
    const p = S.parent[b];
    relRest.set(p < 0 ? rest.subarray(16 * b, 16 * b + 16)
      : mm4(restInv.subarray(16 * p, 16 * p + 16), rest.subarray(16 * b, 16 * b + 16)), 16 * b);
  }
  return { header, nB, index, parent: S.parent, rest, restInv, relRest, length: S.length, vArm: S.v_arm,
    idx: S.idx, wgt, top: header.top, tris: S.tris, nV: header.n_vertices, nT: header.n_triangles,
    rigWorld: Float64Array.from(header.rig_world.flat()) };
}

// ---------------------------------------------------------------------------------------------- поза рига
// Поза костей: matrix_basis каждой кости -> матрицы костей в осях рига (как pose bone .matrix).
function evalPose(rig, basis, P) {
  const tmp = new Float64Array(16);
  for (let b = 0; b < rig.nB; b++) {
    const local = mm4(rig.relRest.subarray(16 * b, 16 * b + 16), basis[b]), p = rig.parent[b];
    if (p < 0) P.set(local, 16 * b); else P.set(mm4(P.subarray(16 * p, 16 * p + 16), local, tmp), 16 * b);
  }
}

// Порт pose_tools.retarget_world + масштаб и сдвиг из pose_mannequin_to_mhr (без контакта с комнатой).
// kp — 70 точек MHR70 [x, y, z] в мировых осях Z вверх. extra — [[кость, ось 'X'|'Y'|'Z', градусы]] поверх позы
// (проверки; сервер так же крутит кости рига). Возвращает { W, P, scale, basis }: W — мировая матрица рига,
// P — матрицы костей (nB×16) в осях рига.
export function retarget(rig, kp, extra = []) {
  const nB = rig.nB, bi = n => rig.index.get(n);
  const has = n => rig.index.has(n);
  const basis = Array.from({ length: nB }, I4);
  const P = new Float64Array(16 * nB);
  evalPose(rig, basis, P);
  const mh = Object.fromEntries(Object.entries(K).map(([k, i]) => [k, kp[i]]));
  // масштаб рига: пропорции MHR к рестовому рига
  const W0 = Float64Array.from(rig.rigWorld);
  const joint = (Pm, W) => {
    const h = n => point(W, head(Pm, bi(n)));
    return { l_hip: h('upperleg01.L'), r_hip: h('upperleg01.R'), l_sh: h('upperarm01.L'), r_sh: h('upperarm01.R'),
      neck: h('neck01'), l_kn: h('lowerleg01.L'), r_kn: h('lowerleg01.R'), l_el: h('lowerarm01.L'), r_el: h('lowerarm01.R'),
      l_wr: h('wrist.L'), r_wr: h('wrist.R'), l_an: h('foot.L'), r_an: h('foot.R') };
  };
  const chain = (J, ank) => norm(sub(J.l_hip, J.l_kn)) + norm(sub(J.r_hip, J.r_kn)) +
    norm(sub(mul(add(J.l_hip, J.r_hip), .5), J.neck)) + ank;
  const Jr = joint(P, W0);
  const ankM = norm(sub(mh.l_kn, mh.l_an)) + norm(sub(mh.r_kn, mh.r_an));
  const ankR = norm(sub(Jr.l_kn, Jr.l_an)) + norm(sub(Jr.r_kn, Jr.r_an));
  const raw = chain(mh, ankM) / Math.max(chain(Jr, ankR), 1e-6);
  const scale = Math.min(Math.max(raw, SCALE_LIMITS[0]), SCALE_LIMITS[1]);
  const W = Float64Array.from(W0);
  for (const i of [0, 1, 2, 4, 5, 6, 8, 9, 10]) W[i] *= scale;
  // цели в осях рига
  const Wi = inv4(W);
  const T = Object.fromEntries(Object.entries(mh).map(([k, p]) => [k, point(Wi, p)]));
  T.mid_hip = mul(add(T.l_hip, T.r_hip), .5);
  T.mid_ear = mul(add(T.l_ear, T.r_ear), .5);
  for (const [side, base] of Object.entries(FINGER_BASE)) for (let f = 0; f < 5; f++) for (let j = 0; j < 4; j++)
    T[`${side}${f + 1}_${3 - j}`] = point(Wi, kp[base + 4 * f + j]);
  // операции над костями
  const H = n => head(P, bi(n)), tail = n => tailOf(P, bi(n), rig.length[bi(n)]);
  const turn = (name, R3) => {
    const b = bi(name), h = H(name), p = rig.parent[b];
    const M = mm4(mm4(translation(h), from3(R3)), mm4(translation(mul(h, -1)), P.subarray(16 * b, 16 * b + 16)));
    const parentTerm = p < 0 ? rig.relRest.subarray(16 * b, 16 * b + 16) : mm4(P.subarray(16 * p, 16 * p + 16), rig.relRest.subarray(16 * b, 16 * b + 16));
    basis[b] = mm4(inv4(parentTerm), M);
    evalPose(rig, basis, P);
  };
  const aim = (name, endPoint, d) => {
    if (!has(name) || norm(d) < 1e-5) return;
    const v = sub(endPoint(), H(name));
    if (norm(v) > 1e-5) turn(name, rotationDifference(v, d));
  };
  // корень: таз и направление позвоночника
  {
    const hipL = H('upperleg01.L'), hipR = H('upperleg01.R');
    const curX = unit(sub(hipL, hipR));
    let curZ = sub(H('neck01'), mul(add(hipL, hipR), .5));
    curZ = unit(sub(curZ, mul(curX, dot(curZ, curX))));
    const tgtX = unit(sub(T.l_hip, T.r_hip));
    let tgtZ = sub(T.neck, T.mid_hip);
    tgtZ = unit(sub(tgtZ, mul(tgtX, dot(tgtZ, tgtX))));
    const frame = (x, z) => { const y = cross(z, x); return [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]]; };   // столбцы x, y, z
    const Mc = frame(curX, curZ), Mt = frame(tgtX, tgtZ);
    const tr = A => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
    const m3 = (A, B) => { const C = new Array(9);
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
      return C; };
    turn('root', m3(Mt, tr(Mc)));
  }
  // торс: плечи и шея относительно таза (изгиб и скрутка), поровну между нижним и верхним позвонком
  const chestNow = () => [H('neck01'), H('upperarm01.L'), H('upperarm01.R'), H('spine05')];
  const goal = [T.neck, T.l_sh, T.r_sh, T.mid_hip];
  turn('spine05', halfRotation(kabschQuat(chestNow(), goal)));
  if (has('spine02')) turn('spine02', kabsch(chestNow(), goal));
  aim('neck01', () => H('head'), sub(T.mid_ear, T.neck));
  if (['head', 'eye.L', 'eye.R'].every(has))
    turn('head', kabsch([H('eye.L'), H('eye.R'), H('neck01')], [T.l_eye, T.r_eye, T.neck]));
  for (const [s, side] of [['L', 'l'], ['R', 'r']]) {
    if (has(`clavicle.${s}`)) {
      const base = add(T.neck, sub(H(`clavicle.${s}`), H('neck01')));
      aim(`clavicle.${s}`, () => H(`upperarm01.${s}`), sub(T[`${side}_sh`], base));
    }
    aim(`upperarm01.${s}`, () => H(`lowerarm01.${s}`), sub(T[`${side}_el`], T[`${side}_sh`]));
    aim(`lowerarm01.${s}`, () => H(`wrist.${s}`), sub(T[`${side}_wr`], T[`${side}_el`]));
    const hand = [2, 3, 5].map(f => `finger${f}-1.${s}`);
    if (hand.every(has))
      turn(`wrist.${s}`, kabsch([H(`wrist.${s}`), ...hand.map(H)], [T[`${side}_wr`], T[`${s}2_0`], T[`${s}3_0`], T[`${s}5_0`]]));
    for (let f = 1; f <= 5; f++) for (let j = 1; j <= 3; j++) {
      const bone = `finger${f}-${j}.${s}`;
      aim(bone, j < 3 ? () => H(`finger${f}-${j + 1}.${s}`) : () => tail(bone), sub(T[`${s}${f}_${j}`], T[`${s}${f}_${j - 1}`]));
    }
    aim(`upperleg01.${s}`, () => H(`lowerleg01.${s}`), sub(T[`${side}_kn`], T[`${side}_hip`]));
    aim(`lowerleg01.${s}`, () => H(`foot.${s}`), sub(T[`${side}_an`], T[`${side}_kn`]));
    if ([`foot.${s}`, `toe1-2.${s}`, `toe5-3.${s}`].every(has))
      turn(`foot.${s}`, kabsch([H(`foot.${s}`), tail(`toe1-2.${s}`), tail(`toe5-3.${s}`)], [T[`${side}_an`], T[`${side}_btoe`], T[`${side}_stoe`]]));
  }
  // сдвиг рига: суставы корпуса на суставы MHR
  const J = joint(P, W);
  const keys = ['l_hip', 'r_hip', 'l_sh', 'r_sh', 'neck'];
  const shift = mul(keys.reduce((a, k) => add(a, sub(mh[k], J[k])), [0, 0, 0]), 1 / keys.length);
  W[3] += shift[0]; W[7] += shift[1]; W[11] += shift[2];
  // дополнительные повороты костей (поверх позы, в локальных осях кости) — для проверок
  for (const [name, axis, deg] of extra) {
    const b = bi(name);
    basis[b] = mm4(basis[b], rotationAbout(axis === 'X' ? [1, 0, 0] : axis === 'Y' ? [0, 1, 0] : [0, 0, 1], deg));
    evalPose(rig, basis, P);
  }
  return { W, P, scale, basis };
}

// ---------------------------------------------------------------------------------------------- скиннинг
// Матрицы «ресты вершины рига -> мир»: D_b = W · P_b · rest_b⁻¹ (nB×16).
export function skinMatrices(rig, pose) {
  const D = new Float64Array(16 * rig.nB), tmp = new Float64Array(16);
  for (let b = 0; b < rig.nB; b++) {
    mm4(pose.P.subarray(16 * b, 16 * b + 16), rig.restInv.subarray(16 * b, 16 * b + 16), tmp);
    D.set(mm4(pose.W, tmp), 16 * b);
  }
  return D;
}
// LBS вершин рига: out (nV·3) — мировые позиции.
export function skinVertices(rig, D, out = new Float64Array(rig.nV * 3)) {
  const { vArm, idx, wgt, top } = rig;
  for (let v = 0; v < rig.nV; v++) {
    const x = vArm[3 * v], y = vArm[3 * v + 1], z = vArm[3 * v + 2];
    let ox = 0, oy = 0, oz = 0;
    for (let k = 0; k < top; k++) {
      const w = wgt[top * v + k];
      if (!w) continue;
      const o = 16 * idx[top * v + k];
      ox += w * (D[o] * x + D[o + 1] * y + D[o + 2] * z + D[o + 3]);
      oy += w * (D[o + 4] * x + D[o + 5] * y + D[o + 6] * z + D[o + 7]);
      oz += w * (D[o + 8] * x + D[o + 9] * y + D[o + 10] * z + D[o + 11]);
    }
    out[3 * v] = ox; out[3 * v + 1] = oy; out[3 * v + 2] = oz;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------- загрузка
export const MPFB_TRIANGLES = 108144;   // треугольников у тела MPFB (рендерный меш: маски + subsurf 1); быстрая проверка «это MPFB?»
let cached = null;
export function loadMpfbRig(url = new URL('./mpfbRig.dat', import.meta.url)) {
  cached ??= fetch(url).then(r => {
    if (!r.ok) throw new Error('mpfbRig.dat: HTTP ' + r.status);
    return r.arrayBuffer();
  }).then(parseRig).catch(e => { cached = null; throw e; });
  return cached;
}

// ---------------------------------------------------------------------------------------------- привязка меша GLB
// Оси окна: Y вверх (локальные оси тела three.js), оси сервера: Z вверх. y_окна = z, z_окна = -y.
const toZ = p => [p[0], -p[2], p[1]];
const toY = p => [p[0], p[2], -p[1]];

// Привязка меша тела окна к ригу. positions — вершины буфера в осях тела (Float32Array, Y вверх), index — индексный
// буфер, kp — 70 точек MHR70 исходной постановки в тех же осях (pose_rig.kp_local).
// Вершины GLB продублированы по граням и перемешаны экспортёром, но треугольники идут в порядке рендерного OBJ сервера
// (как в tris рига), поэтому: вершины GLB с равными координатами = одна вершина; вершина рига = та, что стоит на этих
// углах в большинстве треугольников (в нескольких квадах triangulate выбирает другую диагональ — зависит от позы).
// Не сошлось число треугольников / соответствие / посадка рига на меш — { ok: false, reason }: окно остаётся
// на приблизительных весах.
export function bindMesh(rig, positions, index, kp, limits = { median_m: 0.005, p99_m: 0.03, agree: 0.999 }) {
  const fail = (reason, extra = {}) => ({ ok: false, reason, ...extra });
  if (!index || index.length / 3 !== rig.nT) return fail(`у меша ${index ? index.length / 3 : 0} треугольников, у рига ${rig.nT}`);
  const nG = positions.length / 3, weld = new Int32Array(nG), seen = new Map(), first = [];
  for (let g = 0; g < nG; g++) {
    const key = positions[3 * g] + ',' + positions[3 * g + 1] + ',' + positions[3 * g + 2];
    let w = seen.get(key);
    if (w === undefined) { w = first.length; seen.set(key, w); first.push(g); }
    weld[g] = w;
  }
  if (first.length !== rig.nV) return fail(`у меша ${first.length} уникальных вершин, у рига ${rig.nV}`);
  // большинство голосов (Бойер-Мур) и проверка согласия
  const cand = new Int32Array(rig.nV).fill(-1), votes = new Int32Array(rig.nV);
  for (let c = 0; c < index.length; c++) {
    const r = rig.tris[c], w = weld[index[c]];
    if (votes[r] === 0) { cand[r] = w; votes[r] = 1; } else if (cand[r] === w) votes[r]++; else votes[r]--;
  }
  let agree = 0;
  for (let c = 0; c < index.length; c++) if (cand[rig.tris[c]] === weld[index[c]]) agree++;
  if (agree < index.length * limits.agree) return fail(`соответствие вершин меша и рига: ${(100 * agree / index.length).toFixed(2)}% углов`);
  const w2r = new Int32Array(rig.nV).fill(-1);
  for (let r = 0; r < rig.nV; r++) {
    if (cand[r] < 0 || w2r[cand[r]] >= 0) return fail('соответствие вершин меша и рига неоднозначно');
    w2r[cand[r]] = r;
  }
  const g2r = new Int32Array(nG), base = new Float64Array(rig.nV * 3), have = new Uint8Array(rig.nV).fill(1);
  for (let g = 0; g < nG; g++) g2r[g] = w2r[weld[g]];
  for (let w = 0; w < first.length; w++) base.set(positions.subarray(3 * first[w], 3 * first[w] + 3), 3 * w2r[w]);
  const pose = retarget(rig, kp.map(toZ));
  const fit = skinVertices(rig, skinMatrices(rig, pose));
  const rest = new Float64Array(rig.nV * 3), res = new Float64Array(rig.nV * 3), mean3 = [0, 0, 0];
  let n = 0;
  for (let r = 0; r < rig.nV; r++) {
    if (!have[r]) continue;
    const y = toY([fit[3 * r], fit[3 * r + 1], fit[3 * r + 2]]);
    rest.set(y, 3 * r);
    for (let k = 0; k < 3; k++) { res[3 * r + k] = base[3 * r + k] - y[k]; mean3[k] += res[3 * r + k]; }
    n++;
  }
  if (n < rig.nV * 0.999) return fail('часть вершин рига не покрыта мешем');
  for (let k = 0; k < 3; k++) mean3[k] /= n;
  const d = [];
  for (let r = 0; r < rig.nV; r++) if (have[r])
    d.push(Math.hypot(res[3 * r] - mean3[0], res[3 * r + 1] - mean3[1], res[3 * r + 2] - mean3[2]));
  d.sort((a, b) => a - b);
  const stats = { median_m: d[d.length >> 1], p99_m: d[Math.floor(.99 * (d.length - 1))], max_m: d[d.length - 1], offset_m: mean3 };
  if (stats.median_m > limits.median_m || stats.p99_m > limits.p99_m)
    return { ok: false, reason: `риг не сел на меш: медиана ${(stats.median_m * 1e3).toFixed(1)} мм, 99% ${(stats.p99_m * 1e3).toFixed(1)} мм`, stats };
  return { ok: true, rig, g2r, base, rest, nG, stats, kp };
}

// Положение вершин буфера после правок: kpEdited — точки MHR70 после IK (те же оси окна). Исходная форма меша
// сохраняется, добавляется только смещение, которое даёт риг (так остаются контактный сдвиг и поправки subsurf).
// extra — [[кость, ось, градусы]]: повороты костей рига поверх позы (проверки).
export function editedPositions(binding, kpEdited, out = new Float32Array(binding.nG * 3), extra = []) {
  const { rig, g2r, base, rest } = binding;
  const pose = retarget(rig, kpEdited.map(toZ), extra);
  const moved = skinVertices(rig, skinMatrices(rig, pose));
  const u = new Float64Array(rig.nV * 3);
  for (let r = 0; r < rig.nV; r++) {
    const y = toY([moved[3 * r], moved[3 * r + 1], moved[3 * r + 2]]);
    for (let k = 0; k < 3; k++) u[3 * r + k] = base[3 * r + k] + y[k] - rest[3 * r + k];
  }
  for (let g = 0; g < binding.nG; g++) {
    const r = g2r[g];
    if (r < 0) continue;
    out[3 * g] = u[3 * r]; out[3 * g + 1] = u[3 * r + 1]; out[3 * g + 2] = u[3 * r + 2];
  }
  return out;
}
