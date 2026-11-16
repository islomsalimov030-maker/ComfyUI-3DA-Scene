// E10: 5 ручек манекена в 3D-окне (кисти, стопы, грудь). Выбрал ручку — обычный гизмо перемещения;
// кости руки/ноги/корпуса подстраиваются (IK), меш деформируется по весам LBS в реальном времени.
// Правки хранятся в node.properties['3DA Pose Edits'] (локальные оси тела) и уходят в body_transform
// ключом pose_edits (мировые оси) — сервер повторяет ту же IK перед постановкой.
import { HANDLES, HANDLE_LABELS, handlePoints, solve, buildWeights, deform, posedHandles } from './poseRig.js';

const KEY = '3DA Pose Edits';
const RADIUS = 0.03;
const COLOR = 0xffc400, ACTIVE = 0xff3d00, EDITED = 0x00e5ff;

export function poseHandles(view, body, node, changed) {
  const rig = body.pose_rig;
  const K = rig?.kp_local, conf = rig?.config;
  if (!Array.isArray(K) || K.length !== 70 || !conf) return null;
  const scene = view.getSceneManager?.()?.scene;
  const meshes = [];
  body.object.traverse(o => { if (o.isMesh && o.geometry?.attributes?.position) meshes.push(o); });
  if (!scene || !meshes.length) return null;
  body.object.updateMatrixWorld(true);
  // Точки скелета — в локальных осях тела; меши GLB тела лежат в них же (узел тела или его дети без сдвига).
  const inv = body.object.matrixWorld.clone().invert();
  for (const m of meshes) {
    const e = inv.clone().multiply(m.matrixWorld).elements;
    if (e.some((x, i) => Math.abs(x - (i % 5 === 0 ? 1 : 0)) > 1e-6)) {
      console.warn('3DA: меш тела со своим сдвигом — ручки позы отключены');
      return null;
    }
  }
  const rest = handlePoints(K);
  const vec = (p) => body.object.position.clone().set(p[0], p[1], p[2]);
  const offsets = {};
  const saved = node.properties[KEY];
  if (saved && typeof saved === 'object')
    for (const h of HANDLES) {
      const v = saved[h];
      if (Array.isArray(v) && v.length === 3 && v.every(Number.isFinite) && v.some(x => x !== 0)) offsets[h] = [...v];
    }
  // Сферы-ручки: геометрия и материал — классами уже загруженного меша (второй копии three.js нет).
  const sample = meshes[0];
  const Geometry = sample.geometry.constructor, Attribute = sample.geometry.attributes.position.constructor;
  const Material = Array.isArray(sample.material) ? sample.material[0].constructor : sample.material.constructor;
  const spheres = {};
  for (const h of HANDLES) {
    const { positions, indices } = sphereMesh(RADIUS, 12, 8);
    const geometry = new Geometry();
    geometry.setAttribute('position', new Attribute(positions, 3));
    geometry.setIndex(new Attribute(indices, 1));
    geometry.computeVertexNormals?.();
    const material = new Material();
    material.depthTest = false; material.depthWrite = false; material.transparent = true; material.opacity = 0.95;
    const mesh = new sample.constructor(geometry, material);
    mesh.name = '3DA_handle_' + h;
    mesh.renderOrder = 999;
    mesh.frustumCulled = false;
    mesh.raycast = () => {};
    mesh.userData.threedaHandle = h;
    scene.add(mesh);
    spheres[h] = mesh;
  }
  let active = null, posed = rest, frame = 0;
  const color = (h) => {
    const m = spheres[h].material;
    const c = h === active ? ACTIVE : offsets[h] ? EDITED : COLOR;
    m.color?.setHex?.(c); m.emissive?.setHex?.(c); m.emissiveIntensity = 0.6;
  };
  // Рест-позиции и веса — один раз на геометрию (переживают пересоздание 3D-окна).
  const key = JSON.stringify(K);
  function rigOf(mesh) {
    const g = mesh.geometry, u = g.userData;
    if (!u.threedaRest) u.threedaRest = Float32Array.from(g.attributes.position.array);
    if (!u.threedaRig || u.threedaRigKey !== key) {
      const t0 = performance.now();
      u.threedaRig = buildWeights(u.threedaRest, g.index?.array ?? null, K, conf);
      u.threedaRigKey = key;
      console.info(`[3DA] веса позы: ${u.threedaRig.nU} вершин, диффузия ${u.threedaRig.iters}, ${Math.round(performance.now() - t0)} мс`);
    }
    return u.threedaRig;
  }
  function apply() {
    const edits = Object.keys(offsets).length ? offsets : null;
    const { T } = solve(K, edits ?? {}, conf);
    for (const m of meshes) {
      const g = m.geometry, pos = g.attributes.position;
      if (edits) deform(rigOf(m), T, pos.array);
      else if (g.userData.threedaRest) pos.array.set(g.userData.threedaRest);
      else continue;
      pos.needsUpdate = true;
      g.computeVertexNormals?.();
      g.computeBoundingSphere?.();
      g.computeBoundingBox?.();
    }
    posed = posedHandles(K, T);
    node.properties[KEY] = JSON.parse(JSON.stringify(offsets));
    sync();
  }
  function local(h) { return posed[h]; }
  function sync() {
    body.object.updateMatrixWorld(true);
    for (const h of HANDLES) {
      const p = body.object.localToWorld(vec(local(h)));
      spheres[h].position.copy(p);
      spheres[h].scale.setScalar(body.object.getWorldScale(vec([1, 1, 1])).x);
      color(h);
    }
  }
  function schedule() {
    if (frame) return;
    frame = requestAnimationFrame(() => { frame = 0; apply(); view.forceRender?.(); changed?.(); });
  }
  // Ручку двигает гизмо (или пульт): новый сдвиг = её место в осях тела минус исходное.
  function moved(h) {
    body.object.updateMatrixWorld(true);
    const p = body.object.worldToLocal(spheres[h].position.clone());
    const target = [p.x, p.y, p.z];
    // После предыдущей ручки мир уже деформирован: измеряем новый сдвиг относительно текущего
    // положения IK, не сбрасывая остальные ручки на исходную позу.
    const { T } = solve(K, offsets, conf);
    const current = posedHandles(K, T)[h];
    let d = [target[0] - current[0], target[1] - current[1], target[2] - current[2]];
    const n = Math.hypot(...d), limit = conf.max_offset_m * 0.99;
    if (n > limit) d = d.map(x => x * limit / n);
    const combined = (offsets[h] ?? [0, 0, 0]).map((x, i) => x + d[i]);
    if (Math.hypot(...combined) > 1e-6) offsets[h] = combined; else delete offsets[h];
    schedule();
  }
  // Для body_transform: сдвиг ручки в мировых осях 3D-окна (с учётом поворота/масштаба тела).
  function worldDeltas() {
    body.object.updateMatrixWorld(true);
    const out = {};
    for (const h of Object.keys(offsets)) {
      const a = body.object.localToWorld(vec(rest[h])), b = body.object.localToWorld(vec(local(h)));
      out[h] = [b.x - a.x, b.y - a.y, b.z - a.z].map(x => Math.round(x * 1e6) / 1e6);
    }
    return out;
  }
  if (Object.keys(offsets).length) apply(); else sync();
  return {
    handles: HANDLES, labels: HANDLE_LABELS,
    sphere: h => spheres[h],
    hasEdits: () => Object.keys(offsets).length > 0,
    worldDeltas, moved, sync,
    highlight(h) { active = h; for (const k of HANDLES) color(k); },
    reset(h) { if (h) delete offsets[h]; else for (const k of HANDLES) delete offsets[k]; apply(); view.forceRender?.(); changed?.(); },
    setVisible(v) { for (const h of HANDLES) spheres[h].visible = v; },
    dispose() {
      if (frame) cancelAnimationFrame(frame);
      for (const h of HANDLES) { scene.remove(spheres[h]); spheres[h].geometry.dispose(); spheres[h].material.dispose(); }
      view.forceRender?.();
    } };
}

function sphereMesh(r, seg, rings) {
  const pos = [], idx = [];
  for (let j = 0; j <= rings; j++) {
    const v = j / rings * Math.PI;
    for (let i = 0; i <= seg; i++) {
      const u = i / seg * 2 * Math.PI;
      pos.push(r * Math.sin(v) * Math.cos(u), r * Math.cos(v), r * Math.sin(v) * Math.sin(u));
    }
  }
  for (let j = 0; j < rings; j++) for (let i = 0; i < seg; i++) {
    const a = j * (seg + 1) + i, b = a + seg + 1;
    idx.push(a, b, a + 1, b, b + 1, a + 1);
  }
  return { positions: new Float32Array(pos), indices: new Uint16Array(idx) };
}
