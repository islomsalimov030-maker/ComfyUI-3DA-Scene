// E10: 5 ручек манекена в 3D-окне (кисти, стопы, грудь). Выбрал ручку — обычный гизмо перемещения;
// кости руки/ноги/корпуса подстраиваются (IK), меш деформируется по весам LBS в реальном времени.
// Правки хранятся в node.properties['3DA Pose Edits'] (локальные оси тела) и уходят в body_transform
// ключом pose_edits (мировые оси) — сервер повторяет ту же IK перед постановкой.
// E10c: тело MPFB (108144 треугольника) двигается по настоящему ригу (mpfbRig.js: те же кости и веса, что у сервера);
// остальные тела и сбой привязки — приблизительные геодезические веса (poseRig.buildWeights), как раньше.
import { HANDLES, HANDLE_LABELS, handlePoints, solve, buildWeights, deform, posedHandles, kpBones, applyT } from './poseRig.js';
import { MPFB_TRIANGLES, MPFB_RIG_FILES, loadMpfbRig, bindMesh, editedPositions } from './mpfbRig.js';

const KEY = '3DA Pose Edits';
const RADIUS = 0.03;
const COLOR = 0xffc400, ACTIVE = 0xff3d00, EDITED = 0x00e5ff;

export function poseHandles(view, body, node, changed, report = () => {}, { loadRig = loadMpfbRig } = {}) {
  const rig = body.pose_rig;
  const K = rig?.kp_local, conf = rig?.config;
  if (!Array.isArray(K) || K.length !== 70 || !conf) {
    report('disabled', 'нет корректного скелета позы');
    return null;
  }
  const scene = view.getSceneManager?.()?.scene;
  const meshes = [];
  body.object.traverse(o => { if (o.isMesh && o.geometry?.attributes?.position) meshes.push(o); });
  if (!scene || !meshes.length) {
    report('disabled', !scene ? '3D-сцена недоступна' : 'у тела нет меша с вершинами');
    return null;
  }
  body.object.updateMatrixWorld(true);
  // Отдельные части GLB могут иметь собственные трансформации и общую геометрию.
  // Каждой части нужна своя копия: веса и деформированные вершины зависят от её матрицы.
  const seen = new Set();
  for (const m of meshes) {
    const geometry = m.geometry;
    if (seen.has(geometry)) m.geometry = geometry.clone();
    else seen.add(geometry);
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
    mesh.visible = false;
    mesh.raycast = () => {};
    mesh.userData.threedaHandle = h;
    scene.add(mesh);
    spheres[h] = mesh;
  }
  let active = null, posed = rest, frame = 0, ready = false, disposed = false, visible = true;
  let paintFrame = 0, prewarm = null, finishPreparation, mpfb = null, mpfbName = '', mpfbNote = '';
  const preparation = new Promise(resolve => { finishPreparation = resolve; });
  const pendingEditsError = () => new Error('Ручки позы: сохранение недоступно — правки не применены в 3D-окне. Исправьте ошибку подготовки или загрузите сцену заново.');
  const idle = globalThis.requestIdleCallback ?? (f => setTimeout(f, 0));
  const cancelIdle = globalThis.cancelIdleCallback ?? clearTimeout;
  function fail(e) {
    ready = false;
    // A later part may fail after an earlier one was deformed. Restore all parts.
    for (const m of meshes) {
      const g = m.geometry, rest = g.userData.threedaRest;
      if (!rest) continue;
      g.attributes.position.array.set(rest);
      g.attributes.position.needsUpdate = true;
      try { g.computeVertexNormals?.(); g.computeBoundingSphere?.(); g.computeBoundingBox?.(); } catch (_) { /* keep restored vertices */ }
    }
    posed = rest;
    for (const h of HANDLES) spheres[h].visible = false;
    finishPreparation();
    report('disabled', 'ошибка построения или применения весов: ' + (e?.message || String(e)));
    view.forceRender?.();
  }
  const limit = conf.max_offset_m * 0.99;
  const color = (h) => {
    const m = spheres[h].material;
    const c = h === active ? ACTIVE : offsets[h] ? EDITED : COLOR;
    m.color?.setHex?.(c); m.emissive?.setHex?.(c); m.emissiveIntensity = 0.6;
  };
  // Рест-позиции и веса — один раз на геометрию (переживают пересоздание 3D-окна).
  // skin: { mpfb: привязка к ригу MPFB } или { rig: геодезические веса }.
  // fallback = false: меш не сел на текущий риг — null без дорогих геодезических весов (нужно пробовать риг следующего персонажа).
  function rigOf(mesh, fallback = true) {
    body.object.updateMatrixWorld(true);
    const g = mesh.geometry, u = g.userData;
    const toBody = body.object.matrixWorld.clone().invert().multiply(mesh.matrixWorld);
    const key = JSON.stringify([K, conf, mpfb ? mpfbName || true : false]);
    // inverse(body.world) * mesh.world has round-off noise after moving the body.
    // Compare relative transforms with tolerance, not their JSON representation.
    const sameMatrix = u.threedaRigMatrix?.every((x, i) => Math.abs(x - toBody.elements[i]) <= 1e-9);
    if (!u.threedaRest) u.threedaRest = Float32Array.from(g.attributes.position.array);
    if (!u.threedaSkin || u.threedaRigKey !== key || !sameMatrix) {
      const t0 = performance.now();
      const bodyRest = transformPositions(u.threedaRest, toBody, vec([0, 0, 0]));
      const index = g.index?.array ?? null;
      let skin = null;
      if (mpfb && index && index.length / 3 === MPFB_TRIANGLES) {
        const bound = bindMesh(mpfb, bodyRest, index, K);
        if (bound.ok) {
          skin = { mpfb: bound };
          mpfbNote = '';      // меш мог не сесть на риг предыдущего персонажа — на этом сел
          console.info(`[3DA] риг MPFB: посадка на меш, медиана ${(bound.stats.median_m * 1e3).toFixed(2)} мм, ${Math.round(performance.now() - t0)} мс`);
        } else mpfbNote = bound.reason;
      }
      if (!skin) {
        if (!fallback) return null;
        skin = { rig: buildWeights(bodyRest, index, K, conf) };
        console.info(`[3DA] веса позы: ${skin.rig.nU} вершин, диффузия ${skin.rig.iters}, ${Math.round(performance.now() - t0)} мс`);
      }
      u.threedaSkin = skin;
      u.threedaRigKey = key;
      u.threedaRigMatrix = [...toBody.elements];
    }
    return { skin: u.threedaSkin, toMesh: toBody.invert() };
  }
  // Точки MHR70 после правок: те же IK-повороты костей, что у сервера (pose_ik.apply_edits).
  function editedKeypoints(T) {
    const bones = kpBones(K);
    return K.map((p, i) => applyT(T[bones[i]], p));
  }
  function apply() {
    const edits = Object.keys(offsets).length ? offsets : null;
    const { T } = solve(K, edits ?? {}, conf);
    let kpEdited = null;
    for (const m of meshes) {
      const g = m.geometry, pos = g.attributes.position;
      if (edits) {
        const { skin, toMesh } = rigOf(m);
        if (skin.mpfb) {
          kpEdited ??= editedKeypoints(T);
          skin.scratch = editedPositions(skin.mpfb, kpEdited, skin.scratch);
          transformPositions(skin.scratch, toMesh, vec([0, 0, 0]), pos.array);
        } else {
          deform(skin.rig, T, pos.array);
          transformPositions(pos.array, toMesh, vec([0, 0, 0]), pos.array);
        }
      } else if (g.userData.threedaRest) pos.array.set(g.userData.threedaRest);
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
    if (frame || !ready || disposed) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      try { apply(); view.forceRender?.(); changed?.(); } catch (e) { fail(e); }
    });
  }
  // Ручку двигает гизмо (или пульт): новый сдвиг = её место в осях тела минус исходное.
  function moved(h) {
    if (!ready || disposed) return;
    body.object.updateMatrixWorld(true);
    const p = body.object.worldToLocal(spheres[h].position.clone());
    const target = [p.x, p.y, p.z];
    // После предыдущей ручки мир уже деформирован: измеряем новый сдвиг относительно текущего
    // положения IK, не сбрасывая остальные ручки на исходную позу.
    const { T } = solve(K, offsets, conf);
    const current = posedHandles(K, T)[h];
    const d = [target[0] - current[0], target[1] - current[1], target[2] - current[2]];
    // Предел max_offset_m — для итогового сдвига ручки от исходной позы, а не для одного шага перетаскивания.
    const combined = clampLength((offsets[h] ?? [0, 0, 0]).map((x, i) => x + d[i]), limit);
    if (Math.hypot(...combined) > 1e-6) offsets[h] = combined; else delete offsets[h];
    schedule();
  }
  // Для body_transform: сдвиг ручки в мировых осях 3D-окна (с учётом поворота/масштаба тела).
  function worldDeltas() {
    body.object.updateMatrixWorld(true);
    if (disposed || ((!ready || frame) && Object.keys(offsets).length)) throw pendingEditsError();
    const out = {};
    const current = posed;
    for (const h of Object.keys(offsets)) {
      const a = body.object.localToWorld(vec(rest[h])), b = body.object.localToWorld(vec(current[h]));
      out[h] = clampLength([b.x - a.x, b.y - a.y, b.z - a.z], limit).map(x => Math.round(x * 1e6) / 1e6);
    }
    return out;
  }
  report('preparing', 'Ручки позы: подготовка…');
  // Веса строятся вне основного кадра; интерфейс явно показывает ожидание.
  paintFrame = requestAnimationFrame(() => {
    paintFrame = requestAnimationFrame(() => {
      paintFrame = 0;
      prewarm = idle(async () => {
        prewarm = null;
        try {
          // файл рига MPFB качаем только для тела с тем же числом треугольников; у каждого персонажа свой риг
          // (манекен подогнан под её фигуру): берём первый, на который сел меш окна
          const isMpfb = meshes.some(m => m.geometry.index?.array?.length / 3 === MPFB_TRIANGLES);
          const files = isMpfb ? MPFB_RIG_FILES : [null];
          for (let i = 0; i < files.length; i++) {
            if (files[i] !== null) {
              try { mpfb = await loadRig(files[i]); mpfbName = files[i]; } catch (e) {
                mpfb = null; mpfbNote = 'не загрузился файл рига: ' + (e?.message || String(e));
                for (const m of meshes) rigOf(m);
                break;
              }
              if (disposed) return;
            }
            const last = i === files.length - 1;
            if (meshes.every(m => rigOf(m, last))) break;
          }
          if (disposed) return;
          if (Object.keys(offsets).length) apply(); else sync();
          ready = true;
          for (const h of HANDLES) spheres[h].visible = visible;
          report('ready', mpfbNote ? `Ручки позы готовы (приблизительные веса: ${mpfbNote})` : 'Ручки позы готовы');
          finishPreparation();
          view.forceRender?.();
        } catch (e) { if (!disposed) fail(e); }
      }, { timeout: 2000 });
    });
  });
  return {
    handles: HANDLES, labels: HANDLE_LABELS,
    sphere: h => spheres[h],
    hasEdits: () => Object.keys(offsets).length > 0,
    canCapture: () => !frame && (!Object.keys(offsets).length || (ready && !disposed)),
    async waitForEdits() {
      if (!Object.keys(offsets).length && !frame) return;
      await preparation;
      if (!ready || disposed) throw pendingEditsError();
      if (frame) {
        cancelAnimationFrame(frame); frame = 0;
        try { apply(); view.forceRender?.(); changed?.(); } catch (e) { fail(e); throw pendingEditsError(); }
      }
      return true;
    },
    worldDeltas, moved, sync,
    highlight(h) { active = h; for (const k of HANDLES) color(k); },
    reset(h) {
      if (!ready || disposed) return;
      if (h) delete offsets[h]; else for (const k of HANDLES) delete offsets[k];
      try { apply(); view.forceRender?.(); changed?.(); } catch (e) { fail(e); }
    },
    setVisible(v) { visible = v; for (const h of HANDLES) spheres[h].visible = v && ready; },
    dispose() {
      disposed = true;
      finishPreparation();
      if (frame) cancelAnimationFrame(frame);
      if (paintFrame) cancelAnimationFrame(paintFrame);
      if (prewarm) cancelIdle(prewarm);
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

function transformPositions(src, matrix, target, out = new Float32Array(src.length)) {
  for (let i = 0; i < src.length; i += 3) {
    target.set(src[i], src[i + 1], src[i + 2]).applyMatrix4(matrix);
    out[i] = target.x; out[i + 1] = target.y; out[i + 2] = target.z;
  }
  return out;
}

function clampLength(v, limit) {
  const n = Math.hypot(...v);
  return n > limit ? v.map(x => x * limit / n) : v;
}
