// CPU-only integration test of the actual poseHandles module; minimal Three-compatible scene.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { poseHandles } from '../web/poseHandles.js';
import { sceneSelection } from '../web/sceneSelection.js';
import { buildWeights, solve, deform } from '../web/poseRig.js';

const { K, X, F, conf } = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const edits = { hand_r: [0, 0, .15], foot_l: [0, -.1, 0] };
const EPS = 4e-5;
function near(a, b, message) {
  assert.equal(a.length, b.length);
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  assert.ok(max < EPS, `${message}: max error ${max}`);
}
class V {
  constructor(x = 0, y = 0, z = 0) { this.set(x, y, z); }
  set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
  clone() { return new V(this.x, this.y, this.z); }
  copy(v) { return this.set(v.x, v.y, v.z); }
  applyMatrix4(m) {
    const e = m.elements, {x, y, z} = this;
    return this.set(e[0]*x+e[4]*y+e[8]*z+e[12], e[1]*x+e[5]*y+e[9]*z+e[13], e[2]*x+e[6]*y+e[10]*z+e[14]);
  }
  setScalar(s) { return this.set(s, s, s); }
}
class M {
  constructor(elements = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]) { this.elements = [...elements]; }
  clone() { return new M(this.elements); }
  multiply(b) {
    const a = this.elements, c = b.elements, r = Array(16).fill(0);
    for (let col = 0; col < 4; col++) for (let row = 0; row < 4; row++)
      for (let k = 0; k < 4; k++) r[col*4+row] += a[k*4+row]*c[col*4+k];
    this.elements = r; return this;
  }
  invert() { // Affine TRS, including uniform scale.
    const rows = Array.from({length: 4}, (_, r) => Array.from({length: 8}, (_, c) =>
      c < 4 ? this.elements[c*4+r] : Number(c-4 === r)));
    for (let c = 0; c < 4; c++) {
      const pivot = rows[c][c]; assert.ok(Math.abs(pivot) > 1e-12);
      rows[c] = rows[c].map(x => x / pivot);
      for (let r = 0; r < 4; r++) if (r !== c) {
        const factor = rows[r][c]; rows[r] = rows[r].map((x, j) => x - factor*rows[c][j]);
      }
    }
    this.elements = Array.from({length: 16}, (_, i) => rows[i%4][4+Math.floor(i/4)]);
    return this;
  }
}
function rigid(angle, x, y, z) {
  const c = Math.cos(angle), s = Math.sin(angle);
  return new M([c,s,0,0, -s,c,0,0, 0,0,1,0, x,y,z,1]);
}
function positions(src, matrix) {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 3) {
    const p = new V(src[i], src[i+1], src[i+2]).applyMatrix4(matrix);
    out.set([p.x, p.y, p.z], i);
  }
  return out;
}
class Attribute { constructor(array, itemSize) { this.array = array; this.itemSize = itemSize; } }
class Geometry {
  constructor() { this.attributes = {}; this.userData = {}; this.index = null; }
  setAttribute(name, value) { this.attributes[name] = value; }
  setIndex(value) { this.index = value; }
  clone() {
    const g = new Geometry();
    for (const [name, a] of Object.entries(this.attributes)) g.setAttribute(name, new Attribute(a.array.slice(), a.itemSize));
    if (this.index) g.setIndex(new Attribute(this.index.array.slice(), this.index.itemSize));
    g.userData = structuredClone(this.userData); return g;
  }
  computeVertexNormals() {} computeBoundingSphere() {} computeBoundingBox() {} dispose() {}
}
class Material { constructor() { this.color = {setHex() {}}; this.emissive = {setHex() {}}; } dispose() {} }
class Mesh {
  constructor(geometry, material = new Material()) {
    this.geometry = geometry; this.material = material; this.isMesh = true;
    this.position = new V(); this.scale = new V(1,1,1); this.userData = {}; this.matrixWorld = new M();
  }
}
function geometry() {
  const g = new Geometry();
  g.setAttribute('position', new Attribute(Float32Array.from(X), 3));
  g.setIndex(new Attribute(Uint32Array.from(F), 1));
  return g;
}
// Deterministic frame and idle queues verify that the preparation message is paintable.
let nextId = 1;
const frames = new Map(), idles = new Map();
globalThis.requestAnimationFrame = f => { const id = nextId++; frames.set(id, f); return id; };
globalThis.cancelAnimationFrame = id => frames.delete(id);
globalThis.requestIdleCallback = f => { const id = nextId++; idles.set(id, f); return id; };
globalThis.cancelIdleCallback = id => idles.delete(id);
function runQueue(queue) { const tasks = [...queue.values()]; queue.clear(); tasks.forEach(f => f()); }
function finishPreparation() { runQueue(frames); runQueue(frames); runQueue(idles); }
function setup(transforms, shared, saved = edits, config = conf, autoReady = true) {
  const bodyMatrix = rigid(-.37, 2, -.7, .4), original = geometry();
  const meshes = transforms.map((t, i) => {
    const mesh = new Mesh(shared || i === 0 ? original : original.clone());
    mesh.matrixWorld = bodyMatrix.clone().multiply(t); return mesh;
  });
  const object = {
    position: new V(), matrixWorld: bodyMatrix,
    traverse(fn) { meshes.forEach(fn); }, updateMatrixWorld() {},
    localToWorld(v) { return v.applyMatrix4(this.matrixWorld); },
    worldToLocal(v) { return v.applyMatrix4(this.matrixWorld.clone().invert()); },
    getWorldScale(v) { return v.set(1,1,1); },
  };
  const scene = {items: [], add(o) { this.items.push(o); }, remove(o) { this.items.splice(this.items.indexOf(o), 1); }};
  const node = {properties: {'3DA Pose Edits': structuredClone(saved)}};
  const states = [];
  const controller = poseHandles({getSceneManager: () => ({scene})}, {object, pose_rig: {kp_local: K, config}}, node,
    undefined, (state, message) => states.push({state, message}));
  assert.ok(controller); assert.equal(scene.items.length, 5);
  assert.deepEqual(states, [{state: 'preparing', message: 'Ручки позы: подготовка…'}]);
  assert.ok(scene.items.every(o => !o.visible));
  if (autoReady) {
    finishPreparation();
    assert.equal(states.at(-1).state, 'ready');
    assert.ok(scene.items.every(o => o.visible));
  }
  return {meshes, controller, node, scene, states, object};
}
function expected(transform) {
  // Reference executes the original LBS in body coordinates, independently of poseHandles' matrix chain.
  const bodyRest = positions(Float32Array.from(X), transform);
  const rig = buildWeights(bodyRest, Uint32Array.from(F), K, conf);
  const posed = deform(rig, solve(K, edits, conf).T, new Float32Array(X.length));
  return {bodyRest, bodyPosed: posed, localPosed: positions(posed, transform.clone().invert())};
}
const identity = new M(), shifted = rigid(.63, .23, -.14, .31), other = rigid(-.41, -.18, .12, -.2);
for (const [name, transform] of [['identity/MHR', identity], ['translation+rotation', shifted]]) {
  const {meshes, controller, node, scene} = setup([transform], false);
  const g = meshes[0].geometry, actual = g.attributes.position.array, ref = expected(transform);
  near(g.userData.threedaRest, X, `${name} local rest`);
  near(actual, ref.localPosed, `${name} mesh-local deformation`);
  near(positions(actual, transform), ref.bodyPosed, `${name} mesh→body→mesh`);
  assert.ok(actual.some((v, i) => Math.abs(v-X[i]) > 1e-4), `${name} must deform`);
  controller.reset(); near(actual, X, `${name} reset`);
  assert.deepEqual(node.properties['3DA Pose Edits'], {});
  controller.dispose(); assert.equal(scene.items.length, 0);
}
{
  // Three parts: two share geometry, the third has independent geometry.
  const transforms = [identity, shifted, other], {meshes, controller} = setup(transforms, true);
  assert.equal(new Set(meshes.map(m => m.geometry)).size, 3, 'shared geometry must be isolated');
  for (let i = 0; i < meshes.length; i++) {
    const actual = meshes[i].geometry.attributes.position.array, ref = expected(transforms[i]);
    near(actual, ref.localPosed, `shared part ${i}`);
    near(positions(actual, transforms[i]), ref.bodyPosed, `shared part ${i} body`);
  }
  controller.reset();
  for (const m of meshes) near(m.geometry.attributes.position.array, X, 'shared part reset');
  controller.dispose();
}
{
  const transforms = [shifted, other], {meshes, controller} = setup(transforms, false);
  assert.notEqual(meshes[0].geometry, meshes[1].geometry);
  for (let i = 0; i < meshes.length; i++) near(meshes[i].geometry.attributes.position.array, expected(transforms[i]).localPosed, `separate part ${i}`);
  controller.reset();
  for (const m of meshes) near(m.geometry.attributes.position.array, X, 'separate part reset');
  controller.dispose();
}
{
  const states = [];
  assert.equal(poseHandles({}, {pose_rig: null}, {properties: {}}, null, (state, message) => states.push({state, message})), null);
  assert.match(states[0].message, /скелет/);
  assert.equal(states[0].state, 'disabled');
  const object = {traverse() {}};
  assert.equal(poseHandles({getSceneManager: () => ({scene: {}})}, {object, pose_rig: {kp_local: K, config: conf}},
    {properties: {}}, null, (state, message) => states.push({state, message})), null);
  assert.match(states[1].message, /меша/);
}
{
  const broken = {...conf};
  Object.defineProperty(broken, 'failure', {enumerable: true, get() { throw new Error('тестовая ошибка'); }});
  const {controller, scene, states} = setup([identity], false, {}, broken, false);
  runQueue(frames);
  assert.equal(states.length, 1); assert.equal(idles.size, 0);
  runQueue(frames); assert.equal(states.length, 1); assert.equal(idles.size, 1);
  runQueue(idles);
  assert.equal(states.at(-1).state, 'disabled');
  assert.match(states.at(-1).message, /тестовая ошибка/);
  assert.ok(scene.items.every(o => !o.visible));
  controller.dispose();
}
{
  const {controller, states, scene, node, meshes} = setup([identity], false, edits, conf, false);
  controller.reset(); controller.moved('hand_r');
  assert.deepEqual(node.properties['3DA Pose Edits'], edits);
  assert.equal(meshes[0].geometry.userData.threedaRig, undefined);
  controller.setVisible(false); finishPreparation();
  assert.equal(states.at(-1).state, 'ready');
  assert.ok(scene.items.every(o => !o.visible));
  controller.setVisible(true); assert.ok(scene.items.every(o => o.visible));
  controller.dispose();
}
{
  const {controller, states, scene} = setup([identity], false, {}, conf, false);
  controller.dispose(); finishPreparation();
  assert.equal(states.length, 1); assert.equal(scene.items.length, 0);
}
{
  const {controller, states} = setup([identity], false, {}, conf, false);
  runQueue(frames); runQueue(frames); controller.dispose(); runQueue(idles);
  assert.equal(states.length, 1);
}
// Exercise the actual viewport status element, not just the controller callback.
class Element {
  constructor() { this.style = {}; this.children = []; this.dataset = {}; this.options = []; this.textContent = ''; this.listeners = new Map(); }
  setAttribute() {}
  addEventListener(event, listener) { if (!this.listeners.has(event)) this.listeners.set(event, []); this.listeners.get(event).push(listener); }
  removeEventListener(event, listener) { this.listeners.set(event, (this.listeners.get(event) || []).filter(f => f !== listener)); }
  dispatchEvent(event) { for (const f of this.listeners.get(event.type) || []) f(event); }
  appendChild(child) { this.children.push(child); child.parent = this; return child; }
  append(...children) { children.forEach(c => this.appendChild(c)); }
  add(child) { this.options.push(child); this.appendChild(child); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(c => c !== this); }
}
class OptionElement extends Element { constructor(text, value) { super(); this.text = text; this.value = value; } }
globalThis.Option = OptionElement;
globalThis.document = {createElement: () => new Element(), createElementNS: () => new Element()};
function viewport(config = conf, includeRig = true, savedEdits = {}, savedSelection = 'body') {
  const mesh = new Mesh(geometry());
  const object = {position: new V(), matrixWorld: new M(), traverse(fn) { fn(mesh); }, updateMatrixWorld() {},
    localToWorld(v) { return v; }, worldToLocal(v) { return v; }, getWorldScale(v) { return v.set(1,1,1); }};
  const root = {position: new V(), quaternion: {set() {}}, scale: new V(), updateMatrixWorld() {}, getObjectByName() { return object; }};
  const scene = {add() {}, remove() {}};
  const parent = new Element();
  const gizmo = {getMode: () => 'translate', setupForModel() {}};
  const view = {domElement: {parentElement: parent}, getModelManager: () => ({currentModel: root}),
    getSceneManager: () => ({scene}), getGizmoManager: () => gizmo, setGizmoEnabled() {}, setGizmoMode() {}, forceRender() {}};
  const body = {kind: 'body', id: 'body', name: 'body', label: 'body'};
  if (includeRig) body.pose_rig = {kp_local: K, config};
  object.quaternion = new V(); object.scale = new V(1,1,1);
  object.getWorldPosition = v => v.copy(object.position);
  object.getWorldQuaternion = v => v;
  const node = {properties: {'3DA Scene': {entities: [body]},
    '3DA Pose Edits': structuredClone(savedEdits), '3DA Selected Entity': savedSelection}};
  const controller = sceneSelection(view, node);
  const toolbar = parent.children.find(e => e.className === 'threeda-scene-toolbar');
  const status = toolbar.children.find(e => e.className === 'threeda-pose-status');
  const picker = toolbar.children.find(e => e.className === 'threeda-entity-selector');
  return {controller, status, picker, node};
}
{
  const {controller, status, picker} = viewport();
  assert.equal(status.textContent, 'Ручки позы: подготовка…');
  const group = picker.children.find(e => e.label === 'Ручки позы манекена');
  assert.equal(group.disabled, true);
  finishPreparation();
  assert.equal(status.style.display, 'none'); assert.equal(group.disabled, false);
  controller.dispose();
}
{
  const {controller, status} = viewport(conf, false);
  assert.match(status.textContent, /отключены:.*скелет/);
  controller.dispose();
}
{
  const broken = {...conf};
  Object.defineProperty(broken, 'failure', {enumerable: true, get() { throw new Error('весовой сбой'); }});
  const {controller, status, picker} = viewport(broken, true, edits);
  finishPreparation();
  assert.equal(controller.capture(), null);
  await assert.rejects(controller.waitForEdits(), /правки не применены/);
  assert.match(status.textContent, /отключены:.*весовой сбой/);
  assert.equal(picker.children.find(e => e.label === 'Ручки позы манекена').disabled, true);
  controller.dispose();
}
{
  const {controller, meshes, object} = setup([shifted], false);
  const originalRig = meshes[0].geometry.userData.threedaRig;
  const matrix = rigid(.81, 3.14, -2.34, .71);
  for (let i = 0; i < 12; i++) if (i%4 !== 3) matrix.elements[i] *= 1.37;
  object.matrixWorld = matrix;
  meshes[0].matrixWorld = matrix.clone().multiply(shifted);
  // Explicit sub-ulp-like noise guards the cache even if this fixture cancels exactly.
  meshes[0].matrixWorld.elements[12] += 3e-15;
  controller.reset('foot_l');
  assert.equal(meshes[0].geometry.userData.threedaRig, originalRig, 'body TRS must not rebuild weights');
  controller.dispose();
}
{
  const {controller, meshes, states} = setup([identity, shifted], false);
  meshes[1].geometry.computeVertexNormals = () => { throw new Error('second part failed'); };
  controller.sphere('hand_r').position.z += .05;
  controller.moved('hand_r'); runQueue(frames);
  assert.equal(states.at(-1).state, 'disabled');
  for (const m of meshes) near(m.geometry.attributes.position.array, X, 'partial failure rollback');
  assert.equal(controller.canCapture(), false);
  await assert.rejects(controller.waitForEdits(), /правки не применены/);
  controller.dispose();
}
{
  const {controller, picker, node} = viewport(conf, true, edits, 'h:hand_r');
  assert.equal(node.properties['3DA Selected Entity'], 'h:hand_r', 'temporary fallback preserves saved selection');
  assert.equal(controller.capture(), null, 'pending edits must not overwrite captured transforms');
  let waited = false;
  const waiting = controller.waitForEdits().then(() => { waited = true; });
  await Promise.resolve(); assert.equal(waited, false);
  picker.value = 'body'; picker.dispatchEvent({type: 'change'});
  assert.equal(node.properties['3DA Selected Entity'], 'body', 'real selection is persisted');
  finishPreparation(); await waiting;
  assert.equal(controller.selected(), 'body', 'real selection cancels saved-handle restoration');
  assert.ok(controller.capture()[0].pose_edits);
  controller.dispose();
}
{
  const {controller} = viewport(conf, true, edits, 'h:hand_r');
  finishPreparation();
  assert.equal(controller.selected(), 'h:hand_r', 'saved handle restored without user selection');
  controller.dispose();
}
{
  const {controller, node} = viewport(conf, true, edits, 'h:hand_r');
  controller.dispose();
  assert.equal(node.properties['3DA Selected Entity'], 'h:hand_r', 'closing during preparation preserves saved handle');
  await assert.rejects(controller.waitForEdits(), /правки не применены/);
}
{
  const {controller, node} = setup([identity], false, {});
  controller.sphere('hand_r').position.z += .05;
  controller.moved('hand_r');
  assert.equal(controller.canCapture(), false, 'pending drag must not expose old posed handles');
  assert.deepEqual(node.properties['3DA Pose Edits'], {});
  assert.equal(await controller.waitForEdits(), true);
  assert.equal(controller.canCapture(), true);
  assert.ok(node.properties['3DA Pose Edits'].hand_r);
  assert.ok(Math.hypot(...controller.worldDeltas().hand_r) > .001);
  assert.equal(frames.size, 0, 'serialization flushes and cancels pending drag frame');
  controller.dispose();
}
console.log('poseHandles CPU integration: transforms, viewport status, failures, readiness and cancellation OK');
