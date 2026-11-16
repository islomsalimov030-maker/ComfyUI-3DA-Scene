// Пакетный вход 3DA/auto: «Выбрать папку» (загрузка на сервер), «Запустить пакет» / «Стоп» (драйвер фаз)
// на узле «3DA · пакет: загрузка», «Скачать ZIP» на узле «3DA · пакет: отчёт», текст сводки на узлах.
// Драйвер ставит по задаче на (фаза, строка): все строки фазы 1, потом все строки фазы 2… —
// модели одной фазы остаются в памяти между строками.
const { app } = await import(new URL('scripts/app.js', document.baseURI).href);
const { api } = await import(new URL('scripts/api.js', document.baseURI).href);

const AUTO = ['3DAAutoBatch', '3DAAutoRow', '3DAAutoSave', '3DAAutoReport'];
const DRIVEN = ['3DAAutoRow', '3DAAutoSave'];
const KEEP = /\.(png|jpe?g|webp|bmp|xlsx|csv|txt)$/i;
const MAX_FILE = 50 * 1024 * 1024;
const runs = new Map();          // id узла «пакет: загрузка» → {ids, stop}

const widget = (node, name) => node.widgets?.find(w => w.name === name);
const classOf = node => node.comfyClass ?? node.type;

function subgraphList(graph) {
  const s = graph?.subgraphs;
  return s ? (typeof s.values === 'function' ? [...s.values()] : Object.values(s)) : [];
}
function allNodes() {
  return [...new Set([app.graph, ...subgraphList(app.graph)].flatMap(g => g?._nodes ?? g?.nodes ?? []))];
}

function stamp(d = new Date()) {
  const p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}`;
}
function batchName(folder) {
  const clean = folder.replace(/[^\p{L}\p{N}_\-. ()]+/gu, '_').replace(/^[. ]+/, '').slice(0, 100) || 'batch';
  return `${clean}_${stamp()}`;
}
function parsePhases(text) {
  const out = [];
  for (const part of String(text ?? '').split(/[,;\s]+/)) {
    const n = Number.parseInt(part, 10);
    if (Number.isInteger(n) && n > 0 && !out.includes(n)) out.push(n);
  }
  return out.length ? out : [1];
}

function button(node, label, callback) {
  const w = node.addWidget('button', label, null, callback, { serialize: false });
  w.serialize = false;
  w.baseLabel = label;
  return w;
}
function setLabel(node, w, text) {
  w.label = text ?? w.baseLabel;
  node.setDirtyCanvas?.(true, true);
}

// Текст на узле: свой read-only виджет (в workflow не сохраняется).
async function showText(node, text) {
  let w = node.__autoText;
  if (!w) {
    try {
      const { ComfyWidgets } = await import(new URL('scripts/widgets.js', document.baseURI).href);
      w = ComfyWidgets.STRING(node, 'auto_info', ['STRING', { multiline: true }], app).widget;
      if (w.inputEl) { w.inputEl.readOnly = true; w.inputEl.style.opacity = 0.85; }
    } catch (e) {
      w = node.addWidget('text', 'auto_info', '', () => {}, { serialize: false });
    }
    w.serialize = false;
    w.options = { ...(w.options ?? {}), serialize: false };
    node.__autoText = w;
  }
  w.value = text;
  if (w.inputEl) w.inputEl.value = text;
  node.setDirtyCanvas?.(true, true);
}

async function getJson(route, options) {
  const r = await api.fetchApi(route, options);
  let data = {};
  try { data = await r.json(); } catch { /* не JSON */ }
  if (!r.ok) throw new Error(data.error ?? data.errors?.join('; ') ?? `HTTP ${r.status}`);
  return data;
}

// ---------- «Выбрать папку» ----------
function pickFolder(node, w) {
  const input = document.createElement('input');
  input.type = 'file';
  input.webkitdirectory = true;
  input.multiple = true;
  input.style.display = 'none';
  input.onchange = () => {
    const files = [...input.files];
    input.remove();
    uploadFolder(node, w, files).catch(e => showText(node, `Загрузка не удалась: ${e.message}`));
  };
  document.body.appendChild(input);
  input.click();
}

async function uploadFolder(node, w, files) {
  if (!files.length) return;
  const folder = (files[0].webkitRelativePath || files[0].name).split('/')[0];
  const name = batchName(folder);
  const list = files
    .filter(f => KEEP.test(f.name) && !/^[.~]/.test(f.name))
    .map(f => ({ file: f, path: (f.webkitRelativePath || f.name).split('/').slice(1).join('/') || f.name }));
  const failed = [];
  let done = 0;
  const queue = list.filter(it => {
    if (it.file.size <= MAX_FILE) return true;
    failed.push(`${it.path}: больше 50 МБ`);
    return false;
  });
  const total = list.length;
  setLabel(node, w, `Загрузка 0/${total}…`);
  const worker = async () => {
    while (queue.length) {
      const it = queue.shift();
      const form = new FormData();
      form.append('batch', name);
      form.append('path', it.path);
      form.append('file', it.file, it.file.name);
      try {
        await getJson('/3da/auto/upload', { method: 'POST', body: form });
      } catch (e) {
        failed.push(`${it.path}: ${e.message}`);
      }
      done += 1;
      setLabel(node, w, `Загрузка ${done}/${total} (${Math.round(100 * done / Math.max(total, 1))}%)…`);
    }
  };
  await Promise.all([worker(), worker(), worker()]);
  setLabel(node, w);
  const bw = widget(node, 'batch');
  if (bw) { bw.value = name; bw.callback?.(name); }
  let text = `Пакет ${name}: загружено ${total - failed.length} из ${total} файлов (пропущено лишних: ${files.length - total})`;
  if (failed.length) text += `\nОшибки:\n${failed.slice(0, 15).join('\n')}`;
  try {
    const st = await getJson(`/3da/auto/state?batch=${encodeURIComponent(name)}`);
    const bad = st.rows.filter(r => r.status === 'error').length;
    text += `\nСтрок: ${st.rows_count}, с ошибками: ${bad}` + (st.errors?.length ? `\n${st.errors.join('\n')}` : '');
  } catch (e) {
    text += `\nСостояние не прочитано: ${e.message}`;
  }
  await showText(node, text);
}

// ---------- драйвер фаз: «Запустить пакет» / «Стоп» ----------
function setValue(node, name, value) {
  const w = widget(node, name);
  if (!w) return false;
  w.value = value;
  w.callback?.(value);
  return true;
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function runBatch(node, w) {
  if (runs.get(node.id)?.active) return showText(node, 'Пакет уже ставится в очередь — сначала «Стоп»');
  const batch = String(widget(node, 'batch')?.value ?? '').trim();
  if (!batch) return showText(node, 'Сначала «Выбрать папку» (или впишите имя пакета)');
  const phases = parsePhases(widget(node, 'phases')?.value);
  const st = await getJson(`/3da/auto/state?batch=${encodeURIComponent(batch)}`);
  const targets = allNodes().filter(n => DRIVEN.includes(classOf(n)));
  if (!targets.some(n => classOf(n) === '3DAAutoRow')) return showText(node, 'В воркфлоу нет узла «3DA · пакет: строка»');
  const jobs = [];
  for (const phase of phases) {
    for (let i = 0; i < st.rows_count; i++) {
      const row = st.rows[i];
      if (row?.status === 'error' || row?.phase_done?.[String(phase)]) continue;   // продолжить с места
      jobs.push({ phase, i, rid: row?.row_id });
    }
  }
  if (!jobs.length) return showText(node, `Нечего ставить: все строки прошли фазы ${phases.join(',')} или с ошибкой`);
  const run = { ids: [], stop: false, active: true, capture: false };
  runs.set(node.id, run);
  const original = api.queuePrompt;
  api.queuePrompt = async function (...args) {
    const res = await original.apply(this, args);
    if (run.capture && res?.prompt_id) run.ids.push(res.prompt_id);
    return res;
  };
  let queued = 0;
  try {
    for (const job of jobs) {
      if (run.stop) break;
      while (app.processingQueue) await sleep(50);     // свои значения row_index/phase — только в свою задачу
      for (const t of targets) { setValue(t, 'row_index', job.i); setValue(t, 'phase', job.phase); }
      run.capture = true;
      try { await app.queuePrompt(0, 1); } finally { run.capture = false; }
      queued += 1;
      setLabel(node, w, `В очереди ${queued}/${jobs.length} (фаза ${job.phase}, строка ${job.rid})`);
    }
  } finally {
    api.queuePrompt = original;
    run.active = false;
    setLabel(node, w);
  }
  await showText(node, `Поставлено задач: ${queued} из ${jobs.length} (фазы ${phases.join(',')}, строк ${st.rows_count})`
    + (run.stop ? '\nОстановлено кнопкой «Стоп»' : ''));
}

async function stopBatch(node) {
  const run = runs.get(node.id);
  if (!run) return showText(node, 'Этот узел ничего не ставил в очередь');
  run.stop = true;
  const q = await getJson('/queue');
  const pending = new Set((q.queue_pending ?? []).map(item => item[1]));
  const ids = run.ids.filter(id => pending.has(id));       // только свои и ещё не начатые
  if (ids.length) {
    await getJson('/queue', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ delete: ids }) });
  }
  run.ids = run.ids.filter(id => !ids.includes(id));
  await showText(node, `Стоп: снято из очереди ${ids.length} своих задач; текущая доработает`);
}

// ---------- «Скачать ZIP» ----------
function batchFor(reportNode) {
  const link = reportNode.inputs?.find(i => i.name === 'batch')?.link;
  if (link != null) {
    const graph = reportNode.graph ?? app.graph;
    const l = graph.links?.get?.(link) ?? graph.links?.[link];
    const src = graph.getNodeById?.(l?.origin_id);
    const v = src && classOf(src) === '3DAAutoBatch' ? widget(src, 'batch')?.value : null;
    if (v) return v;
  }
  const any = allNodes().filter(n => classOf(n) === '3DAAutoBatch').map(n => widget(n, 'batch')?.value).filter(Boolean);
  return any.length === 1 ? any[0] : null;
}
async function downloadZip(node, w) {
  const batch = batchFor(node);
  if (!batch) return showText(node, 'Не найден пакет: подключите вход batch к «3DA · пакет: загрузка»');
  setLabel(node, w, 'Собираю ZIP…');
  try {
    const r = await api.fetchApi(`/3da/auto/zip?batch=${encodeURIComponent(batch)}`);
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).error ?? msg; } catch { /* не JSON */ }
      throw new Error(msg);
    }
    const url = URL.createObjectURL(await r.blob());
    const a = document.createElement('a');
    a.href = url;
    a.download = `${batch}.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  } finally {
    setLabel(node, w);
  }
}

function onText(node, message) {
  const text = message?.text?.[0];
  if (typeof text === 'string') showText(node, text);
}

app.registerExtension({
  name: 'ThreeDA.AutoBatch',
  setup() {
    // Узлы внутри сабграфа не получают onExecuted — ловим executed по id «<сабграф>:<узел>».
    api.addEventListener('executed', ({ detail }) => {
      const id = String(detail?.display_node ?? detail?.node ?? '');
      if (!id.includes(':')) return;
      const local = id.split(':').pop();
      const node = allNodes().find(n => String(n.id) === local && AUTO.includes(classOf(n)));
      if (node) onText(node, detail?.output);
    });
  },
  nodeCreated(node) {
    const cls = classOf(node);
    if (!AUTO.includes(cls)) return;
    const original = node.onExecuted;
    node.onExecuted = function (message) {
      const r = original?.apply(this, arguments);
      onText(this, message);
      return r;
    };
    if (cls === '3DAAutoBatch') {
      const pick = button(node, 'Выбрать папку', () => pickFolder(node, pick));
      const run = button(node, 'Запустить пакет', () => runBatch(node, run).catch(e => { setLabel(node, run); showText(node, `Запуск не удался: ${e.message}`); }));
      button(node, 'Стоп', () => stopBatch(node).catch(e => showText(node, `Стоп не удался: ${e.message}`)));
    } else if (cls === '3DAAutoReport') {
      const zip = button(node, 'Скачать ZIP', () => downloadZip(node, zip).catch(e => showText(node, `ZIP не скачан: ${e.message}`)));
    }
  },
});
