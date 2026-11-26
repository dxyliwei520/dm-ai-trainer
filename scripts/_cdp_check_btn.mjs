// 用 CDP 打开页面，检查按钮状态 / JS 错误 / 点击是否有效
const BASE = 'http://127.0.0.1:9223';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getTargets() {
  const r = await fetch(BASE + '/json/list');
  return r.json();
}

async function main() {
  let targets = [];
  for (let i = 0; i < 20; i++) {
    try { targets = await getTargets(); if (targets.length) break; } catch (e) {}
    await sleep(500);
  }
  const page = targets.find((t) => t.type === 'page');
  if (!page) { console.log('NO PAGE TARGET'); process.exit(1); }

  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 0;
  const pending = new Map();
  const errors = [];
  ws.addEventListener('message', (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    if (m.method === 'Runtime.exceptionThrown') {
      errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
    }
    if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') {
      errors.push('[log] ' + m.params.entry.text);
    }
  });
  const send = (method, params = {}) => new Promise((res) => {
    const mid = ++id;
    pending.set(mid, res);
    ws.send(JSON.stringify({ id: mid, method, params }));
  });

  await new Promise((r) => ws.addEventListener('open', r));
  await send('Runtime.enable');
  await send('Log.enable');
  await send('Page.enable');
  await send('Page.navigate', { url: 'http://127.0.0.1:8198/?cdp=1' });
  await sleep(4000);

  const evalJS = async (expr) => {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    return r.result?.result?.value;
  };

  const info = await evalJS(`(() => {
    const b = document.getElementById('btnTrain');
    const s = document.getElementById('projects');
    return {
      title: document.title,
      hasBtn: !!b,
      btnText: b ? b.textContent.trim() : null,
      btnDisabled: b ? b.disabled : null,
      btnClass: b ? b.className : null,
      btnRect: b ? JSON.stringify(b.getBoundingClientRect()) : null,
      topElAtBtn: b ? (() => { const r = b.getBoundingClientRect();
          const el = document.elementFromPoint(r.left + r.width/2, r.top + r.height/2);
          return el ? (el.tagName + '#' + (el.id||'') + '.' + (el.className||'')) : null; })() : null,
      projectOptions: s ? s.options.length : null,
      currentProject: window.current || null,
      hasRefreshStatus: typeof refreshStatus,
      hasJpost: typeof jpost,
      build: (typeof BUILD !== 'undefined') ? BUILD : 'undefined',
      toastVisible: (() => { const t = document.getElementById('toast'); return t ? t.className : null; })(),
      modalVisible: (() => { const m = document.getElementById('loraModal'); return m ? m.className : null; })(),
    };
  })()`);

  console.log('=== PAGE INFO ===');
  console.log(JSON.stringify(info, null, 2));
  console.log('=== JS ERRORS ===');
  console.log(errors.length ? errors.join('\n---\n') : '(none)');

  process.exit(0);
}

main().catch((e) => { console.error('FAIL', e); process.exit(1); });
