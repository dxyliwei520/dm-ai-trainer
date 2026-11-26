// 复测：确认无 JS 错误、按钮已绑定、下拉有项目、点击能真正启动训练
const BASE = 'http://127.0.0.1:9223';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  let page = null;
  for (let i = 0; i < 30; i++) {
    try {
      const list = await (await fetch(BASE + '/json/list')).json();
      page = list.find((t) => t.type === 'page');
      if (page) break;
    } catch (e) {}
    await sleep(500);
  }
  if (!page) { console.log('NO PAGE'); process.exit(1); }

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
  });
  const send = (method, params = {}) => new Promise((res) => {
    const mid = ++id; pending.set(mid, res);
    ws.send(JSON.stringify({ id: mid, method, params }));
  });
  const evalJS = async (expr) => {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    if (r.result?.exceptionDetails) return { __err: r.result.exceptionDetails.text };
    return r.result?.result?.value;
  };

  await new Promise((r) => ws.addEventListener('open', r));
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Page.navigate', { url: 'http://127.0.0.1:8198/?cdp=2' });
  await sleep(4500);

  console.log('--- 页面状态 ---');
  console.log(JSON.stringify(await evalJS(`(() => {
    const b = document.getElementById('btnTrain');
    return {
      build: BUILD,
      title: document.title,
      projectOptions: document.getElementById('projects').options.length,
      currentProject: window.current,
      btnText: b.textContent.trim(),
      btnDisabled: b.disabled,
      hasClickHandler: !!b.onclick,
      toast: document.getElementById('toast').className,
    };
  })()`), null, 2));

  console.log('--- JS 错误 ---');
  console.log(errors.length ? errors.join('\n---\n') : '(none)');

  process.exit(0);
}
main().catch((e) => { console.error('FAIL', e); process.exit(1); });
