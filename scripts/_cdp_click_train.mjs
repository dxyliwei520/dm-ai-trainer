// 看 toast 内容 + 真实点击「开始训练」，确认状态变 running
const BASE = 'http://127.0.0.1:9223';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const page = (await (await fetch(BASE + '/json/list')).json()).find((t) => t.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 0; const pending = new Map();
  ws.addEventListener('message', (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  });
  const send = (method, params = {}) => new Promise((res) => { const mid = ++id; pending.set(mid, res); ws.send(JSON.stringify({ id: mid, method, params })); });
  const evalJS = async (expr) => {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    if (r.result?.exceptionDetails) return { __err: r.result.exceptionDetails.text + ' ' + (r.result.exceptionDetails.exception?.description || '') };
    return r.result?.result?.value;
  };
  await new Promise((r) => ws.addEventListener('open', r));
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Page.navigate', { url: 'http://127.0.0.1:8198/?cdp=3' });
  await sleep(4500);

  console.log('toast 文案:', await evalJS(`document.getElementById('toast').textContent`));
  console.log('当前项目:', await evalJS(`document.getElementById('projects').value`));
  console.log('qcReport:', await evalJS(`document.getElementById('qcReport').textContent.trim().slice(0,160)`));

  console.log('--- 点击「开始训练」---');
  await evalJS(`(() => { const b = document.getElementById('btnTrain');
      b.dispatchEvent(new MouseEvent('click', {bubbles:true})); return 'clicked'; })()`);
  await sleep(6000);

  console.log('点击后按钮:', await evalJS(`(() => { const b=document.getElementById('btnTrain'); return b.textContent.trim()+' disabled='+b.disabled; })()`));
  console.log('toast:', await evalJS(`document.getElementById('toast').textContent`));
  const st = await (await fetch('http://127.0.0.1:8198/api/status')).json();
  console.log('后端状态:', JSON.stringify({ running: st.running, project: st.project, stage: st.last }));
  process.exit(0);
}
main().catch((e) => { console.error('FAIL', e); process.exit(1); });
