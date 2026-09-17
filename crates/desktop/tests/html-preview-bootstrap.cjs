// Run with: node crates/desktop/tests/html-preview-bootstrap.cjs
// A suspended native WebView must report DOM changes without animation frames.
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { runInNewContext } = require('node:vm');

const events = {}, messages = [], microtasks = [];
let mutation, height = 198;
const body = {
  children: [],
  get scrollHeight() { return height; },
  getBoundingClientRect: () => ({ top: 0, bottom: height }),
};
runInNewContext(readFileSync(`${__dirname}/../src/html_preview/document.js`, 'utf8'), {
  parent: { postMessage: message => messages.push(message) },
  document: { body, fonts: { ready: { then() {} } } },
  addEventListener: (name, callback) => { events[name] = callback; },
  getComputedStyle: () => ({ marginBottom: '0' }),
  scrollY: 0,
  requestAnimationFrame() {}, // Suspended by the native browser.
  queueMicrotask: callback => microtasks.push(callback),
  ResizeObserver: class { observe() {} },
  MutationObserver: class {
    constructor(callback) { mutation = callback; }
    observe() {}
  },
});
const flush = () => { while (microtasks.length) microtasks.shift()(); };
events.DOMContentLoaded();
flush();
assert.equal(messages.at(-1)?.height, 198);
height = 233;
mutation();
mutation();
flush();
assert.equal(messages.at(-1)?.height, 233);
assert.equal(messages.length, 2, 'coalesce duplicate mutations');
height = 198;
mutation();
flush();
assert.equal(messages.at(-1)?.height, 198, 'collapsing content shrinks the row');
console.log('HTML preview bootstrap: suspended-frame expand/collapse passed');

// A reused native browser must identify the document that emitted each callback.
const ipc = [], hostEvents = {};
const element = () => ({
  style: {}, attributes: {}, disabled: false,
  classList: { values: new Set(), toggle(name, on) { on ? this.values.add(name) : this.values.delete(name); },
    add(name) { this.values.add(name); }, remove(name) { this.values.delete(name); } },
  setAttribute(name, value) { this.attributes[name] = value; }, focus() { this.focused = true; },
});
const frame = { ...element(), contentWindow: { postMessage() {} }, addEventListener() {} };
const elements = { iframe: frame };
for (const id of ['source', 'tools', 'download', 'expand', 'code']) elements['#'+id] = element();
const bodyElement = element();
const hostWindow = {
  ipc: { postMessage: message => ipc.push(JSON.parse(message)) },
  addEventListener: (name, callback) => { hostEvents[name] = callback; },
};
const host = readFileSync(`${__dirname}/../src/html_preview/host.html`, 'utf8')
  .split('<script>')[1].split('</script>')[0]
  .replaceAll('__TOKEN__', JSON.stringify('host-capability')).replaceAll('__GENERATION__', '42').replaceAll('__DARK__', 'false')
  .replace('__DOCUMENT__', JSON.stringify('<p>Current document</p>'))
  .replace('__SOURCE__', JSON.stringify('<p>Current document</p>'));
runInNewContext(host, {
  window: hostWindow,
  document: { querySelector: selector => elements[selector], documentElement: element(), body: bodyElement },
});
assert.deepEqual(ipc[0], { kind: 'ready', generation: 42, token: 'host-capability' });
hostEvents.message({ source: frame.contentWindow, data: { kind: 'height', height: 233, generation: 1 } });
assert.deepEqual(ipc[1], { kind: 'height', height: 233, generation: 42, token: 'host-capability' });
hostEvents.message({ source: {}, data: { kind: 'height', height: 999 } });
assert.equal(ipc.length, 2, 'ignore messages outside the sandboxed document');
console.log('HTML preview host: document generation routing passed');

const payload = frame.srcdoc;
hostWindow.previewLayout(600, 800, 0, 0, 120, 300);
assert.equal(elements['#tools'].style.top, '28px', 'clipping must not move the toolbar down the document');
assert.equal(elements['#tools'].style.right, 'calc(100% - 600px + 28px)');
hostWindow.previewLayout(600, 800, -40, -120);
assert.equal(elements['#tools'].style.top, '-92px', 'a translated document takes its toolbar offscreen with it');
assert.equal(elements['#tools'].style.right, 'calc(100% - 560px + 28px)', 'horizontal clipping must not reanchor the toolbar');
for (const action of ['source', 'expand', 'download']) {
  elements[action === 'source' ? '#code' : '#'+action].onclick();
  assert.deepEqual(ipc.at(-1), {kind:'action', action, generation:42, token:'host-capability'});
}
assert(bodyElement.classList.values.has('capturing'));
assert(elements['#download'].disabled);
hostWindow.previewCaptureFinished();
assert(!bodyElement.classList.values.has('capturing'));
assert(!elements['#download'].disabled);
hostWindow.previewMode(true, true);
assert(bodyElement.classList.values.has('source'));
assert.equal(elements['#source'].textContent, '<p>Current document</p>');
assert(elements['#source'].focused);
hostEvents.keydown({key:'Escape'});
assert.equal(ipc.at(-1).action, 'source', 'Escape closes source before the enlarged preview');
hostWindow.previewMode(true, false);
hostEvents.message({source:frame.contentWindow, data:{kind:'escape'}});
assert.equal(ipc.at(-1).action, 'dismiss');
assert.equal(frame.srcdoc, payload, 'source and enlarged modes retain the running iframe');
const count = ipc.length;
hostEvents.message({source:frame.contentWindow, data:{kind:'action', action:'download'}});
assert.equal(ipc.length, count, 'untrusted document cannot invoke native toolbar actions');
console.log('HTML preview toolbar: actions, capture, source, Escape and trust boundary passed');

hostWindow.previewLayout(600, 800, 0, 0);
assert.equal(elements['#tools'].style.top, '28px', 'toolbar sits inside the document margin');
assert.equal(elements['#tools'].style.right, 'calc(100% - 600px + 28px)');
