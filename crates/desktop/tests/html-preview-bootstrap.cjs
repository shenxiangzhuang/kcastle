// Run with: node crates/desktop/tests/html-preview-bootstrap.cjs
// A suspended native WebView must report DOM changes without animation frames.
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { runInNewContext } = require('node:vm');

const events = {}, messages = [], microtasks = [];
let mutation, height = 198;
const parentWindow = { postMessage: message => messages.push(message) };
const style = () => ({ setProperty(name, value) { this[name] = value; } });
class Element {
  constructor(properties = {}) {
    Object.assign(this, {style: style(), parentElement: null, scrollTop: 0, scrollLeft: 0,
      clientHeight: 100, scrollHeight: 100, clientWidth: 100, scrollWidth: 100}, properties);
  }
  scrollBy({top, left, behavior}) {
    assert.equal(behavior, 'instant');
    this.scrollTop = Math.max(0, Math.min(this.scrollHeight - this.clientHeight, this.scrollTop + top));
    this.scrollLeft = Math.max(0, Math.min(this.scrollWidth - this.clientWidth, this.scrollLeft + left));
  }
}
const root = new Element();
const body = {
  style: style(),
  children: [],
  get scrollHeight() { return height; },
  getBoundingClientRect: () => ({ top: 0, bottom: height }),
};
runInNewContext(readFileSync(`${__dirname}/../src/html_preview/document.js`, 'utf8'), {
  parent: parentWindow,
  document: { body, elementFromPoint: () => root, documentElement: root, scrollingElement: root, fonts: { ready: { then() {} } } },
  Element,
  addEventListener: (name, callback) => { events[name] = callback; },
  getComputedStyle: node => ({ marginBottom: '0', overflowX: 'visible', overflowY: node.style['overflow-y'] || 'visible', ...node.style }),
  innerHeight: 240,
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
assert.equal(root.style['overflow-y'], 'auto', 'bounded inline previews scroll internally');
assert.equal(root.style['overscroll-behavior'], 'none');
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
const measurements = messages.length;
events.message({source:parentWindow, data:{kind:'expanded', value:false}});
flush();
assert.equal(messages.length, measurements + 1, 'returning from the sidebar republishes the inline height');
console.log('HTML preview bootstrap: suspended-frame expand/collapse passed');

// Reproduce the native trace: no DOM wheel event arrives at all.
const nativeBefore = messages.length;
events.message({source:parentWindow, data:{kind:'nativeWheel', x:12, y:34, dx:0, dy:40}});
assert.equal(messages.length, nativeBefore + 1, 'native input must work without a DOM wheel event');
assert.equal(messages.at(-1).kind, 'wheel');
assert.equal(messages.at(-1).dy, 40);

// A horizontal table must not consume a vertical gesture (including trackpad drift).
const table = new Element({style: {overflowX:'auto'}, scrollWidth:400});
const wheel = (target, dx, dy, options = {}) => {
  const before = messages.length;
  const event = {target, deltaX:dx, deltaY:dy, deltaMode:0, clientX:12, clientY:34,
    preventDefault() { this.prevented = true; }, ...options};
  events.wheel(event);
  return {forwarded: messages.length > before, prevented: !!event.prevented};
};
const shortContent = new Element({parentElement:root});
for (const dy of [-40, 40]) {
  assert(wheel(shortContent, 0, dy).forwarded, 'short content immediately hands off in both directions');
}
// CSS/geometry can suggest overflow without creating a movable scroll container.
const immovable = new Element({parentElement:shortContent, style:{overflowY:'auto'}, scrollHeight:400,
  scrollBy() {}});
assert(wheel(immovable, 0, 40).forwarded, 'an attempted scroll that did not move must not swallow the wheel');
assert.deepEqual(wheel(table, 0, 40), {forwarded:true, prevented:true}, 'horizontal tables must not trap vertical wheels');
assert.deepEqual(wheel(table, 1, 40), {forwarded:true, prevented:true}, 'small horizontal drift must not change scroll ownership');
assert.deepEqual(wheel(table, 40, 1), {forwarded:false, prevented:true}, 'one owner consumes horizontal scrolling');
assert.equal(table.scrollLeft, 40);
table.scrollLeft = 300;
assert(wheel(table, 40, 0).forwarded, 'forward at the table boundary');
const widget = new Element({style: {overflowY:'auto'}, scrollHeight:400});
assert(!wheel(widget, 0, 40).forwarded, 'scrollable widgets keep vertical input');
assert.equal(widget.scrollTop, 40);
assert(wheel(widget, 40, 0).forwarded, 'vertical widgets must not trap horizontal wheels');
widget.scrollTop = 300;
assert(wheel(widget, 0, 40).forwarded, 'forward at the widget boundary');
assert(!wheel(table, 0, 40, {ctrlKey:true}).forwarded, 'preserve browser zoom');
assert(!wheel(table, 0, 40, {defaultPrevented:true}).forwarded, 'respect document event handlers');
events.message({source:parentWindow, data:{kind:'expanded', value:true}});
assert.equal(root.style['overflow-y'], 'scroll', 'the sidebar has its own scrollbar');
root.scrollTop = 500;
root.scrollHeight = 1000;
table.parentElement = root;
assert.deepEqual(wheel(table, 0, 40), {forwarded:false, prevented:true}, 'sidebar content owns input before its boundary');
assert.equal(root.scrollTop, 540);
root.scrollTop = 900;
assert(!wheel(table, 0, 40).forwarded, 'sidebar bottom must not scroll the transcript');
assert(!wheel(table, 0, -40).forwarded, 'reversing direction immediately scrolls inside');
assert.equal(root.scrollTop, 860);
root.scrollTop = 0;
assert(!wheel(table, 0, -40).forwarded, 'sidebar top must not scroll the transcript');
root.scrollTop = 890;
assert.deepEqual(wheel(table, 0, 40), {forwarded:false, prevented:true}, 'a wheel reaching the boundary must not also move the transcript');
assert.equal(root.scrollTop, 900);
assert(!wheel(table, 0, 40).forwarded, 'sidebar remains isolated on subsequent wheels');
flush();
events.message({source:parentWindow, data:{kind:'expanded', value:false}});
flush();
assert.equal(root.scrollTop, 900, 'switching modes retains the internal scroll position');
assert(wheel(table, 0, 40).forwarded, 'inline bottom hands off too');
assert(!wheel(table, 0, -40).forwarded, 'inline direction reversal returns ownership inside');
assert.equal(root.style['overflow-y'], 'auto');
const offset = root.scrollTop;
height = 40000;
mutation(); flush();
assert.equal(root.style['overflow-y'], 'auto', 'content beyond the native height limit remains reachable');
height = 198;
mutation(); flush();
assert.equal(root.scrollTop, offset, 'height reports must not reset the internal scroll position');
console.log('HTML preview scrolling: directional ownership and boundaries passed');

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
assert.equal(elements['#expand'].attributes['aria-label'], '返回对话 (Esc)');
assert(bodyElement.classList.values.has('source'));
assert.equal(elements['#source'].textContent, '<p>Current document</p>');
assert(elements['#source'].focused);
hostEvents.keydown({key:'Escape'});
assert.equal(ipc.at(-1).action, 'source', 'Escape closes source before the enlarged preview');
hostWindow.previewMode(true, false);
hostEvents.message({source:frame.contentWindow, data:{kind:'escape'}});
assert.equal(ipc.at(-1).action, 'dismiss');
assert.equal(frame.srcdoc, payload, 'source and enlarged modes retain the running iframe');
hostWindow.previewMode(false, false);
assert.equal(elements['#expand'].attributes['aria-label'], '在右侧栏打开');
const count = ipc.length;
hostEvents.message({source:frame.contentWindow, data:{kind:'action', action:'download'}});
assert.equal(ipc.length, count, 'untrusted document cannot invoke native toolbar actions');
console.log('HTML preview toolbar: actions, capture, source, Escape and trust boundary passed');

hostWindow.previewLayout(600, 800, 0, 0);
assert.equal(elements['#tools'].style.top, '28px', 'toolbar sits inside the document margin');
assert.equal(elements['#tools'].style.right, 'calc(100% - 600px + 28px)');

// Exercise the exact host -> opaque frame -> IPC route, without dispatching DOM wheel.
frame.getBoundingClientRect = () => ({left:0, top:0});
frame.contentWindow.postMessage = data => events.message({source:parentWindow, data});
parentWindow.postMessage = data => hostEvents.message({source:frame.contentWindow, data});
root.scrollHeight = root.clientHeight;
root.scrollTop = 0;
let routed = ipc.length;
hostWindow.previewWheel(12, 34, 0, 40);
assert.equal(ipc.length, routed + 1, 'native wheels reach inline fallback without WebKit DOM events');
assert.equal(ipc.at(-1).dy, 40);
hostWindow.previewMode(true, false);
routed = ipc.length;
for (const dy of [-40, 40]) hostWindow.previewWheel(12, 34, 0, dy);
assert.equal(ipc.length, routed, 'short sidebar content cannot move the transcript');
hostEvents.message({source:frame.contentWindow, data:{kind:'wheel', x:0, y:0, dx:0, dy:40}});
assert.equal(ipc.length, routed, 'the host also rejects sidebar wheel handoff');
console.log('HTML preview native bridge: missing DOM delivery and sidebar isolation passed');


// Source lives in the trusted host, so its wheels never reach the iframe listener.
const sourceElement = elements['#source'];
Object.assign(sourceElement, new Element());
sourceElement.scrollBy = Element.prototype.scrollBy;
const sourceWheel = (dx, dy, options = {}) => {
  const event = {deltaX:dx, deltaY:dy, deltaMode:0, clientX:12, clientY:34,
    preventDefault() { this.prevented = true; }, ...options};
  const before = ipc.length;
  hostEvents.wheel?.(event);
  return {forwarded:ipc.length - before, prevented:!!event.prevented};
};
hostWindow.previewMode(false, true);
for (const dy of [-40, 40]) {
  assert.deepEqual(sourceWheel(0, dy), {forwarded:1, prevented:true}, 'short source hands off exactly once');
}
sourceElement.scrollHeight = 400;
assert.deepEqual(sourceWheel(0, 40), {forwarded:0, prevented:true});
assert.equal(sourceElement.scrollTop, 40);
sourceElement.scrollTop = 290;
assert.equal(sourceWheel(0, 40).forwarded, 0, 'reaching the boundary keeps one owner');
assert.equal(sourceElement.scrollTop, 300);
assert.equal(sourceWheel(0, 40).forwarded, 1);
assert.equal(sourceWheel(0, -40).forwarded, 0);
assert.equal(sourceElement.scrollTop, 260);
assert.deepEqual(sourceWheel(0, 40, {ctrlKey:true}), {forwarded:0, prevented:false});
assert.deepEqual(sourceWheel(0, 40, {defaultPrevented:true}), {forwarded:0, prevented:false});
hostWindow.previewMode(true, true);
sourceElement.scrollTop = 300;
assert.equal(sourceWheel(0, 40).forwarded, 0, 'sidebar source never scrolls chat');
sourceElement.scrollTop = 0;
assert.equal(sourceWheel(0, -40).forwarded, 0);
console.log('HTML preview source: DOM handoff, single consumption and sidebar isolation passed');
