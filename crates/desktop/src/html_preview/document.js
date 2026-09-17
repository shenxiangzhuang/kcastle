(() => {
  const send = data => parent.postMessage(data, '*');
  let scheduled = false, lastHeight = 0, expanded = false;
  const measure = () => {
    scheduled = false;
    if (!document.body) return;
    const body = document.body;
    const style = getComputedStyle(body);
    const margin = parseFloat(style.marginBottom) || 0;
    const rect = body.getBoundingClientRect();
    let bottom = Math.max(rect.bottom, rect.top + body.scrollHeight) + scrollY + margin;
    for (const child of body.children) {
      if (['SCRIPT','STYLE','LINK'].includes(child.tagName)) continue;
      bottom = Math.max(bottom, child.getBoundingClientRect().bottom + scrollY + margin);
    }
    const height = Math.min(32768, Math.max(64, Math.ceil(bottom)));
    if (height !== lastHeight) { lastHeight = height; send({kind:'height', height}); }
  };
  const schedule = () => {
    // WebKit can suspend animation frames in an occluded native child view.
    if (!scheduled) { scheduled = true; queueMicrotask(measure); }
  };
  addEventListener('DOMContentLoaded', () => {
    const observer = new ResizeObserver(schedule);
    observer.observe(document.body);
    new MutationObserver(schedule).observe(document.body, {subtree:true, childList:true, attributes:true, characterData:true});
    schedule();
  });
  addEventListener('load', schedule);
  addEventListener('resize', schedule);
  document.fonts.ready.then(schedule);
  addEventListener('message', event => {
    if (event.source === parent && event.data?.kind === 'expanded') expanded = !!event.data.value;
    if (event.source === parent && event.data?.kind === 'theme') {
      document.documentElement.style.colorScheme = event.data.dark ? 'dark' : 'light';
      dispatchEvent(new CustomEvent('kcastle-theme', {detail:{dark:event.data.dark}}));
      schedule();
    }
  });
  // Inner scrollable widgets keep their wheel events. At their edge, continue the transcript.
  addEventListener('wheel', event => {
    if (expanded) return;
    if (event.ctrlKey || event.metaKey || event.defaultPrevented) return;
    let node = event.target instanceof Element ? event.target : event.target.parentElement;
    while (node) {
      const style = getComputedStyle(node);
      const dy = event.deltaY, dx = event.deltaX;
      const y = /(auto|scroll)/.test(style.overflowY) && node.scrollHeight > node.clientHeight + 1;
      const x = /(auto|scroll)/.test(style.overflowX) && node.scrollWidth > node.clientWidth + 1;
      if ((y && (dy < 0 ? node.scrollTop > 0 : node.scrollTop + node.clientHeight < node.scrollHeight - 1)) ||
          (x && (dx < 0 ? node.scrollLeft > 0 : node.scrollLeft + node.clientWidth < node.scrollWidth - 1))) return;
      node = node.parentElement;
    }
    event.preventDefault();
    const factor = event.deltaMode === 1 ? 20 : event.deltaMode === 2 ? innerHeight : 1;
    send({kind:'wheel', x:event.clientX, y:event.clientY, dx:event.deltaX*factor, dy:event.deltaY*factor});
  }, {passive:false});
  addEventListener('keydown', event => {
    if (event.key === 'Escape') send({kind:'escape'});
  });
  addEventListener('click', event => {
    const target = event.target instanceof Element ? event.target.closest('a, input[type=file]') : null;
    if (target?.matches('input[type=file]') || (target?.tagName === 'A' && !target.getAttribute('href')?.startsWith('#'))) event.preventDefault();
  }, true);
  addEventListener('error', event => send({kind:'error', message:event.message || 'Script error'}));
  addEventListener('unhandledrejection', event => send({kind:'error', message:String(event.reason)}));
})();
