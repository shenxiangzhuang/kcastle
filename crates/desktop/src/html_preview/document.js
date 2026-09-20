(() => {
  const send = data => parent.postMessage(data, '*');
  const forwardedWheels = new WeakSet();
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
    document.documentElement.style.setProperty('overflow-y', 'auto', 'important');
    document.documentElement.style.setProperty('overscroll-behavior', 'none', 'important');
    const observer = new ResizeObserver(schedule);
    observer.observe(document.body);
    new MutationObserver(schedule).observe(document.body, {subtree:true, childList:true, attributes:true, characterData:true});
    schedule();
  });
  addEventListener('load', schedule);
  addEventListener('resize', schedule);
  document.fonts.ready.then(schedule);
  addEventListener('message', event => {
    if (event.source === parent && event.data?.kind === 'expanded') {
      expanded = event.data.value;
      document.documentElement.style.setProperty('overflow-y', expanded ? 'scroll' : 'auto', 'important');
      // Republish measurements after a mode change.
      lastHeight = 0;
      schedule();
    }
    if (event.source === parent && event.data?.kind === 'nativeWheel') {
      const {x, y, dx, dy} = event.data;
      const target = document.elementFromPoint(x, y);
      dispatchWheel(target || document.documentElement, {bubbles:true, cancelable:true,
        clientX:x, clientY:y, deltaX:dx, deltaY:dy});
    }
    if (event.source === parent && event.data?.kind === 'theme') {
      document.documentElement.style.colorScheme = event.data.dark ? 'dark' : 'light';
      dispatchEvent(new CustomEvent('kcastle-theme', {detail:{dark:event.data.dark}}));
      schedule();
    }
  });
  // Both native input and browser wheels use the same single-owner scroll policy.
  const scroll = (target, x, y, dx, dy) => {
    const vertical = Math.abs(dy) >= Math.abs(dx);
    let node = target instanceof Element ? target : target?.parentElement;
    while (node) {
      const style = getComputedStyle(node);
      const y = /(auto|scroll)/.test(style.overflowY) && node.scrollHeight > node.clientHeight + 1;
      const x = /(auto|scroll)/.test(style.overflowX) && node.scrollWidth > node.clientWidth + 1;
      if ((vertical && dy && y && (dy < 0 ? node.scrollTop > 0 : node.scrollTop + node.clientHeight < node.scrollHeight - 1)) ||
          (!vertical && dx && x && (dx < 0 ? node.scrollLeft > 0 : node.scrollLeft + node.clientWidth < node.scrollWidth - 1))) {
        // Synchronous movement avoids native scroll chaining and smooth-scroll lag
        // racing the next boundary decision. Excess delta stays with this owner.
        const before = vertical ? node.scrollTop : node.scrollLeft;
        node.scrollBy({top:vertical ? dy : 0, left:vertical ? 0 : dx, behavior:'instant'});
        if ((vertical ? node.scrollTop : node.scrollLeft) !== before) return;
      }
      node = node.parentElement;
    }
    if (!expanded) send({kind:'wheel', x, y, dx, dy});
  };
  const dispatchWheel = (target, properties) => {
    const wheel = new WheelEvent('wheel', properties);
    forwardedWheels.add(wheel);
    // Both entry points wait for every page listener, including later window listeners.
    if (target.dispatchEvent(wheel)) {
      const factor = wheel.deltaMode === 1 ? 20 : wheel.deltaMode === 2 ? innerHeight : 1;
      scroll(target, wheel.clientX, wheel.clientY, wheel.deltaX * factor, wheel.deltaY * factor);
    }
  };
  addEventListener('wheel', event => {
    if (forwardedWheels.has(event) || event.ctrlKey || event.metaKey || event.defaultPrevented || !event.cancelable) return;
    // Suppress browser scrolling and duplicate page delivery before redispatching once.
    event.preventDefault();
    event.stopImmediatePropagation();
    dispatchWheel(event.target, event);
  }, {capture:true, passive:false});
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
