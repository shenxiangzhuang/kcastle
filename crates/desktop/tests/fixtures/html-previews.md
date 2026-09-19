# 两个独立的交互式图解

两个页面应自动出现。分别拖动滑块，然后滚动、调整窗口大小，再回来检查数值。

```html
<style>
body { margin: 20px; color: #354458; background: #f2f6fa; font-family: system-ui }
h3 { margin: 0 0 8px } label { display:flex; gap:12px; align-items:center }
input { flex:1; accent-color:#537e9a } .track { height:48px; position:relative; margin-top:16px; background:#dce6ee; border-radius:8px }
.dot { position:absolute; width:28px; height:28px; border-radius:50%; background:#537e9a; top:10px }
button { margin-top:12px; padding:6px 12px }
</style>
<h3>匀速运动</h3>
<label>速度 <input aria-label="速度" id="speed" type="range" min="0" max="100" value="30"><output id="value">30</output></label>
<div class="track"><div class="dot" id="dot"></div></div>
<button id="expand">展开解释</button><p id="detail" hidden>位移等于速度乘以时间。这个按钮改变内容高度，后面的消息应自然下移。</p>
<script>
const speed = document.querySelector('#speed'), value = document.querySelector('#value');
speed.oninput = () => { value.textContent=speed.value; document.querySelector('#dot').style.left=`calc(${speed.value}% - ${speed.value*0.28}px)`; };
speed.oninput();
document.querySelector('#expand').onclick = () => document.querySelector('#detail').hidden = !document.querySelector('#detail').hidden;
</script>
```

这段文字属于原生对话，应该正常选择和复制。下面是另一个独立页面。

```html
<style>
body { margin:20px; color:#554257; background:#f7f2f6; font-family:system-ui }
h3 { margin:0 0 8px } label { display:flex; gap:12px; align-items:center }
input { flex:1; accent-color:#947397 } svg { display:block; width:100%; height:80px }
</style>
<h3>概率分布</h3>
<label>概率 <input aria-label="概率" id="probability" type="range" min="0" max="100" value="65"><output id="value">65%</output></label>
<svg viewBox="0 0 500 80"><rect x="0" y="15" width="500" height="45" rx="8" fill="#e6dce6"/><rect id="fill" x="0" y="15" width="325" height="45" rx="8" fill="#947397"/></svg>
<p>长表格：纵向滚动应连续，横向滚动只移动表格。</p>
<div style="overflow-x:auto;overflow-y:hidden"><table style="width:900px;border-collapse:collapse"><tbody id="rows"></tbody></table></div>
<script>
document.querySelector('#rows').innerHTML = Array.from({length:40}, (_, i) => `<tr><td style="padding:12px">第 ${i+1} 行</td><td>横向滚动检查</td><td>右侧列</td></tr>`).join('');
const slider = document.querySelector('#probability');
slider.oninput = () => { document.querySelector('#value').textContent=slider.value+'%'; document.querySelector('#fill').setAttribute('width', slider.value*5); };
</script>
```

## 检查滚动后的状态

向上滚动，两个图解应保留各自数值。页面不能遮住标题栏、输入框或设置弹窗。

第一段普通文字。

第二段普通文字。

第三段普通文字。

第四段普通文字。

第五段普通文字。

第六段普通文字。

第七段普通文字。

第八段普通文字。

第九段普通文字。

第十段普通文字。
