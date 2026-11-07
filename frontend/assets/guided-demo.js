const $ = id => document.getElementById(id);
const scenarioButtons = [...document.querySelectorAll('[data-scenario]')];
const scenarioCopy = {
  delayed_refund: '配送异常时，系统会先整理订单事实与处理依据，再交由售后审核。',
  track_delivery: '标准配送查询直接返回结果，不占用人工审核。',
  delivered_address: '订单状态不允许时，系统会说明原因并阻止无效操作。',
};
const actionLabels = {refund_request: '退款申请', change_address: '修改收货地址', cancel_order: '取消订单', complaint_escalation: '投诉处理'};
const statusLabels = {pending_review: '等待人工审核', executed: '申请已处理', rejected: '申请未通过', resolved: '已完成'};
let selectedScenario = 'delayed_refund';
let activeSession = '';
let busy = false;

function setBusy(value) {
  busy = value;
  $('start').disabled = value;
  $('approve').disabled = value;
  $('reject').disabled = value;
  scenarioButtons.forEach(button => { button.disabled = value; });
}

function addBubble(text, type, title) {
  const node = document.createElement('div');
  node.className = `bubble ${type}`;
  const label = document.createElement('strong');
  label.textContent = title;
  const content = document.createElement('span');
  content.textContent = text;
  node.append(label, content);
  $('conversation').append(node);
}

function resetView() {
  activeSession = '';
  $('conversation').replaceChildren();
  const welcome = document.createElement('div');
  welcome.className = 'welcome';
  welcome.innerHTML = '<strong>你好，我是订单与售后助手。</strong><span>我会先核对订单和规则，再给出下一步。</span>';
  $('conversation').append(welcome);
  $('customerState').className = 'state-pill';
  $('customerState').textContent = '等待开始';
  $('orderName').textContent = '选择情境后加载';
  $('orderDetail').textContent = '公开历史订单 · 仅用于本地体验';
  $('timeline').innerHTML = '<li><span>1</span><div><strong>等待用户请求</strong><small>选择一个情境开始体验</small></div></li><li class="muted"><span>2</span><div><strong>核对订单与规则</strong></div></li><li class="muted"><span>3</span><div><strong>生成处理结果</strong></div></li>';
  $('reviewCard').hidden = true;
  $('resultCard').hidden = true;
  $('feedback').textContent = '';
  $('start').textContent = '开始体验';
}

function renderTimeline(items) {
  $('timeline').replaceChildren();
  items.forEach((item, index) => {
    const li = document.createElement('li');
    const step = document.createElement('span');
    step.textContent = String(index + 1);
    const body = document.createElement('div');
    const label = document.createElement('strong');
    label.textContent = item.label;
    const status = document.createElement('small');
    status.textContent = item.status === 'awaiting_confirmation' ? '等待工作人员审核' : '已完成';
    body.append(label, status); li.append(step, body); $('timeline').append(li);
  });
}

function addFact(container, label, value) {
  if (value === undefined || value === null || value === '') return;
  const row = document.createElement('div');
  const key = document.createElement('dt');
  const content = document.createElement('dd');
  key.textContent = label; content.textContent = String(value); row.append(key, content); container.append(row);
}

function renderReview(packet, needsReview) {
  const card = $('reviewCard');
  if (!needsReview) { card.hidden = true; return; }
  card.hidden = false;
  const facts = packet.order_facts || {};
  const list = $('reviewFacts');
  list.replaceChildren();
  addFact(list, '申请类型', actionLabels[packet.action_type] || packet.action_type);
  addFact(list, '订单状态', facts.order_status);
  addFact(list, '配送延迟', facts.delay_days === null ? '' : `${facts.delay_days} 天`);
  addFact(list, '订单金额', facts.payment_value === undefined ? '' : `R$ ${facts.payment_value}`);
  addFact(list, '风险等级', packet.risk_level === 'high' ? '高' : packet.risk_level === 'medium' ? '中' : '低');
  const reasons = [...(packet.customer_message_points || []), ...(packet.handoff_reasons || [])];
  $('reviewReason').textContent = reasons.length ? reasons.slice(0, 3).join('；') : '系统已完成订单与规则核对，请确认是否继续处理。';
}

function renderResult(data) {
  const result = $('resultCard');
  result.hidden = false;
  const status = statusLabels[data.status] || data.status;
  result.textContent = `${status}\n${data.customer.answer}`;
}

function renderPayload(data, {appendCustomer = true} = {}) {
  activeSession = data.session_id;
  $('orderName').textContent = data.scenario.title;
  const facts = data.review_packet?.order_facts || {};
  $('orderDetail').textContent = facts.order_status ? `订单状态：${facts.order_status} · 订单金额：R$ ${facts.payment_value ?? '—'}` : '订单信息已核对';
  if (appendCustomer) {
    addBubble(data.customer.request, 'user', '用户');
    addBubble(data.customer.answer, 'agent', '订单与售后助手');
  } else {
    addBubble(data.customer.answer, 'agent', '订单与售后助手');
  }
  renderTimeline(data.timeline || []);
  const reviewNeeded = Boolean(data.review_required);
  renderReview(data.review_packet || {}, reviewNeeded);
  $('customerState').className = `state-pill ${reviewNeeded ? 'active' : 'done'}`;
  $('customerState').textContent = reviewNeeded ? '等待人工审核' : (statusLabels[data.status] || '已完成');
  if (!reviewNeeded) renderResult(data);
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || '服务暂时不可用，请稍后重试。');
  return body;
}

async function startScenario() {
  if (busy) return;
  setBusy(true); resetView();
  $('feedback').textContent = '正在核对订单、处理规则与服务记录…';
  try {
    const data = await request('/demo/run', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({scenario_id: selectedScenario})});
    renderPayload(data);
    $('feedback').textContent = data.review_required ? '申请已生成，切换到右侧完成审核。' : '该请求已在当前服务环节完成。';
  } catch (error) {
    $('feedback').textContent = `无法启动体验：${error.message}`;
  } finally { setBusy(false); }
}

async function review(decision) {
  if (!activeSession || busy) return;
  setBusy(true);
  $('feedback').textContent = decision === 'approve' ? '正在写入本地售后申请…' : '正在同步审核结果…';
  try {
    const data = await request(`/demo/sessions/${encodeURIComponent(activeSession)}/review`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({decision})});
    renderPayload(data, {appendCustomer: false});
    $('feedback').textContent = '本次体验已完成。点击“开始体验”可从干净的本地沙箱重新运行。';
    $('start').textContent = '重新开始体验';
  } catch (error) {
    $('feedback').textContent = `审核结果未能提交：${error.message}`;
  } finally { setBusy(false); }
}

scenarioButtons.forEach(button => button.addEventListener('click', () => {
  selectedScenario = button.dataset.scenario;
  scenarioButtons.forEach(item => item.classList.toggle('selected', item === button));
  $('scenarioSummary').textContent = scenarioCopy[selectedScenario];
  resetView();
}));
$('start').addEventListener('click', startScenario);
$('approve').addEventListener('click', () => review('approve'));
$('reject').addEventListener('click', () => review('reject'));
resetView();
