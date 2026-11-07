const $ = id => document.getElementById(id);
const session = sessionStorage.getItem('support-session') || crypto.randomUUID();
sessionStorage.setItem('support-session', session);

let selectedOrder = null;
let busy = false;
let lastStatus = '';
let polling = false;
let progressExpanded = false;
let lastMessageId = 0;
let latestActions = [];
window.customerOrders = [];

const states = {
  pending_review: '等待审核',
  appealed_pending_review: '复核中',
  approving: '处理中',
  executed: '已处理',
  rejected: '未通过审核',
  timeout_canceled: '已关闭',
  execution_failed: '处理异常',
  withdrawn: '已撤回',
};
const caseActionNames = {
  refund_request: '退款申请',
  cancel_order: '取消订单',
  change_address: '修改地址',
  invoice_request: '发票申请',
  complaint_escalation: '投诉处理',
  open_support_case: '售后跟进',
};
const nextSteps = {
  pending_review: '审核通过后会继续处理，请留意消息。',
  appealed_pending_review: '我们正在复核您补充的信息。',
  approving: '审核已通过，系统正在执行，请稍候。',
  executed: '已完成，可在订单或售后记录中查看结果。',
  rejected: '如有疑问，可补充说明或联系客服。',
  timeout_canceled: '本次申请已关闭，如仍需处理可重新发起。',
  withdrawn: '您已撤回这项申请，不会再处理；如仍需办理可重新发起。',
  execution_failed: '处理遇到问题，客服会继续跟进。',
};

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error('暂时无法完成操作，请稍后重试。');
  return response.json();
}

function add(message, type = 'agent') {
  const element = document.createElement('div');
  element.className = `message ${type}`;
  element.textContent = message;
  $('chat').append(element);
  $('chat').scrollTop = $('chat').scrollHeight;
}

function formatStatus(status) {
  return ({created: '待付款', approved: '已付款', invoiced: '待发货', processing: '处理中', shipped: '运输中', delivered: '已送达', canceled: '已取消'})[status] || status;
}

function formatCategory(value) {
  return ({
    health_beauty: '健康与美容用品',
    bed_bath_table: '家居用品',
    computers_accessories: '电脑配件',
    baby: '母婴用品',
  })[value] || (value ? value.replaceAll('_', ' ') : '商品订单');
}

function chooseOrder(order) {
  selectedOrder = order;
  document.querySelectorAll('.order').forEach(node => node.classList.toggle('selected', node.dataset.orderId === order.order_id));
  $('selectionHint').textContent = '已关联这笔订单。您可以继续查看进度，或直接发起售后申请。';
}

function actionButton(action) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = action.style === 'primary' ? 'action primary' : 'action';
  button.textContent = action.label;
  button.onclick = () => {
    const order = action.order_id ? window.customerOrders.find(item => item.order_id === action.order_id) : selectedOrder;
    if (order) chooseOrder(order);
    sendMessage(action.message || action.label, action.id, action.confirmation_id || null);
  };
  return button;
}

function renderOrders(payload) {
  window.customerOrders = payload.orders || [];
  const summary = payload.summary || {};
  $('orderCount').textContent = `${summary.order_count || 0} 笔`;
  $('orderSummary').replaceChildren();
  for (const [label, value] of [['在途', summary.active_shipments || 0], ['待留意', summary.attention_count || 0]]) {
    const metric = document.createElement('div');
    metric.innerHTML = `<strong>${value}</strong><span>${label}</span>`;
    $('orderSummary').append(metric);
  }
  $('orders').replaceChildren();
  for (const order of window.customerOrders) {
    const card = document.createElement('article');
    card.className = 'order';
    card.dataset.orderId = order.order_id;
    const top = document.createElement('div');
    top.className = 'order-top';
    top.innerHTML = `<strong>${order.category_label || formatCategory(order.category)}</strong><span class="order-status ${order.needs_attention ? 'attention' : ''}">${formatStatus(order.status)}</span>`;
    const amount = document.createElement('span');
    amount.className = 'price';
    amount.textContent = new Intl.NumberFormat('zh-CN', {style: 'currency', currency: order.currency || 'BRL'}).format(order.amount);
    const meta = document.createElement('small');
    meta.textContent = `下单时间：${String(order.purchased_at || '').slice(0, 10) || '暂未提供'}`;
    card.append(top, amount, meta);
    if (order.needs_attention) {
      const note = document.createElement('small');
      note.className = 'order-note';
      note.textContent = order.delay_days > 0 ? `预计延迟 ${order.delay_days} 天` : '有售后进度更新';
      card.append(note);
    }
    const actions = document.createElement('div');
    actions.className = 'order-actions';
    (order.actions || []).slice(0, 3).forEach(action => actions.append(actionButton(action)));
    actions.onclick = event => event.stopPropagation();
    card.append(actions);
    $('orders').append(card);
  }
}

function renderFollowUps(actions) {
  latestActions = actions || [];
  if (!latestActions.length) return;
  // Confirmation cards: one tray per card, never truncated, labelled with its order when there are several.
  const cards = new Map();
  const others = [];
  for (const action of latestActions) {
    if (!action.confirmation_id) { others.push(action); continue; }
    if (!cards.has(action.confirmation_id)) cards.set(action.confirmation_id, []);
    cards.get(action.confirmation_id).push(action);
  }
  for (const group of cards.values()) {
    const tray = document.createElement('div');
    tray.className = 'message-actions confirmation-card';
    const orderId = group.find(action => action.order_id)?.order_id;
    const order = orderId ? (window.customerOrders || []).find(item => item.order_id === orderId) : null;
    if (order && cards.size > 1) {
      const caption = document.createElement('span');
      caption.className = 'action-caption';
      caption.textContent = `${order.category_label || formatCategory(order.category)} ·${new Intl.NumberFormat('zh-CN', {style: 'currency', currency: order.currency || 'BRL'}).format(order.amount)}`;
      tray.append(caption);
    }
    group.forEach(action => tray.append(actionButton(action)));
    // A card is answered once: its token is single use on the server as well.
    tray.addEventListener('click', event => {
      if (event.target.closest('button')) tray.querySelectorAll('button').forEach(button => { button.disabled = true; });
    });
    $('chat').append(tray);
  }
  if (others.length) {
    const tray = document.createElement('div');
    tray.className = 'message-actions';
    others.slice(0, 4).forEach(action => tray.append(actionButton(action)));
    $('chat').append(tray);
  }
  $('chat').scrollTop = $('chat').scrollHeight;
}

function progressCounts(cases, summary) {
  const counts = summary || {};
  const processed = counts.processed ?? cases.filter(item => item.display_status === 'processed' || item.status === 'executed').length;
  const pending = counts.pending_review ?? cases.filter(item => ['pending_review', 'appealed_pending_review', 'approving'].includes(item.status)).length;
  return {processed, pending};
}

function renderProgress(cases, summary) {
  $('progress').hidden = false;
  $('progress').replaceChildren();
  const {processed, pending} = progressCounts(cases, summary);
  const partial = processed > 0 && pending > 0;
  const heading = document.createElement('strong');
  heading.className = 'progress-heading';
  heading.textContent = partial
    ? `售后进度（已处理 ${processed} · 待审核 ${pending}）`
    : `售后进度（${cases.length}）`;
  $('progress').append(heading);
  if (partial) {
    const rollup = document.createElement('div');
    rollup.className = 'progress-summary';
    const done = document.createElement('span');
    done.className = 'progress-summary-done';
    done.textContent = `已处理 ${processed}`;
    const todo = document.createElement('span');
    todo.className = 'progress-summary-pending';
    todo.textContent = `待审核 ${pending}`;
    rollup.append(done, todo);
    $('progress').append(rollup);
  }
  // A partial-auto request must show executed and pending sub-cases together;
  // collapsing the remaining list behind a toggle would hide the split result.
  const visible = progressExpanded || partial ? cases : cases.slice(0, 2);
  for (const current of visible) {
    const item = document.createElement('div');
    item.className = 'progress-case';
    const title = document.createElement('strong');
    const action = caseActionNames[current.action_type] || '售后申请';
    title.textContent = `${action} · ${states[current.status] || '处理中'}`;
    const meta = document.createElement('small');
    const shortId = current.case_id ? current.case_id.slice(-6) : '';
    meta.textContent = `服务单 ${shortId}${current.order_id ? ` · 订单 ${current.order_id}` : ''}`;
    const next = document.createElement('small');
    next.className = 'progress-next';
    next.textContent = nextSteps[current.status] || '请留意后续消息。';
    item.append(title, meta, next);
    $('progress').append(item);
  }
  if (cases.length > 2) {
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'progress-toggle';
    toggle.setAttribute('aria-expanded', String(progressExpanded));
    toggle.textContent = progressExpanded ? '收起' : `查看全部 ${cases.length} 个申请`;
    toggle.onclick = () => {
      progressExpanded = !progressExpanded;
      renderProgress(cases, summary);
    };
    $('progress').append(toggle);
  }
}

async function progress() {
  if (polling || document.hidden) return;
  polling = true;
  try {
    const data = await api(`/customer/sessions/${encodeURIComponent(session)}/progress`);
    const cases = data.cases || (data.case ? [data.case] : []);
    if (!cases.length) return;
    renderProgress(cases);
    const current = cases[cases.length - 1];
    if (lastStatus && lastStatus !== current.status) {
      const action = caseActionNames[current.action_type] || '售后申请';
      add(`${action}状态已更新：${states[current.status] || '处理中'}`, 'update');
    }
    lastStatus = current.status;
  } catch {
    $('status').textContent = '申请进度暂时无法刷新，将自动重试。';
  } finally {
    polling = false;
  }
}

async function sendMessage(message, requestedAction = null, confirmationId = null) {
  if (!message || busy) return;
  busy = true;
  $('send').disabled = true;
  add(message, 'user');
  $('status').textContent = requestedAction ? '正在准备处理…' : '正在查询…';
  try {
    const data = await api('/customer/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message,
        session_id: session,
        requested_action: requestedAction,
        confirmation_id: confirmationId || undefined,
        page_context: {surface: selectedOrder ? 'order_detail' : 'order_center', selected_order_id: selectedOrder?.order_id || null},
      }),
    });
    add(data.answer);
    renderFollowUps(data.ui_actions);
    $('status').textContent = '';
    await progress();
  } catch (error) {
    $('status').textContent = error.message;
  } finally {
    busy = false;
    $('send').disabled = false;
  }
}

async function send(event) {
  event.preventDefault();
  const message = $('message').value.trim();
  if (!message) return;
  $('message').value = '';
  await sendMessage(message);
}

$('compose').onsubmit = send;
$('resetSession').onclick = () => {
  sessionStorage.removeItem('support-session');
  window.location.reload();
};
document.querySelectorAll('[data-query]').forEach(button => button.onclick = () => {
  if (button.dataset.action) sendMessage(button.dataset.query, button.dataset.action);
  else {
    $('message').value = button.dataset.query;
    $('message').focus();
  }
});

add('你好，我是订单与售后助手。请直接描述遇到的问题，例如“订单还没到”“我要取消订单”或“我想申请退款”。');
api('/customer/context').then(renderOrders).catch(error => {$('orders').textContent = error.message;});
api('/runtime/status').then(data => {
  if (data.mode !== 'live_llm_agent') {
    $('availability').hidden = false;
    $('availability').textContent = '当前处于基础服务模式，复杂问题会按售后流程继续处理。';
  }
}).catch(() => {});
progress();
setInterval(progress, 6000);

async function receive() {
  if (busy || document.hidden) return;
  const response = await fetch(`/customer/sessions/${encodeURIComponent(session)}/messages`);
  if (!response.ok) return;
  const data = await response.json();
  const last = data.messages.at(-1)?.id || 0;
  if (!last || last === lastMessageId) return;
  $('chat').replaceChildren();
  for (const item of data.messages) add((item.sender === 'staff' ? '人工客服：' : '') + item.content, item.sender === 'customer' ? 'user' : 'agent');
  if (latestActions.length) renderFollowUps(latestActions);
  lastMessageId = last;
}
setInterval(() => receive().catch(() => {}), 4000);
