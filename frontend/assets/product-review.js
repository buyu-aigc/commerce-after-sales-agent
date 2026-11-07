const $ = id => document.getElementById(id);
const names = {refund_request: '退款申请', cancel_order: '取消订单', change_address: '修改收货地址', invoice_request: '发票申请', complaint_escalation: '投诉处理', open_support_case: '售后跟进'};
const statusNames = {delivered: '已送达', shipped: '配送中', canceled: '已取消', created: '待处理', approved: '已确认'};
const senderNames = {customer: '客户', assistant: '智能客服', staff: '人工客服'};
let token = '';
let selected = null;
let loadingConversation = false;

async function api(path, options = {}) {
  const headers = {'X-Review-Token': token, ...(options.headers || {})};
  if (options.body) headers['Content-Type'] = 'application/json';
  const response = await fetch(path, {...options, headers});
  if (!response.ok) {
    if (response.status === 403) throw new Error('访问凭证无效或权限不足。');
    if (response.status === 409) throw new Error('该 Case 已被其他审核员处理，请刷新查看最新状态。');
    let detail = '';
    try {
      const body = await response.json();
      detail = typeof body?.detail === 'string' ? body.detail : '';
    } catch (_error) { /* ignore malformed error body */ }
    throw new Error(detail || '操作未完成，请刷新后重试。');
  }
  return response.json();
}

function setStatus(text) {$('notice').textContent = text;}
function addFact(label, value) {
  const row = document.createElement('div');
  const term = document.createElement('dt');
  const detail = document.createElement('dd');
  term.textContent = label;
  detail.textContent = value == null || value === '' ? '未提供' : String(value);
  row.append(term, detail);
  $('facts').append(row);
}
function fillList(id, values) {
  $(id).replaceChildren();
  for (const value of values?.length ? values : ['暂无依据']) {
    const item = document.createElement('li');
    item.textContent = value;
    $(id).append(item);
  }
}
function formatTimestamp(value) {return value ? value.slice(0, 16).replace('T', ' ') : '刚刚';}
const reviewStatusNames = {pending_review: '待审核', appealed_pending_review: '申诉待审', approving: '处理中', executed: '已执行', rejected: '已驳回', execution_failed: '执行失败', timeout_canceled: '已超时'};
function formatAge(seconds) {
  const value = Number(seconds || 0);
  if (value < 60) return `${value}s`;
  if (value < 3600) return `${Math.floor(value / 60)}m`;
  return `${Math.floor(value / 3600)}h`;
}
function pct(value) {return `${Math.round((value || 0) * 10000) / 100}%`;}

function metricCard(title, value, note) {
  const card = document.createElement('div');
  card.className = 'metric-card';
  const heading = document.createElement('strong');
  const number = document.createElement('b');
  const caption = document.createElement('small');
  heading.textContent = title;
  number.textContent = value;
  caption.textContent = note;
  card.append(heading, number, caption);
  return card;
}

async function renderMetrics() {
  const panel = $('quality');
  try {
    const data = await api('/review/metrics');
    const product = data.product_evaluation || {};
    const core = product.core || {};
    const planner = product.llm_planner || {};
    const ablation = product.ablation || {};
    panel.replaceChildren();
    const title = document.createElement('h2');
    title.textContent = '质量概览';
    const grid = document.createElement('div');
    grid.className = 'metric-grid';
    const held = (product.baselines || {}).heldout_current;
    if (held) grid.append(
      metricCard('评测集诉求正确', `${held.requests_correct}/${held.requests_total}`, `整条办结 ${held.cases_complete}/${held.cases_total}`),
      metricCard('额外人工', `${held.human_involved - held.policy_floor}`, `人工介入 ${held.human_involved}/${held.requests_total}，策略下限 ${held.policy_floor}`),
      metricCard('零容忍', `${held.wrong_executions} / ${held.high_risk_auto} / ${held.leaks}`, '错误执行 / 高风险误自动 / 信息泄露'),
      metricCard('客户点击', `${held.customer_clicks}`, `可自动的实际自动 ${held.auto_realized}/${held.auto_total}`),
    );
    else grid.append(
      metricCard('售后主链路', core.case_oracle_pass || '-', `转审核 P/R ${core.handoff_precision || '-'}/${core.handoff_recall || '-'}`),
      metricCard('真实 LLM 规划', planner.exact_task_plan || '-', `${planner.live_planner_turns || '-'} 真实规划 · P95 ${planner.p95_latency_ms || '-'}ms`),
      metricCard('风险门禁', core.forbidden_cancellation || '-', `禁取消写入 · 安全自动 ${core.safe_auto_resolution || '-'}`),
      metricCard('消融对照', (ablation.execute_on_intent || {}).forbidden_cancellation || '-', '识别即写会错误取消；完整门禁为 0/1'),
    );
    const scope = document.createElement('p');
    scope.className = 'metric-scope';
    scope.textContent = product.scope || '本地 trace 仅用于排障，不与业务 KPI 混算。';
    panel.append(title, grid, scope);
    panel.hidden = false;
  } catch (_error) {
    panel.hidden = true;
  }
}

async function renderShadow() {
  const panel = $('shadow');
  try {
    const data = await api('/shadow/metrics');
    const summary = data.summary || {};
    panel.replaceChildren();
    const title = document.createElement('h2');
    title.textContent = 'Shadow 对照（只建议，不执行）';
    const grid = document.createElement('div');
    grid.className = 'metric-grid';
    grid.append(
      metricCard('Shadow auto 建议', summary.auto_suggestions || 0, `已决定 ${summary.auto_decided || 0} · 人工批准 ${summary.auto_approved || 0}`),
      metricCard('自动建议人工驳回率', summary.false_auto_rate == null ? '待审核' : pct(summary.false_auto_rate), `${summary.false_auto || 0} 条；业务决定对照，非自动资格标签`),
      metricCard('漏自动率', summary.missed_auto_rate == null ? '待独立标注' : pct(summary.missed_auto_rate), `${summary.missed_auto || 0} 条经标注适合自动处理`),
      metricCard('高风险 auto', summary.high_risk_auto_suggestions || 0, '必须为 0；影子建议不会执行'),
    );
    const status = document.createElement('p');
    status.className = 'metric-scope';
    status.textContent = data.enabled
      ? `影子模式运行中 · 采样 ${pct(data.sample_rate)} · 当前 ${summary.pending_decision || 0} 条等待人工标签`
      : '影子模式未启用；以下为历史对照数据。';
    panel.append(title, grid, status);
    for (const row of (data.by_risk || [])) {
      const line = document.createElement('p');
      line.className = 'shadow-row';
      line.textContent = `${row.key} · 建议 ${row.suggestions} · auto ${row.auto_suggestions} · 误自动 ${row.false_auto} · 高风险 auto ${row.high_risk_auto}`;
      panel.append(line);
    }
    panel.hidden = false;
  } catch (_error) {
    panel.hidden = true;
  }
}

async function renderConversation() {
  if (!selected || loadingConversation || document.hidden) return;
  loadingConversation = true;
  const id = selected.session_id;
  try {
    const data = await api(`/review/sessions/${encodeURIComponent(id)}/messages`);
    if (!selected || selected.session_id !== id) return;
    $('conversation').replaceChildren();
    for (const message of data.messages) {
      const row = document.createElement('div');
      row.className = 'desk-message';
      const label = document.createElement('strong');
      const content = document.createElement('p');
      label.textContent = senderNames[message.sender] || '系统';
      content.textContent = message.content;
      row.append(label, content);
      $('conversation').append(row);
    }
    $('conversation').scrollTop = $('conversation').scrollHeight;
    $('takeover').disabled = data.human;
    $('automatic').disabled = !data.human;
  } catch (error) {
    $('replyStatus').textContent = error.message;
  } finally {
    loadingConversation = false;
  }
}

function renderCasePicker(cases) {
  const picker = $('casePicker');
  picker.replaceChildren();
  if (!cases || cases.length < 2) {
    picker.hidden = true;
    return;
  }
  picker.hidden = false;
  const label = document.createElement('strong');
  label.textContent = '本会话有多个售后 Case，请选择要处理的 Case';
  picker.append(label);
  for (const item of cases) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'case-chip';
    const order = item.order_id ? ` · 订单 ${String(item.order_id).slice(-6)}` : '';
    const attention = item.needs_attention ? ' · 需关注' : '';
    button.textContent = `${names[item.action_type] || '售后'}${order} · ${reviewStatusNames[item.status] || item.status} · ${formatAge(item.age_seconds)}${attention}`;
    button.classList.toggle('selected', item.case_id === selected.caseId);
    button.onclick = () => loadCaseDetail(selected, item.case_id);
    picker.append(button);
  }
}

function renderReviewData(record, data, caseId) {
  const caseRecord = data.after_sales_cases?.at(-1) || {};
  const draft = data.escalation_draft || {};
  const decision = draft.decision || caseRecord.decision || {};
  const facts = data.order_facts || {};
  const isCase = data.after_sales_cases?.length || data.has_pending;
  selected.caseId = caseId || null;
  selected.workflowAttempt = data.workflow_attempt || 1;
  $('caseTitle').textContent = isCase ? (names[data.action_type || record.action_type] || '售后申请') : '客户咨询';
  const statusLabel = reviewStatusNames[data.status] || (data.has_pending ? '待审核' : (isCase ? '已处理' : '咨询中'));
  $('caseStatus').textContent = data.needs_attention ? `${statusLabel} · 需关注` : statusLabel;
  $('request').textContent = caseRecord.user_request || (isCase ? '售后申请正在生成处理依据。' : '尚未创建售后申请，请先了解客户的具体诉求。');
  const appealReason = data.appeal_reason || draft.appeal_reason;
  if (appealReason) $('request').textContent += `\n补充申诉说明：${appealReason}`;
  $('facts').replaceChildren();
  addFact('订单号', facts.order_id || record.order_id);
  addFact('订单状态', statusNames[facts.order_status] || facts.order_status);
  addFact('支付金额', facts.payment_value == null ? null : `R$ ${facts.payment_value}`);
  addFact('配送延迟', facts.delay_days == null ? null : `${facts.delay_days} 天`);
  fillList('evidence', decision.customer_message_points || decision.evidence || []);
  fillList('policies', decision.policy_refs || []);
  $('approve').disabled = !data.has_pending;
  $('reject').disabled = !data.has_pending;
  if (isCase && !data.has_pending) {
    $('result').textContent = data.status === 'approving'
      ? '审核已通过，系统正在处理。'
      : '该申请已经处理完成，没有待恢复的审核流程。';
  }
  setStatus('');
}

async function loadCaseDetail(record, caseId) {
  try {
    const data = await api(`/review/cases/${encodeURIComponent(caseId)}`);
    renderReviewData(record, data, caseId);
    if (selected.cases) renderCasePicker(selected.cases);
    await renderConversation();
  } catch (error) {
    setStatus(error.message);
  }
}

async function openCase(record) {
  selected = {...record, caseId: record.case_id};
  $('detail').hidden = false;
  $('result').textContent = '';
  $('replyStatus').textContent = '';
  $('approve').disabled = true;
  $('reject').disabled = true;
  document.querySelectorAll('.queue-item').forEach(button => button.classList.toggle('selected', button.dataset.case === record.case_id));
  try {
    const listing = await api(`/review/sessions/${encodeURIComponent(record.session_id)}/cases`);
    selected.cases = listing.cases || [];
    renderCasePicker(selected.cases);
    await loadCaseDetail(selected, record.case_id);
  } catch (error) {
    setStatus(error.message);
  }
}

async function openConversation(record) {
  selected = record;
  $('detail').hidden = false;
  $('result').textContent = '';
  $('replyStatus').textContent = '';
  $('approve').disabled = true;
  $('reject').disabled = true;
  document.querySelectorAll('.queue-item').forEach(button => button.classList.toggle('selected', button.dataset.session === record.session_id));
  try {
    const listing = await api(`/review/sessions/${encodeURIComponent(record.session_id)}/cases`);
    const cases = listing.cases || [];
    selected.cases = cases;
    renderCasePicker(cases);
    if (cases.length) {
      const chosen = cases.find(item => item.has_pending) || cases[cases.length - 1];
      await loadCaseDetail(record, chosen.case_id);
      return;
    }
    const data = await api(`/review/sessions/${encodeURIComponent(record.session_id)}`);
    renderReviewData(record, data, null);
    await renderConversation();
  } catch (error) {
    setStatus(error.message);
  }
}

async function refreshQueue() {
  const data = await api('/review/cases');
  const cases = data.cases || [];
  $('queue').replaceChildren();
  $('count').textContent = cases.length;
  setStatus(cases.length ? '选择一个 Case 查看详情；需关注的 Case 已排在前面。' : '当前没有待审 Case。');
  let lastSession = null;
  for (const record of cases) {
    if (record.session_id !== lastSession) {
      const sessionLabel = document.createElement('div');
      sessionLabel.className = 'queue-session';
      sessionLabel.textContent = `会话 ${String(record.session_id).slice(0, 8)}`;
      $('queue').append(sessionLabel);
      lastSession = record.session_id;
    }
    const button = document.createElement('button');
    const meta = document.createElement('small');
    button.type = 'button';
    button.className = 'queue-item';
    if (record.needs_attention) button.classList.add('attention');
    button.dataset.case = record.case_id;
    const order = record.order_id ? ` · 订单 ${String(record.order_id).slice(-6)}` : '';
    button.textContent = `${names[record.action_type] || '售后'}${order}`;
    meta.textContent = `${reviewStatusNames[record.status] || record.status} · ${formatAge(record.age_seconds)}`;
    button.append(meta);
    button.onclick = () => openCase(record);
    $('queue').append(button);
  }
}

async function updateHandoff(human) {
  if (!selected) return;
  try {
    await api(`/review/sessions/${encodeURIComponent(selected.session_id)}/handoff`, {method: 'POST', body: JSON.stringify({human})});
    $('replyStatus').textContent = human ? '已接手会话，智能客服将暂停自动回复。' : '已恢复智能客服。';
    await renderConversation();
  } catch (error) {$('replyStatus').textContent = error.message;}
}

async function sendReply(event) {
  event.preventDefault();
  if (!selected) return;
  const content = $('reply').value.trim();
  if (!content) return;
  $('replySend').disabled = true;
  try {
    await api(`/review/sessions/${encodeURIComponent(selected.session_id)}/messages`, {method: 'POST', body: JSON.stringify({content})});
    $('reply').value = '';
    $('replyStatus').textContent = '已发送给客户。';
    await renderConversation();
  } catch (error) {$('replyStatus').textContent = error.message;} finally {$('replySend').disabled = false;}
}

async function suggestReply() {
  if (!selected) return;
  $('suggest').disabled = true;
  $('replyStatus').textContent = '正在生成建议…';
  try {
    const data = await api(`/review/sessions/${encodeURIComponent(selected.session_id)}/suggestion`, {method: 'POST'});
    $('reply').value = data.draft;
    $('attention').textContent = `${data.attention}${data.signals.length ? `；本轮用词：${data.signals.join('、')}` : ''}。${data.signal_method}`;
    $('replyStatus').textContent = '建议已填入草稿，请核对后发送。';
  } catch (error) {$('replyStatus').textContent = error.message;} finally {$('suggest').disabled = false;}
}

async function decide(action) {
  if (!selected) return;
  $('approve').disabled = true;
  $('reject').disabled = true;
  try {
    const path = selected.caseId
      ? `/review/cases/${encodeURIComponent(selected.caseId)}/${action}`
      : `/review/sessions/${encodeURIComponent(selected.session_id)}/${action}`;
    const data = await api(path, {method: 'POST', body: JSON.stringify({expected_attempt: selected.workflowAttempt, reviewer_id: 'local-reviewer', role: 'after_sales_operator', auth_scopes: ['after_sales:write']})});
    if (selected.caseId) await loadCaseDetail(selected, selected.caseId);
    $('result').textContent = data.answer || (action === 'approve' ? '已批准申请，系统将继续处理。' : '已驳回申请，客户会看到处理结果。');
    await refreshQueue();
    await renderMetrics();
    await renderShadow();
  } catch (error) {$('result').textContent = error.message;}
}

$('login').onsubmit = async event => {
  event.preventDefault();
  token = $('token').value.trim();
  try {
    await refreshQueue();
    await renderMetrics();
    await renderShadow();
    $('login').hidden = true;
    $('refresh').hidden = false;
  } catch (error) {setStatus(error.message);}
};
$('refresh').onclick = () => refreshQueue().catch(error => setStatus(error.message));
$('takeover').onclick = () => updateHandoff(true);
$('automatic').onclick = () => updateHandoff(false);
$('replyForm').onsubmit = sendReply;
$('suggest').onclick = suggestReply;
$('approve').onclick = () => decide('approve');
$('reject').onclick = () => decide('reject');

const replyOptions = document.createElement('div');
replyOptions.className = 'shortcuts';
for (const [label, content] of [
  ['确认收货', '请问您是否已经收到包裹？商品本身是否完好？'],
  ['了解退款原因', '我来协助您处理。请问您希望退款的具体原因是什么？'],
  ['补充问题描述', '为了准确核查，麻烦您说明商品出现了什么问题，以及发现问题的时间。'],
]) {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = label;
  button.onclick = () => {$('reply').value = content; $('reply').focus();};
  replyOptions.append(button);
}
$('replyForm').before(replyOptions);
setInterval(() => renderConversation(), 5000);
