# Demo Marketplace Support Policy

This policy file is a demo business configuration used to show how long-form support
knowledge can be connected to the e-commerce customer support Agent. It is not part
of the public Olist dataset. Olist provides transaction, product, seller, payment,
delivery, and review records, but not the original platform's internal refund,
compensation, escalation, or customer-service FAQ documents.

In a production deployment, this file would be replaced by versioned internal policy
documents from the support, legal, risk, and operations teams. Transaction facts and
policy knowledge are intentionally separated: order state must come from business
systems, while support policy can come from markdown, a CMS, Confluence, Zendesk
articles, internal FAQ pages, or a vector/BM25 knowledge base.

## Order Status Response Policy

When a customer asks for order status, the Agent must query the order system before
answering. The response should include the current order status, purchase timestamp,
estimated delivery date, actual delivery date if available, customer state, payment
value, review score if available, and product category summary.

The Agent must not infer delivery status from natural language alone. If no complete
order id is present, ask the customer for the full order id. If the order id is not
found, say that the order was not found and ask the customer to verify the id.

## Delivery Delay Policy

An order is considered delayed when the actual customer delivery date is later than
the estimated delivery date. If the order has not been delivered yet, the Agent may
say that delivery is still pending, but it must not claim a final delay value unless
the tool returns one.

For delayed orders, the Agent should acknowledge the issue, mention the number of
delay days if available, and offer to open a follow-up case. The Agent should not
promise a refund, replacement, coupon, or compensation unless a downstream policy
tool or human operator confirms eligibility.

## Low Review Recovery Policy

Review scores of 1 or 2 are treated as low-review recovery cases. The Agent should
summarize the order facts, acknowledge the customer's dissatisfaction, and prepare a
support follow-up message. Low-review recovery should focus on empathy and next-step
verification, not automatic compensation.

If the review is low but the order is otherwise delivered on time, the Agent should
still offer a service-recovery case because the complaint may relate to product
quality, seller behavior, missing items, packaging, or post-delivery support.

## Cancellation Follow-Up Policy

Canceled orders require special care because the customer's desired next step may be
ambiguous. The Agent should first confirm whether the customer wants an explanation,
a reorder recommendation, or a support case. Do not create a case for a canceled
order unless the user explicitly asks for follow-up or confirms the proposed case.

The Agent must not describe a canceled order as delivered. If an order is canceled
and has payment records, the Agent may mention that payment facts exist but should
not claim refund completion without a refund tool.

## Compensation Boundary Policy

Compensation, coupons, refunds, replacements, manual credits, and seller penalties
are side-effect actions. The Agent may draft a recommendation, but it must not execute
these actions without human confirmation and a policy/tool response confirming
eligibility.

When a user asks for compensation directly, the Agent should say that it can prepare
a case for review. It should collect or verify order id, status, delay information,
review score, and customer complaint context before proposing any next step.

## Invoice Request Policy

Invoice requests require the order id, customer identity or tax profile, invoice
title, billing identifier, and contact email. The Agent may explain what information
is needed and prepare an invoice request draft, but it must not claim that an invoice
has been issued until the invoice tool returns a confirmed result.

If the customer asks for an invoice without an order id, ask for the full order id
first. If the request includes an order id and asks to submit the invoice request,
route it as a side-effect escalation and require human confirmation before calling
the invoice tool.

## Address Change Policy

Shipping address changes are allowed only before the downstream order or logistics
system marks the shipment as locked, shipped, or delivered. The Agent should first
check the order status, explain that address changes may be unavailable after
shipment, and collect the new recipient name, phone number, and complete delivery
address before it creates a change request.

Changing an address modifies fulfillment data. The Agent may submit a low-risk
address-change request only when the order is still pre-shipment, the payment is
below the low-risk threshold, the new recipient details are complete, and no risk
signal is present. Other address-change requests require human confirmation before
calling an address-change tool. If the order has already been delivered, the Agent
should not attempt to change the address and should offer a support follow-up case
instead.

## Human Approval Policy

Refunds, compensation, invoices, complaints, high-value order changes, and any
request with incomplete facts or risk signals must pass through human-in-the-loop
confirmation. The Agent presents the reason, draft message, and intended action,
then pauses for an operator decision. A narrow low-risk gate may auto-submit a
pre-shipment cancellation or address-change request after deterministic fact,
policy, and verifier checks; the durable audit record must still be written.

Accepted confirmations include clear affirmative user responses such as "yes",
"confirm", "create", "确认", or "创建". Ambiguous replies should be treated as not
confirmed, and the Agent should ask a clarifying question instead of executing the
side effect.

## Customer Message Style

Customer-facing messages should be concise, concrete, and empathetic. The Agent
should include facts returned by tools: order status, product category, delay days,
review score, and the reason for follow-up. The tone should acknowledge the issue
without over-promising.

Avoid internal language such as "tool call", "RAG", "policy chunk", "checkpoint",
or "workflow node" in customer-facing responses. Use plain language and focus on
what the support team will verify next.

## Tool Boundary Policy

Order facts must come from `get_order_status`. Category-level operational insight
must come from `search_category_risk`. Escalation drafts must come from
`draft_escalation` or an equivalent support-case tool. The Agent should not invent
order status, payment value, review score, delay days, or case id.

If a tool returns an error or missing parameter message, the Agent should repair the
argument if possible. If repair is not possible, it should ask the user for the
missing field rather than repeatedly calling the same invalid tool.

## Category Operations Policy

Category-level answers are operational summaries, not guarantees about a specific
order. The Agent may discuss delay rate, low-review rate, cancellation rate, average
review score, average payment value, and sample order ids for debugging. For exact
facts about a specific order, the Agent must use order status lookup.

Category insights should be framed as "this category has elevated operational risk"
when delay or low-review rates are high. The Agent should avoid blaming customers or
sellers without additional evidence.

## Audit and Trace Policy

Every support task should be traceable. A production system should record user
message, routed skill, extracted slots, retrieved policy sections, retrieved category
insights, tool calls, tool arguments, tool outputs, human approval decision, final
answer, and created case id.

Audit traces are used for debugging, compliance review, offline evaluation, cost
analysis, and improving prompts or retrieval. Sensitive customer data should be
redacted or access-controlled according to tenant and role permissions.
