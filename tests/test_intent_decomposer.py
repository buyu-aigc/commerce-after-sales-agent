from app.intent.decomposer import decompose_business_message


def test_decompose_order_policy_escalation_without_duplicates() -> None:
    tasks = decompose_business_message(
        "查订单 203096f03d82e0dffbc41ebc2e2bcfb7 状态，另外退款政策是什么，另外我要升级人工"
    )
    assert [task.intent for task in tasks] == ["order_status", "policy", "escalation"]
    assert [task.side_effect for task in tasks] == [False, False, True]


def test_decompose_prefers_policy_for_cancellation_fee() -> None:
    tasks = decompose_business_message("what is the fee for canceling the contract?")
    assert [task.intent for task in tasks] == ["policy"]


def test_decompose_supports_common_connectives() -> None:
    tasks = decompose_business_message("查订单状态，并且说明退款政策，然后我要升级人工")
    assert [task.intent for task in tasks] == ["order_status", "policy", "escalation"]


def test_decompose_keeps_same_intent_for_different_orders() -> None:
    tasks = decompose_business_message(
        "查订单 203096f03d82e0dffbc41ebc2e2bcfb7 状态，"
        "另外查订单 4a90af3e85dd563884e2afeab1091394 状态"
    )
    assert [task.intent for task in tasks] == ["order_status", "order_status"]


def test_decompose_category_risk_as_qa() -> None:
    tasks = decompose_business_message("health beauty 类目有什么运营风险？")
    assert [task.intent for task in tasks] == ["qa"]


def test_decompose_after_sales_ops_decision_as_read_only_task() -> None:
    tasks = decompose_business_message("生成审核台优先处理队列，列出最需要人工跟进的类目和订单")
    assert [task.intent for task in tasks] == ["ops_decision"]
    assert tasks[0].side_effect is False


def test_decompose_refund_compensation_question_as_policy() -> None:
    tasks = decompose_business_message("退款补偿能不能直接承诺？")
    assert [task.intent for task in tasks] == ["policy"]


def test_decompose_chinese_status_as_order_status() -> None:
    tasks = decompose_business_message("帮我查一下订单 203096f03d82e0dffbc41ebc2e2bcfb7 现在是什么状态")
    assert [task.intent for task in tasks] == ["order_status"]


def test_decompose_infix_address_change_as_write_action() -> None:
    tasks = decompose_business_message(
        "请帮我修改订单 8aec3a066f732dd927ec8fef1752415b 的地址"
    )
    assert [task.intent for task in tasks] == ["escalation"]
    assert tasks[0].action_type == "change_address"


def test_decompose_followup_script_as_escalation() -> None:
    tasks = decompose_business_message(
        "客户给订单 203096f03d82e0dffbc41ebc2e2bcfb7 打了低分，生成一段客服跟进话术"
    )
    assert [task.intent for task in tasks] == ["escalation"]


def test_decompose_distinguishes_small_talk_and_ambiguous_support() -> None:
    assert [task.intent for task in decompose_business_message("你好")] == ["small_talk"]
    assert [task.intent for task in decompose_business_message("订单有问题")] == ["clarify"]
