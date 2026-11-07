from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from app.agent.actions import AgentActions
from app.agent.state import AgentState


class AgentGraph:
    """Build and compile the Olist marketplace support graph."""

    def __init__(self, actions: AgentActions) -> None:
        self._actions = actions

    def build(self, checkpointer: BaseCheckpointSaver) -> StateGraph:
        """Graph topology:

        plan_tasks -> select_next_task -> extract_slots -> retrieve_context
        retrieve_context -> execute_read_task for read-only tasks
        retrieve_context -> build_after_sales_case for write-intent tasks
        build_after_sales_case -> await_confirmation for high-risk writes
        build_after_sales_case -> select_next_task after reject/clarify/auto-execute
        finalize_escalation -> select_next_task after HITL resume
        """
        graph: StateGraph = StateGraph(AgentState)

        graph.add_node("plan_tasks", self._actions.plan_tasks)
        graph.add_node("select_next_task", self._actions.select_next_task)
        graph.add_node("extract_slots", self._actions.extract_slots)
        graph.add_node("retrieve_context", self._actions.retrieve_context)
        graph.add_node("execute_read_task", self._actions.execute_read_task)
        graph.add_node("build_after_sales_case", self._actions.build_after_sales_case)
        graph.add_node("policy_check", self._actions.apply_policy_auto)
        graph.add_node("await_confirmation", self._actions.await_confirmation)
        graph.add_node("finalize_escalation", self._actions.finalize_escalation)
        graph.add_node("finalize_answer", self._actions.finalize_answer)

        graph.set_entry_point("plan_tasks")
        graph.add_edge("plan_tasks", "select_next_task")
        graph.add_conditional_edges(
            "select_next_task",
            self._route_after_select,
            {
                "run": "extract_slots",
                "finalize": "finalize_answer",
            },
        )
        graph.add_edge("extract_slots", "retrieve_context")
        graph.add_conditional_edges(
            "retrieve_context",
            self._route_after_context,
            {
                "after_sales": "build_after_sales_case",
                "read": "execute_read_task",
            },
        )
        graph.add_edge("execute_read_task", "select_next_task")
        graph.add_conditional_edges(
            "build_after_sales_case",
            self._route_after_after_sales,
            {
                "await_confirmation": "await_confirmation",
                "policy": "policy_check",
                "continue": "select_next_task",
            },
        )
        graph.add_edge("policy_check", "select_next_task")
        graph.add_edge("await_confirmation", "finalize_escalation")
        graph.add_edge("finalize_escalation", "select_next_task")
        graph.add_edge("finalize_answer", END)

        return graph.compile(checkpointer=checkpointer)

    def build_case(self, checkpointer: BaseCheckpointSaver):
        """Compile the isolated review workflow for one after-sales Case."""
        graph: StateGraph = StateGraph(AgentState)
        graph.add_node("await_confirmation", self._actions.await_confirmation)
        graph.add_node("finalize_escalation", self._actions.finalize_escalation)
        graph.set_entry_point("await_confirmation")
        graph.add_edge("await_confirmation", "finalize_escalation")
        graph.add_edge("finalize_escalation", END)
        return graph.compile(checkpointer=checkpointer)

    @staticmethod
    def _route_after_select(state: AgentState) -> str:
        if state.get("workflow_complete"):
            return "finalize"
        return "run"

    @staticmethod
    def _route_after_context(state: AgentState) -> str:
        task = state.get("current_task", {})
        if task.get("side_effect") or task.get("intent") == "escalation":
            return "after_sales"
        return "read"

    @staticmethod
    def _route_after_after_sales(state: AgentState) -> str:
        if state.get("policy_pending"):
            return "policy"
        if state.get("decouple_cases"):
            # V2: the case is dispatched to its own workflow; the conversation
            # graph must not stop at a human-review interrupt.
            return "continue"
        if state.get("pending_side_effect", {}).get("requires_confirmation"):
            return "await_confirmation"
        return "continue"
