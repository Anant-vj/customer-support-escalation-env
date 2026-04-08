from typing import Tuple
from tickets import TICKETS
from models import Action, Observation, Reward


class SupportEnv:
    def __init__(self):
        self.tickets = TICKETS
        self.task: str = "basic_triage"
        self.step: int = 0
        self.trajectory: list = []
        self.done: bool = False

    def reset(self, task: str = "basic_triage") -> Observation:
        self.task = task
        self.step = 0
        self.trajectory = []
        self.done = False
        return Observation(
            current_ticket=self.tickets[0],
            step=0,
            tickets_remaining=len(self.tickets),
            processed_ids=[],
            message="Episode started",
        )

    def step_env(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        ticket = self.tickets[self.step]

        # ── Reward computation ──────────────────────────────────────────────
        reward = 0.0
        severity_correct = action.severity == ticket["true_severity"]
        action_correct = action.action_type == ticket["true_action"]
        trap_type = ticket.get("trap_type")

        if severity_correct:
            reward += 0.3
        if action_correct:
            reward += 0.4

        # Context trap bonus: T7 and T12
        if trap_type == "context_dependent":
            context_correct = False
            if ticket["id"] == "T7":
                prior = next(
                    (t for t in self.trajectory if t["ticket"]["id"] == "T3"), None
                )
                if prior and prior["action"]["action_type"] == "escalate":
                    context_correct = True
            if ticket["id"] == "T12":
                prior = next(
                    (t for t in self.trajectory if t["ticket"]["id"] == "T1"), None
                )
                if prior and prior["action"]["action_type"] == "escalate":
                    context_correct = True
            if context_correct:
                reward += 0.2
            else:
                reward -= 0.1

        if trap_type == "polite_critical" and action_correct:
            reward += 0.1
        if trap_type == "aggressive_trivial" and action_correct:
            reward += 0.1
        if trap_type == "false_urgency" and action_correct:
            reward += 0.1

        # Hard penalties
        if action.action_type == "ignore" and ticket["true_severity"] == "high":
            reward -= 0.3
        if action.action_type == "escalate" and ticket["true_severity"] == "low":
            reward -= 0.2

        reward = max(0.0, min(1.0, reward))
        # ───────────────────────────────────────────────────────────────────

        self.trajectory.append(
            {"ticket": ticket, "action": action.dict(), "reward": reward}
        )
        self.step += 1
        self.done = self.step >= len(self.tickets)

        processed_ids = [t["ticket"]["id"] for t in self.trajectory]

        if not self.done:
            obs = Observation(
                current_ticket=self.tickets[self.step],
                step=self.step,
                tickets_remaining=len(self.tickets) - self.step,
                processed_ids=processed_ids,
                message=f"Processed {ticket['id']}",
            )
        else:
            obs = Observation(
                current_ticket=ticket,
                step=self.step,
                tickets_remaining=0,
                processed_ids=processed_ids,
                message="Episode complete",
            )

        reward_reason_parts = []
        if severity_correct:
            reward_reason_parts.append("severity correct")
        if action_correct:
            reward_reason_parts.append("action correct")
        if not reward_reason_parts:
            reward_reason_parts.append("no base credit")
        reward_reason = "; ".join(reward_reason_parts)

        reward_obj = Reward(
            value=reward,
            reason=reward_reason,
            partial_credit=(reward > 0.0 and not (severity_correct and action_correct)),
        )

        return (obs, reward_obj, self.done, {"step": self.step})

    def state(self) -> dict:
        return {
            "current_task": self.task,
            "step": self.step,
            "tickets_remaining": len(self.tickets) - self.step,
            "episode_score": self._episode_score(),
        }

    def _episode_score(self) -> float:
        if not self.trajectory:
            return 0.0
        return sum(t["reward"] for t in self.trajectory) / len(self.trajectory)
