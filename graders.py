from typing import List


def grade_easy(trajectory: List[dict]) -> float:
    """Score only on tickets with trap_type=None (T1, T5, T8, T11)."""
    scores = []
    for item in trajectory:
        ticket = item["ticket"]
        action = item["action"]
        if ticket.get("trap_type") is not None:
            continue
        score = 0.0
        if action["severity"] == ticket["true_severity"]:
            score += 0.5
        if action["action_type"] == ticket["true_action"]:
            score += 0.5
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def grade_medium(trajectory: List[dict]) -> float:
    """Score on all tickets with partial credit: severity=0.4, action=0.6."""
    scores = []
    for item in trajectory:
        ticket = item["ticket"]
        action = item["action"]
        score = 0.0
        if action["severity"] == ticket["true_severity"]:
            score += 0.4
        if action["action_type"] == ticket["true_action"]:
            score += 0.6
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def grade_hard(trajectory: List[dict]) -> float:
    """
    Score on all tickets with extra weight on traps.
    Non-trap: severity=0.3, action=0.4 (max 0.7)
    Trap: severity=0.2, action=0.3, trap_bonus=0.5 if both correct (max 1.0)
    Context-dependent traps: cap at 0.3 if prior referenced ticket not correctly escalated.
    """
    scores = []

    for item in trajectory:
        ticket = item["ticket"]
        action = item["action"]
        trap_type = ticket.get("trap_type")
        severity_correct = action["severity"] == ticket["true_severity"]
        action_correct = action["action_type"] == ticket["true_action"]

        if trap_type is None:
            score = 0.0
            if severity_correct:
                score += 0.3
            if action_correct:
                score += 0.4
        else:
            score = 0.0
            if severity_correct:
                score += 0.2
            if action_correct:
                score += 0.3

            # trap_bonus only if both correct
            if severity_correct and action_correct:
                score += 0.5

            # context_dependent cap check
            if trap_type == "context_dependent":
                prior_escalated = False
                if ticket["id"] == "T7":
                    prior = next(
                        (t for t in trajectory if t["ticket"]["id"] == "T3"), None
                    )
                    if prior and prior["action"]["action_type"] == "escalate":
                        prior_escalated = True
                elif ticket["id"] == "T12":
                    prior = next(
                        (t for t in trajectory if t["ticket"]["id"] == "T1"), None
                    )
                    if prior and prior["action"]["action_type"] == "escalate":
                        prior_escalated = True
                if not prior_escalated:
                    score = min(score, 0.3)

        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


GRADERS = {
    "basic_triage": grade_easy,
    "mixed_signals": grade_medium,
    "context_trap": grade_hard,
}
