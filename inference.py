import os
import json
import requests
from typing import List
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")

MAX_STEPS = 12
TEMPERATURE = 0.2
MAX_TOKENS = 256
TASKS = ["basic_triage", "mixed_signals", "context_trap"]

SYSTEM_PROMPT = """You are an AI agent managing customer support tickets.

For each step, you will receive a support ticket.
You must classify it and decide the correct action.

severity options: low, medium, high
action_type options: reply, escalate, ignore

Rules:
- High severity: fraud, unauthorized access, payment issues, security
- Medium severity: account access problems, repeated failures, billing errors
- Low severity: UI complaints, feature requests, unsubscribe requests

Important:
- Tone is misleading. Polite language does not mean low severity.
- Aggressive language does not mean high severity.
- Some tickets reference earlier tickets by ID — treat repeated or escalating issues as higher severity.
- "URGENT" in a newsletter unsubscribe is not urgent.

Respond ONLY with valid JSON. No extra text. Schema:
{
  "severity": "low|medium|high",
  "action_type": "reply|escalate|ignore",
  "reasoning": "one sentence"
}"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def run_task(task: str, client: OpenAI) -> float:
    log_start(task=task, env="customer-support-escalation-env", model=MODEL_NAME)

    reset_resp = requests.post(f"{BASE_URL}/reset?task={task}")
    reset_resp.raise_for_status()
    observation = reset_resp.json()

    rewards: List[float] = []
    steps_taken = 0
    done = False
    last_reward = 0.0

    while not done and steps_taken < MAX_STEPS:
        ticket_text = observation.get("current_ticket", {}).get("text", "")
        step_num = observation.get("step", steps_taken)

        user_prompt = (
            f"Step {step_num} | Last reward: {last_reward:.2f}\n\n"
            f"Ticket:\n{ticket_text}"
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            severity = parsed["severity"]
            action_type = parsed["action_type"]
            reasoning = parsed.get("reasoning", "")
        except Exception:
            severity = "low"
            action_type = "reply"
            reasoning = "parse error"

        action_payload = {
            "severity": severity,
            "action_type": action_type,
            "reasoning": reasoning,
        }

        step_resp = requests.post(f"{BASE_URL}/step", json=action_payload)
        step_resp.raise_for_status()
        step_data = step_resp.json()

        observation = step_data["observation"]
        reward_val = step_data["reward"]["value"]
        done = step_data["done"]

        rewards.append(reward_val)
        last_reward = reward_val
        log_step(
            step=steps_taken,
            action=action_type,
            reward=reward_val,
            done=done,
            error=None,
        )
        steps_taken += 1

    score = sum(rewards) / len(rewards) if rewards else 0.0
    success = score >= 0.5
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    results = {}

    for task in TASKS:
        try:
            score = run_task(task, client)
            results[task] = score
        except Exception:
            print(
                "[END] success=false steps=0 score=0.000 rewards=",
                flush=True,
            )
            results[task] = 0.0

    avg = sum(results.values()) / len(results) if results else 0.0
    print("=== BASELINE RESULTS ===", flush=True)
    print(f"basic_triage:   {results.get('basic_triage', 0.0):.3f}", flush=True)
    print(f"mixed_signals:  {results.get('mixed_signals', 0.0):.3f}", flush=True)
    print(f"context_trap:   {results.get('context_trap', 0.0):.3f}", flush=True)
    print(f"average:        {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
