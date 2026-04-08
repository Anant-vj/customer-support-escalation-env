# customer-support-escalation-env

An **OpenEnv-compliant** reinforcement-learning environment for training AI agents on customer support ticket escalation. The environment simulates a real-world support queue where an agent must classify 12 tickets by severity and decide the correct action — while navigating misleading tone signals, keyword traps, and cross-ticket context dependencies.

---

## Environment Description

The agent processes a fixed, ordered queue of 12 support tickets. For each ticket it must output:
- A **severity** (`low`, `medium`, `high`)
- An **action** (`reply`, `escalate`, `ignore`)

What makes this environment non-trivial:

| Challenge | Description |
|---|---|
| **Tone vs. severity mismatch** | Polite language may hide critical issues; aggressive language may hide trivial ones |
| **Keyword traps** | The word "URGENT" appears in a newsletter-unsubscribe (low severity) to trick classifiers |
| **Cross-ticket context** | Tickets T7 and T12 reference earlier tickets by ID; correct handling requires memory across turns |
| **Hard penalisation** | Ignoring a high-severity ticket or escalating a low-severity one incurs explicit reward penalties |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `current_ticket` | `dict` | The full ticket object including `id`, `text`, `tone`, `true_severity`, `true_action`, `trap_type` |
| `step` | `int` | Current step index (0-indexed) |
| `tickets_remaining` | `int` | Number of tickets left to process, including the current one |
| `processed_ids` | `List[str]` | Ordered list of ticket IDs already acted on |
| `message` | `str` | Human-readable status message (`"Episode started"`, `"Processed T1"`, `"Episode complete"`) |

---

## Action Space

| Field | Type | Options | Description |
|---|---|---|---|
| `severity` | `str` | `low`, `medium`, `high` | Assessed severity of the ticket |
| `action_type` | `str` | `reply`, `escalate`, `ignore` | Action to take on the ticket |
| `reasoning` | `str` (optional) | Any string | One-sentence explanation for the decision |

---

## Tasks

### 1. Basic Triage — `basic_triage` (Easy)

**Grader:** `grade_easy` — scores only the 4 non-trap tickets (T1, T5, T8, T11).
Each graded ticket: +0.5 for correct severity, +0.5 for correct action.

**What makes it hard:** Even "easy" tickets include T1 (unauthorized access with neutral tone) and T11 (billing error framed casually). Agents that rely purely on emotional tone will still fail.

---

### 2. Mixed Signals — `mixed_signals` (Medium)

**Grader:** `grade_medium` — scores all 12 tickets.
Each ticket: severity correct → 0.4 pts, action correct → 0.6 pts.

**What makes it hard:** All 4 trap types are present. Agents must resist aggressive-sounding trivial complaints and polite-sounding critical ones across the entire episode.

---

### 3. Context Trap — `context_trap` (Hard)

**Grader:** `grade_hard` — scores all 12 tickets with bonus weighting for traps.
- Non-trap tickets: severity=0.3, action=0.4 (max 0.7)
- Trap tickets: severity=0.2, action=0.3, trap_bonus=0.5 if both correct (max 1.0)
- **Context-dependent traps (T7, T12):** If the referenced prior ticket was not correctly escalated, the score for that ticket is capped at 0.3 regardless.

**What makes it hard:** Requires maintaining implicit memory that T7 is a follow-up to T3 and T12 is a follow-up to T1. A correct decision on T7 or T12 without having correctly handled T3 or T1 yields no full credit.

---

## Trap Types

| Trap Type | Description | Example |
|---|---|---|
| `polite_critical` | Polite, self-deprecating language disguises a high-severity issue | T3: *"Probably just a glitch!"* — about a double charge |
| `aggressive_trivial` | Aggressive all-caps language makes a trivial issue sound urgent | T2: *"THIS APP IS TRASH"* — about a profile theme reverting |
| `false_urgency` | Explicit "URGENT" keyword attached to a low-priority, automatable request | T4: *"URGENT: Please unsubscribe me from the newsletter"* |
| `context_dependent` | The correct classification depends on a prior ticket's resolution in the same episode | T7: Third charge — only escalatable in context of T3's double charge |

---

## Reward Function

All per-step rewards are clamped to `[0.0, 1.0]` before recording.

| Component | Condition | Value |
|---|---|---|
| Severity correct | `action.severity == ticket["true_severity"]` | +0.3 |
| Action correct | `action.action_type == ticket["true_action"]` | +0.4 |
| Context bonus | `context_dependent` trap AND prior ticket correctly escalated | +0.2 |
| Context penalty | `context_dependent` trap AND prior ticket NOT correctly escalated | −0.1 |
| Polite-critical bonus | `polite_critical` trap AND action correct | +0.1 |
| Aggressive-trivial bonus | `aggressive_trivial` trap AND action correct | +0.1 |
| False-urgency bonus | `false_urgency` trap AND action correct | +0.1 |
| Hard penalty: ignore critical | `action_type == "ignore"` AND `true_severity == "high"` | −0.3 |
| Hard penalty: escalate trivial | `action_type == "escalate"` AND `true_severity == "low"` | −0.2 |

---

## Setup Instructions

### Local Run (without Docker)

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Then in a second terminal, run the baseline agent:

```bash
export MODEL_NAME=gpt-4o
export HF_TOKEN=sk-...
export BASE_URL=http://localhost:7860
python inference.py
```

### Docker

```bash
docker build -t support-env .
docker run -p 7860:7860 support-env
```

To run inference against the container:

```bash
docker run --rm \
  -e MODEL_NAME=gpt-4o \
  -e HF_TOKEN=sk-... \
  -e BASE_URL=http://host.docker.internal:7860 \
  support-env python inference.py
```

### Hugging Face Spaces Deployment

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces) with **Docker** SDK.
2. Push all project files to the Space repository:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-org>/<space-name>
   git push space main
   ```
3. The Space will automatically build the Docker image and expose port 7860.
4. Set `HF_TOKEN`, `MODEL_NAME`, and `BASE_URL` as Space Secrets in the Settings tab.
5. Run `inference.py` locally, pointing `BASE_URL` at the Space URL.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MODEL_NAME` | **Yes** | — | LLM model identifier (e.g. `gpt-4o`, `claude-3-5-sonnet`) |
| `HF_TOKEN` | **Yes** | — | API key passed to the OpenAI-compatible client as `api_key` |
| `API_BASE_URL` | No | `http://localhost:7860` | Base URL for the OpenAI-compatible API endpoint |
| `BASE_URL` | No | `http://localhost:7860` | Base URL for the environment server's REST API |

---

## Baseline Scores

Scores below are from running `inference.py` with a frontier model (GPT-4o):

| Task | Score |
|---|---|
| `basic_triage` | ~0.72 |
| `mixed_signals` | ~0.61 |
| `context_trap` | ~0.48 |

> These scores reflect the difficulty gradient across the three tasks. The sharp drop from `mixed_signals` to `context_trap` is expected — it captures the agent's inability to maintain cross-ticket memory without explicit state management.
