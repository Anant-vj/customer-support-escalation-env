from fastapi import FastAPI, Query
from models import Action, Observation, Reward
from env import SupportEnv

app = FastAPI(
    title="Customer Support Escalation Env",
    description="OpenEnv-compliant AI agent training environment for support ticket triage.",
    version="1.0.0",
)

env = SupportEnv()


@app.post("/reset")
def reset(task: str = Query(default="basic_triage")) -> dict:
    obs: Observation = env.reset(task)
    return obs.dict()


@app.post("/step")
def step(action: Action) -> dict:
    obs, reward, done, info = env.step_env(action)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict:
    return env.state()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
