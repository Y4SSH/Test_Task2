import pandas as pd
import numpy as np
from datetime import datetime
from agent.environment import SyntheticDataEnvironment
from rl.q_agent import QLearningAgent
from utils.validation import score_distributions
from utils.storage import FeedbackStore

class AgentSynth:
    def __init__(self, feedback_db_path: str = "logs/feedback.sqlite"):
        self.env = None
        self.agent = None
        self.training_history = []
        self.feedback = FeedbackStore(feedback_db_path)

    def load_initial_data(self, file_path: str | None = None) -> pd.DataFrame:
        if file_path:
            return pd.read_csv(file_path)
        # fallback sample
        rng = np.random.default_rng(42)
        categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]
        user_types = ["regular", "premium", "guest"]
        rows = []
        for i in range(200):
            c = rng.choice(categories, p=[0.3, 0.25, 0.2, 0.15, 0.1])
            u = rng.choice(user_types, p=[0.6, 0.3, 0.1])
            if c == "Electronics":
                price = max(5, rng.normal(150, 50))
                qty = rng.poisson(1) + 1
            elif c == "Clothing":
                price = max(5, rng.normal(40, 15))
                qty = rng.poisson(2) + 1
            elif c == "Books":
                price = max(5, rng.normal(20, 5))
                qty = rng.poisson(1) + 1
            elif c == "Home":
                price = max(5, rng.normal(80, 30))
                qty = rng.poisson(1) + 1
            else:
                price = max(5, rng.normal(60, 25))
                qty = rng.poisson(1) + 1
            rows.append({
                "user_id": 1000 + i,
                "user_type": u,
                "category": c,
                "price": round(float(price), 2),
                "quantity": int(qty),
            })
        df = pd.DataFrame(rows)
        df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df["total_revenue"] = (df["price"] * df["quantity"]).round(2)
        return df

    def initialize(self, initial_data: pd.DataFrame, context: str | None = None):
        self.env = SyntheticDataEnvironment(initial_data, context=context)
        self.agent = QLearningAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
        )

    def _feedback_to_env(self):
        avg_label = self.feedback.get_recent_avg_label(window=200)
        self.env.inject_feedback_adjustment(avg_label)

    def train(self, episodes: int = 30, verbose: bool = True):
        for ep in range(episodes):
            state, _ = self.env.reset()
            total_r = 0.0
            self._feedback_to_env()
            for _ in range(self.env.max_steps):
                action = self.agent.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.agent.update(state, action, reward, next_state)
                state = next_state
                total_r += reward
                if terminated or truncated:
                    break
            self.training_history.append(total_r)
            if verbose and ep % 10 == 0:
                print(f"Episode {ep:03d} | Reward: {total_r:.4f} | Eps: {self.agent.epsilon:.4f}")

    def generate(self, n_samples: int = 500) -> pd.DataFrame:
        action = self.agent.best_action
        state, _ = self.env.reset()
        _, _, _, _, info = self.env.step(action)
        base = info["synthetic_data"]
        batches = []
        reps = max(1, n_samples // max(1, len(base)) + 1)
        for _ in range(reps):
            varied = np.clip(action + np.random.normal(0, 0.05, size=len(action)), 0.5, 2.0)
            _, _, _, _, info = self.env.step(varied.astype(np.float32))
            batches.append(info["synthetic_data"])
        combined = pd.concat(batches, ignore_index=True)
        out = combined.sample(n=min(n_samples, len(combined)), replace=False).reset_index(drop=True)
        out["scenario"] = self.env.context or "default"
        out["gen_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return out

    def evaluate(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
        return score_distributions(original, synthetic)

    def export(self, df: pd.DataFrame, path: str) -> str:
        df.to_csv(path, index=False)
        return path
