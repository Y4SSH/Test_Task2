import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces
from sklearn.metrics import mean_squared_error

class SyntheticDataEnvironment(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, original_data: pd.DataFrame, context: str | None = None, feedback_bonus: float = 2.0):
        super().__init__()
        self.original_data = original_data.reset_index(drop=True)
        self.context = context
        self.feedback_bonus = feedback_bonus
        self.feature_stats = self._calculate_feature_stats()
        self.action_space = spaces.Box(low=0.5, high=2.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.current_step = 0
        self.max_steps = 100
        self._last_feedback_adjustment = 0.0

    def _calculate_feature_stats(self) -> dict:
        return {
            "price_mean": self.original_data["price"].mean(),
            "price_std": self.original_data["price"].std(ddof=0),
            "quantity_mean": self.original_data["quantity"].mean(),
            "quantity_std": self.original_data["quantity"].std(ddof=0),
            "category_dist": self.original_data["category"].value_counts(normalize=True).to_dict(),
            "user_type_dist": self.original_data["user_type"].value_counts(normalize=True).to_dict() if "user_type" in self.original_data.columns else {},
            "total_revenue_mean": (self.original_data["price"] * self.original_data["quantity"]).mean(),
        }

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_observation(), {}

    def _get_observation(self):
        return np.array([
            0.5, 0.5, 0.5, 0.5,
            0.5, 0.5,
            self.current_step / self.max_steps,
            0.5
        ], dtype=np.float32)

    def inject_feedback_adjustment(self, avg_label: float | None):
        if avg_label is None:
            self._last_feedback_adjustment = 0.0
        else:
            # avg_label in [-1, +1]; map to additive reward term
            self._last_feedback_adjustment = float(avg_label) * self.feedback_bonus

    def step(self, action):
        price_factor, quantity_factor, category_factor, seasonal_factor = action
        synthetic_data = self._generate_data(price_factor, quantity_factor, category_factor, seasonal_factor)
        reward = self._calculate_reward(synthetic_data) + self._last_feedback_adjustment
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {"synthetic_data": synthetic_data, "context": self.context, "feedback_reward": self._last_feedback_adjustment}
        return self._get_observation(), float(reward), terminated, truncated, info

    def _generate_data(self, price_factor, quantity_factor, category_factor, seasonal_factor) -> pd.DataFrame:
        n_samples = len(self.original_data)
        rows = []
        categories = list(self.feature_stats["category_dist"].keys())
        cat_probs = np.array(list(self.feature_stats["category_dist"].values()), dtype=float)
        cat_probs = cat_probs / cat_probs.sum() if cat_probs.sum() > 0 else np.ones_like(cat_probs) / len(cat_probs)

        # scenario heuristics
        hour_pool = range(0, 24)
        if self.context and "evening" in self.context.lower():
            hour_pool = range(17, 21)
        if self.context and "weekend" in self.context.lower():
            seasonal_factor = min(2.0, float(seasonal_factor) * 1.2)

        for _ in range(n_samples):
            category = np.random.choice(categories, p=(cat_probs * category_factor) / (cat_probs * category_factor).sum())
            base_price = np.random.normal(self.feature_stats["price_mean"], max(1e-6, self.feature_stats["price_std"]))
            price = max(1.0, base_price * price_factor * (0.8 + 0.4 * random.random()))
            base_quantity = np.random.normal(self.feature_stats["quantity_mean"], max(1e-6, self.feature_stats["quantity_std"]))
            quantity = max(1, int(base_quantity * quantity_factor))

            day_of_year = random.randint(1, 365)
            seasonal_multiplier = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365) * seasonal_factor
            price *= seasonal_multiplier

            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=365)
            timestamp = base_date + timedelta(days=random.randint(0, 365))
            hour = random.choice(list(hour_pool))
            timestamp = timestamp.replace(hour=hour)

            user_id = random.randint(1000, 9999)
            total_revenue = price * quantity

            rows.append({
                "user_id": user_id,
                "category": category,
                "price": round(float(price), 2),
                "quantity": int(quantity),
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "total_revenue": round(float(total_revenue), 2),
            })

        df = pd.DataFrame(rows)

        # realism constraints
        df = df[(df["price"] > 0) & (df["quantity"] > 0)]
        df["total_revenue"] = (df["price"] * df["quantity"]).round(2)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.loc[df["timestamp"] > now_str, "timestamp"] = now_str
        return df.reset_index(drop=True)

    def _calculate_reward(self, synthetic: pd.DataFrame) -> float:
        r = 0.0
        try:
            price_mse = mean_squared_error([self.feature_stats["price_mean"]], [synthetic["price"].mean()])
            r += (1.0 / (1.0 + price_mse)) * 0.25

            qty_mse = mean_squared_error([self.feature_stats["quantity_mean"]], [synthetic["quantity"].mean()])
            r += (1.0 / (1.0 + qty_mse)) * 0.2

            syn_cat = synthetic["category"].value_counts(normalize=True)
            cat_sim = sum(min(self.feature_stats["category_dist"].get(c, 0.0), syn_cat.get(c, 0.0)) for c in set(self.feature_stats["category_dist"]) | set(syn_cat))
            r += cat_sim * 0.25

            orig_rev = self.feature_stats["total_revenue_mean"]
            syn_rev = synthetic["total_revenue"].mean()
            r += (1.0 / (1.0 + abs(orig_rev - syn_rev) / max(1e-6, orig_rev))) * 0.25

            rej_rate = 1.0 - (len(synthetic) / max(1, len(self.original_data)))
            r += max(0.0, 1.0 - rej_rate) * 0.05
        except Exception:
            return -1.0
        return float(r)
