import os
import matplotlib.pyplot as plt
from agent.engine import AgentSynth
import pandas as pd

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def run():
    ensure_dirs()
    eng = AgentSynth(feedback_db_path="logs/feedback.sqlite")

    print("Loading seed data...")
    original = eng.load_initial_data("data/sample_data.csv") if os.path.exists("data/sample_data.csv") else eng.load_initial_data()
    original.to_csv("data/sample_data.csv", index=False)

    print("Initializing environment and agent...")
    eng.initialize(original, context="evening_rush")

    print("Training...")
    eng.train(episodes=30, verbose=True)

    print("Generating synthetic data...")
    synth = eng.generate(n_samples=500)
    out_path = "data/generated.csv"
    eng.export(synth, out_path)
    print(f"Saved: {out_path}")

    print("Evaluating...")
    scores = eng.evaluate(original, synth)
    print("Scores:", scores)

    # simple training curve plot
    plt.figure(figsize=(8,5))
    plt.plot(eng.training_history)
    plt.title("Training Progress – Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/training_progress.png", dpi=200)
    print("Saved: data/training_progress.png")

if __name__ == "__main__":
    run()
