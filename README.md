# Test_Task2
# AgentSynth: RL-based Synthetic Data Generator

## Overview
AgentSynth is a reinforcement learning-based synthetic data generator.  
It creates **realistic, structured datasets** such as user purchases, categories, and revenue, while preserving statistical properties of the original data.

This project combines:
- A **custom Gymnasium environment** for synthetic data generation
- A **Q-learning agent** for training and optimization
- **Validation & visualization tools** to compare real vs synthetic data
- An optional **Streamlit web UI** for interactive exploration

---

## Features
- ✅ Generates synthetic e-commerce datasets (user, category, price, quantity, timestamp, revenue)  
- ✅ Reinforcement Learning loop with reward feedback  
- ✅ Validation against original dataset (mean, std, distributions)  
- ✅ Visualization: price, quantity, category, revenue, and user type comparisons  
- ✅ Export results to CSV  
- ✅ Interactive Streamlit dashboard  

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/AgentSynth.git
cd AgentSynth
