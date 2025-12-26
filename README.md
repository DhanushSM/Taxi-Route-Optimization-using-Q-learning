```markdown
# Taxi Route Optimization with Reinforcement Learning (GUI)

This project is a self-contained Python application that demonstrates a Taxi Route Optimization system using tabular Q-Learning with a GUI built in Tkinter. The environment simulates a grid-based city where a taxi learns to pick up and drop off passengers efficiently while considering traffic congestion.

Features
- Gym-style environment (TaxiEnv) with state: taxi position, passenger location (including "in taxi"), destination, and traffic level.
- Actions: move up/down/left/right, pick up, drop off.
- Reward function:
  - Positive rewards for successful pickup and dropoff.
  - Negative rewards for extra movement, invalid actions, and traffic.
- Training using tabular Q-Learning with epsilon-greedy exploration.
- GUI (Tkinter) with:
  - Grid-based map visualization and real-time taxi animation.
  - Buttons: Start Training, Reset Environment, Test Trained Agent, Save / Load Q-table.
  - Toggle for traffic density (Low/Medium/High).
  - Real-time metrics: episode reward, steps, distance traveled.
  - Training visualization using a live-updating reward plot.

Requirements
- Python 3.8+
- Packages: numpy, matplotlib
  Install via:
    pip install numpy matplotlib

How to run
  python taxi_rl_gui.py

Usage
- Traffic Density: choose Low / Medium / High (affects movement penalty and congestion).
- Start Training: begins training in a background thread. The reward plot updates as episodes complete.
- Test Trained Agent: runs one episode with the current Q-table and animates taxi movement.
- Reset Environment: places taxi, passenger, and destination randomly.
- Save Q-table / Load Q-table: store or restore learned Q-values (files saved in numpy .npz format).

Files
- taxi_rl_gui.py — main Python program (contains env, agent, and GUI).
- README.md — this file.

Notes
- The implementation is intentionally kept compact and educational (tabular Q-learning).
- You can increase grid size, change reward magnitudes, or replace the agent with a DQN for more complex experiments.

```
