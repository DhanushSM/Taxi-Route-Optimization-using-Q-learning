#!/usr/bin/env python3


import random
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

# Matplotlib for plotting training progress inside Tkinter
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ----- Environment -----

class TaxiEnv:
    """
    Simple grid-based taxi environment.

    State:
      - taxi position (x, y) on grid W x H
      - passenger location index: one of predefined locations or 'in_taxi' (encoded as len(locs))
      - destination index: one of predefined locations
      - traffic_level: 0 (low), 1 (medium), 2 (high)

    Actions (6):
      0 - up
      1 - down
      2 - left
      3 - right
      4 - pickup
      5 - dropoff
    """

    def __init__(self, width=5, height=5, locations=None, traffic_level=0, max_steps=200):
        self.width = width
        self.height = height
        # Default passenger/destination candidate locations (corners)
        if locations is None:
            self.locs = [(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)]
        else:
            self.locs = list(locations)
        self.num_locs = len(self.locs)
        self.in_taxi_idx = self.num_locs
        self.action_space_n = 6
        self.traffic_level = traffic_level  # 0,1,2
        self.max_steps = max_steps

        # state variables
        self.taxi_pos = (0, 0)
        self.pass_idx = 0
        self.dest_idx = 1
        self.steps = 0
        self.distance = 0  # movement count
        self.total_reward = 0

        self.reset()

    def seed(self, s):
        random.seed(s)
        np.random.seed(s)

    def reset(self, traffic_level=None):
        """Randomize taxi, passenger, and destination. Optionally set traffic."""
        if traffic_level is not None:
            self.traffic_level = traffic_level
        # random taxi position anywhere
        self.taxi_pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        # random passenger and destination distinct
        self.pass_idx = random.randint(0, self.num_locs - 1)
        self.dest_idx = random.randint(0, self.num_locs - 1)
        while self.dest_idx == self.pass_idx:
            self.dest_idx = random.randint(0, self.num_locs - 1)
        self.steps = 0
        self.distance = 0
        self.total_reward = 0
        return self._get_obs()

    def _get_obs(self):
        # observation as tuple (taxi_x, taxi_y, pass_idx, dest_idx, traffic_level)
        return (self.taxi_pos[0], self.taxi_pos[1], self.pass_idx, self.dest_idx, self.traffic_level)

    def _at_loc(self, loc_idx):
        return self.taxi_pos == self.locs[loc_idx]

    def _move(self, dx, dy):
        x, y = self.taxi_pos
        nx = min(max(x + dx, 0), self.width - 1)
        ny = min(max(y + dy, 0), self.height - 1)
        moved = (nx != x) or (ny != y)
        self.taxi_pos = (nx, ny)
        if moved:
            self.distance += 1

    def step(self, action):
        """
        Executes action and returns (obs, reward, done, info)
        """
        self.steps += 1
        reward = 0
        done = False
        info = {}

        # Movement actions
        if action == 0:  # up
            prev = self.taxi_pos
            self._move(0, -1)
            if self.taxi_pos == prev:
                # hit boundary - small penalty
                reward -= 1
            else:
                reward -= 1  # step cost
                # traffic penalty proportional to level if moving
                reward -= self.traffic_level * 0.5
                # small chance movement is slowed/blocked by congestion (stays in place)
                # we simulate by additional penalty if traffic is high
                if self.traffic_level == 2 and random.random() < 0.15:
                    # stalled: penalize
                    reward -= 2
        elif action == 1:  # down
            prev = self.taxi_pos
            self._move(0, 1)
            if self.taxi_pos == prev:
                reward -= 1
            else:
                reward -= 1
                reward -= self.traffic_level * 0.5
                if self.traffic_level == 2 and random.random() < 0.15:
                    reward -= 2
        elif action == 2:  # left
            prev = self.taxi_pos
            self._move(-1, 0)
            if self.taxi_pos == prev:
                reward -= 1
            else:
                reward -= 1
                reward -= self.traffic_level * 0.5
                if self.traffic_level == 2 and random.random() < 0.15:
                    reward -= 2
        elif action == 3:  # right
            prev = self.taxi_pos
            self._move(1, 0)
            if self.taxi_pos == prev:
                reward -= 1
            else:
                reward -= 1
                reward -= self.traffic_level * 0.5
                if self.traffic_level == 2 and random.random() < 0.15:
                    reward -= 2

        elif action == 4:  # pickup
            # valid if passenger at taxi position
            if self.pass_idx < self.num_locs and self._at_loc(self.pass_idx):
                # pick up
                self.pass_idx = self.in_taxi_idx
                reward += 10.0  # reward for successful pickup
            else:
                reward -= 10.0  # invalid pickup attempt
        elif action == 5:  # dropoff
            # valid only when passenger in taxi and at destination
            if self.pass_idx == self.in_taxi_idx and self._at_loc(self.dest_idx):
                reward += 20.0  # successful drop-off
                done = True
            else:
                reward -= 10.0  # invalid drop-off
        else:
            reward -= 1.0  # unknown action

        # End if too many steps
        if self.steps >= self.max_steps:
            done = True

        self.total_reward += reward
        return self._get_obs(), reward, done, info

    def render_ascii(self):
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        # mark locations
        for i, (x, y) in enumerate(self.locs):
            char = chr(ord('A') + i)
            grid[y][x] = char
        tx, ty = self.taxi_pos
        grid[ty][tx] = 'T'
        lines = [''.join(row) for row in grid]
        print('\n'.join(lines))
        print(f"pass_idx={self.pass_idx} dest_idx={self.dest_idx} traffic={self.traffic_level}")

    def encode_state(self, obs):
        """Encode state tuple into single integer for table indexing."""
        tx, ty, pass_idx, dest_idx, traffic = obs
        pos_idx = ty * self.width + tx
        # dims: pos (width*height) * (num_locs+1) * num_locs * traffic_levels
        s = pos_idx
        s *= (self.num_locs + 1)
        s += pass_idx
        s *= (self.num_locs)
        s += dest_idx
        s *= 3  # traffic levels assumed 0..2
        s += traffic
        return int(s)

    def n_states(self):
        return (self.width * self.height) * (self.num_locs + 1) * (self.num_locs) * 3

    def n_actions(self):
        return self.action_space_n


# ----- Agent -----

class QLearningAgent:
    """
    Tabular Q-learning agent for discrete state and action spaces.
    """

    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=1.0, eps_min=0.05, eps_decay=0.9995):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        # Q-table
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

    def act(self, state_idx, greedy=False):
        if (not greedy) and (random.random() < self.epsilon):
            return random.randint(0, self.n_actions - 1)
        else:
            # break ties randomly
            q = self.Q[state_idx]
            maxv = q.max()
            choices = np.where(np.isclose(q, maxv))[0]
            return int(np.random.choice(choices))

    def learn(self, s_idx, a, r, s2_idx, done):
        q = self.Q[s_idx, a]
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q[s2_idx])
        self.Q[s_idx, a] += self.lr * (target - q)
        # decay epsilon
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay
            if self.epsilon < self.eps_min:
                self.epsilon = self.eps_min

    def save(self, path):
        np.savez(path, Q=self.Q, epsilon=self.epsilon)

    def load(self, path):
        data = np.load(path + '.npz')
        self.Q = data['Q']
        if 'epsilon' in data:
            self.epsilon = float(data['epsilon'])


# ----- GUI -----

class TaxiGUI:
    """
    Tkinter-based GUI that embeds environment visualization and controls.
    """

    CELL_SIZE = 80
    MARGIN = 20

    def __init__(self, master):
        self.master = master
        master.title("Taxi Route Optimization - Q-Learning")

        # Environment parameters
        self.grid_w = 5
        self.grid_h = 5
        self.env = TaxiEnv(width=self.grid_w, height=self.grid_h, traffic_level=0)
        self.agent = QLearningAgent(n_states=self.env.n_states(), n_actions=self.env.n_actions(),
                                    lr=0.1, gamma=0.99, epsilon=1.0,
                                    eps_min=0.05, eps_decay=0.999)
        self.training_thread = None
        self.training = False
        self.train_lock = threading.Lock()

        # Metrics
        self.episode_rewards = []
        self.avg_rewards = []
        self.current_episode = 0

        # Build UI
        self._build_controls()
        self._build_canvas()
        self._build_plot()
        self._update_canvas()

        # schedule periodic UI update (to show training progress)
        self.master.after(200, self._periodic_update)

    def _build_controls(self):
        frm = tk.Frame(self.master)
        frm.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # Traffic density option
        tk.Label(frm, text="Traffic Density:").pack(anchor=tk.W)
        self.traffic_var = tk.IntVar(value=0)
        traffic_frame = tk.Frame(frm)
        traffic_frame.pack(anchor=tk.W)
        tk.Radiobutton(traffic_frame, text="Low", variable=self.traffic_var, value=0,
                       command=self._on_traffic_change).pack(side=tk.LEFT)
        tk.Radiobutton(traffic_frame, text="Medium", variable=self.traffic_var, value=1,
                       command=self._on_traffic_change).pack(side=tk.LEFT)
        tk.Radiobutton(traffic_frame, text="High", variable=self.traffic_var, value=2,
                       command=self._on_traffic_change).pack(side=tk.LEFT)

        # Buttons
        btn_frame = tk.Frame(frm)
        btn_frame.pack(pady=10, anchor=tk.W)

        self.btn_start = tk.Button(btn_frame, text="Start Training", command=self._on_start_training, width=16)
        self.btn_start.pack(pady=3)
        tk.Button(btn_frame, text="Reset Environment", command=self._on_reset, width=16).pack(pady=3)
        tk.Button(btn_frame, text="Test Trained Agent", command=self._on_test_agent, width=16).pack(pady=3)

        # Save/Load
        tk.Button(btn_frame, text="Save Q-table", command=self._on_save_q, width=16).pack(pady=3)
        tk.Button(btn_frame, text="Load Q-table", command=self._on_load_q, width=16).pack(pady=3)

        # Training controls
        tk.Label(frm, text="Training Episodes:").pack(anchor=tk.W, pady=(10, 0))
        self.episodes_var = tk.IntVar(value=400)
        tk.Entry(frm, textvariable=self.episodes_var, width=8).pack(anchor=tk.W)

        tk.Label(frm, text="Max Steps / Episode:").pack(anchor=tk.W, pady=(10, 0))
        self.maxsteps_var = tk.IntVar(value=200)
        tk.Entry(frm, textvariable=self.maxsteps_var, width=8).pack(anchor=tk.W)

        # Agent parameters
        tk.Label(frm, text="Learning Rate (alpha):").pack(anchor=tk.W, pady=(10, 0))
        self.lr_var = tk.DoubleVar(value=self.agent.lr)
        tk.Entry(frm, textvariable=self.lr_var, width=8).pack(anchor=tk.W)

        tk.Label(frm, text="Discount (gamma):").pack(anchor=tk.W, pady=(10, 0))
        self.gamma_var = tk.DoubleVar(value=self.agent.gamma)
        tk.Entry(frm, textvariable=self.gamma_var, width=8).pack(anchor=tk.W)

        # Metrics display
        tk.Label(frm, text="Metrics:").pack(anchor=tk.W, pady=(10, 0))
        self.metrics_text = tk.Text(frm, width=28, height=10, state=tk.DISABLED)
        self.metrics_text.pack(pady=(2, 0))

    def _build_canvas(self):
        # Canvas to draw grid and taxi
        canvas_w = self.grid_w * self.CELL_SIZE + 2 * self.MARGIN
        canvas_h = self.grid_h * self.CELL_SIZE + 2 * self.MARGIN
        self.canvas = tk.Canvas(self.master, width=canvas_w, height=canvas_h, bg="white")
        self.canvas.pack(side=tk.TOP, padx=8, pady=8)

    def _build_plot(self):
        # Matplotlib figure for rewards
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Episode Reward")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Total Reward")
        self.line_avg, = self.ax.plot([], [], label='Avg reward')
        self.line_raw, = self.ax.plot([], [], alpha=0.3, label='Episode reward')
        self.ax.legend()
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_fig.get_tk_widget().pack(side=tk.BOTTOM, padx=8, pady=8)

    def _on_traffic_change(self):
        lvl = self.traffic_var.get()
        self.env.traffic_level = lvl
        # Do not reset agent; just update environment traffic for subsequent runs
        self._update_canvas()

    def _on_reset(self):
        self.env.reset(traffic_level=self.traffic_var.get())
        self._update_canvas()
        self._log_metrics("Environment reset.")

    def _on_start_training(self):
        if self.training:
            messagebox.showinfo("Training", "Training is already running.")
            return

        # Update env and agent parameters
        try:
            episodes = int(self.episodes_var.get())
            max_steps = int(self.maxsteps_var.get())
            self.env.max_steps = max_steps
            self.agent.lr = float(self.lr_var.get())
            self.agent.gamma = float(self.gamma_var.get())
        except Exception as e:
            messagebox.showerror("Invalid parameters", str(e))
            return

        self.training = True
        self.btn_start.config(text="Training...", state=tk.DISABLED)
        # start training in separate thread
        self.training_thread = threading.Thread(target=self._train_worker, args=(episodes,), daemon=True)
        self.training_thread.start()
        self._log_metrics(f"Started training for {episodes} episodes...")

    def _train_worker(self, episodes):
        # Basic Q-learning training
        ep_rewards = []
        for ep in range(1, episodes + 1):
            obs = self.env.reset(traffic_level=self.traffic_var.get())
            s_idx = self.env.encode_state(obs)
            total_r = 0.0
            done = False
            steps = 0
            while not done:
                a = self.agent.act(s_idx, greedy=False)
                obs2, r, done, _ = self.env.step(a)
                s2_idx = self.env.encode_state(obs2)
                self.agent.learn(s_idx, a, r, s2_idx, done)
                s_idx = s2_idx
                total_r += r
                steps += 1
            # record episode
            ep_rewards.append(total_r)
            self.episode_rewards.append(total_r)
            self.current_episode += 1
            # moving average
            if len(self.episode_rewards) >= 50:
                avg = np.mean(self.episode_rewards[-50:])
            else:
                avg = np.mean(self.episode_rewards)
            self.avg_rewards.append(avg)

            # periodically update plot/metrics - schedule on main thread via after
            if ep % 5 == 0 or ep == episodes:
                self.master.after(1, self._update_plot)
                self.master.after(1, lambda: self._log_metrics(
                    f"Episode {self.current_episode}: reward={total_r:.2f}  avg={avg:.2f}  eps={self.agent.epsilon:.3f}"))
            # allow UI responsiveness
            time.sleep(0.001)

        self.training = False
        self.master.after(1, self._on_training_finished)

    def _on_training_finished(self):
        self.btn_start.config(text="Start Training", state=tk.NORMAL)
        self._log_metrics("Training finished.")

    def _on_test_agent(self):
        if self.training:
            messagebox.showinfo("Busy", "Training in progress. Please wait.")
            return
        # Run one test episode using greedy policy and animate
        obs = self.env.reset(traffic_level=self.traffic_var.get())
        self._update_canvas()
        s_idx = self.env.encode_state(obs)
        done = False
        steps = 0
        total_r = 0.0

        def step_anim():
            nonlocal s_idx, done, steps, total_r
            if done or steps >= self.env.max_steps:
                self._log_metrics(f"Test finished: reward={total_r:.2f}, steps={steps}, distance={self.env.distance}")
                return
            a = self.agent.act(s_idx, greedy=True)
            obs2, r, done, _ = self.env.step(a)
            s_idx = self.env.encode_state(obs2)
            total_r += r
            steps += 1
            self._update_canvas()
            # schedule next step
            self.master.after(200, step_anim)

        self._log_metrics("Testing trained agent (greedy policy)...")
        step_anim()

    def _on_save_q(self):
        path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NPZ files", "*.npz")])
        if not path:
            return
        try:
            self.agent.save(path)
            self._log_metrics(f"Saved Q-table to {path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def _on_load_q(self):
        path = filedialog.askopenfilename(filetypes=[("NPZ files", "*.npz")])
        if not path:
            return
        try:
            self.agent.load(path.replace('.npz', ''))
            self._log_metrics(f"Loaded Q-table from {path}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _update_canvas(self):
        # Draw grid, locations, taxi, passenger, destination
        self.canvas.delete("all")
        W = self.grid_w
        H = self.grid_h
        cs = self.CELL_SIZE
        m = self.MARGIN

        # draw grid lines
        for i in range(W + 1):
            x = m + i * cs
            self.canvas.create_line(x, m, x, m + H * cs, fill="#cccccc")
        for j in range(H + 1):
            y = m + j * cs
            self.canvas.create_line(m, y, m + W * cs, y, fill="#cccccc")

        # draw special locations
        for i, (lx, ly) in enumerate(self.env.locs):
            x = m + lx * cs + cs / 2
            y = m + ly * cs + cs / 2
            self.canvas.create_oval(x - 18, y - 18, x + 18, y + 18, fill="#ffe4b5", outline="#c07a00")
            self.canvas.create_text(x, y, text=chr(ord('A') + i), font=("Arial", 14, "bold"))

        # passenger marker if not in taxi
        if self.env.pass_idx < self.env.num_locs:
            px, py = self.env.locs[self.env.pass_idx]
            x = m + px * cs + cs / 2
            y = m + py * cs + cs / 2
            self.canvas.create_text(x - 16, y + 18, text="P", font=("Arial", 12), fill="blue")

        # destination marker
        dx, dy = self.env.locs[self.env.dest_idx]
        x = m + dx * cs + cs / 2
        y = m + dy * cs + cs / 2
        self.canvas.create_text(x + 16, y + 18, text="D", font=("Arial", 12), fill="green")

        # taxi marker
        tx, ty = self.env.taxi_pos
        x = m + tx * cs + cs / 2
        y = m + ty * cs + cs / 2
        self.canvas.create_rectangle(x - 22, y - 22, x + 22, y + 22, fill="#ff6666", outline="#660000")
        self.canvas.create_text(x, y, text="Taxi", font=("Arial", 10, "bold"), fill="white")

        # traffic display
        traffic_str = ["Low", "Medium", "High"][self.env.traffic_level]
        self.canvas.create_text(m + 6, m + H * cs + 14, anchor=tk.W, text=f"Traffic: {traffic_str}", font=("Arial", 10))

    def _update_plot(self):
        if not self.episode_rewards:
            return
        xs = np.arange(1, len(self.episode_rewards) + 1)
        self.line_raw.set_data(xs, self.episode_rewards)
        self.line_avg.set_data(xs, self.avg_rewards)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas_fig.draw()

    def _log_metrics(self, msg):
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.insert(tk.END, f"{msg}\n")
        self.metrics_text.see(tk.END)
        self.metrics_text.config(state=tk.DISABLED)

    def _periodic_update(self):
        # called periodically by Tkinter mainloop
        self._update_canvas()
        self._update_plot()
        # update small status info
        if self.episode_rewards:
            last = self.episode_rewards[-1]
            avg = self.avg_rewards[-1]
            status = f"Episodes: {len(self.episode_rewards)}  LastR: {last:.2f}  Avg(50): {avg:.2f}  Eps: {self.agent.epsilon:.3f}"
        else:
            status = "No training yet."
        # put status into title
        self.master.title(f"Taxi Route Optimization - {status}")
        self.master.after(500, self._periodic_update)


def main():
    root = tk.Tk()
    app = TaxiGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
