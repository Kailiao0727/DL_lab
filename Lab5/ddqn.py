# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DuelingDQN(nn.Module):
    def __init__(self, num_actions, input_channels=4):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(3136, 512)
        
        # Two streams
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, num_actions)

    def forward(self, x):
        x = x / 255.0
        x = self.feature(x)
        x = torch.relu(self.fc(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class DQN_task2(nn.Module):
    def __init__(self, num_actions, input_channels=4):
        super(DQN_task2, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, num_actions),)
    
    def forward(self, x):
        return self.network(x / 255.0)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
    
    def __len__(self):
        return len(self.buffer)

    def _get_n_step_transition(self):
        """Return combined transition from n-step buffer."""
        R = 0
        for idx in range(self.n_step):
            r, d = self.n_step_buffer[idx][2], self.n_step_buffer[idx][4]
            R += (self.gamma ** idx) * r
            if d:
                break  # stop if done early

        s, a = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        _, _, _, s_next, d = self.n_step_buffer[-1]
        return (s, a, R, s_next, d)

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return

        transition = self._get_n_step_transition()
        priority = (abs(error) + 1e-5) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        
        if transition[4]:
            self.n_step_buffer.clear()
        return  
        ########## END OF YOUR CODE (for Task 3) ########## 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Compute IS weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize 

        # Unzip transitions
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )
        ########## END OF YOUR CODE (for Task 3) ########## 
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
        return            
        ########## END OF YOUR CODE (for Task 3) ########## 
    
    def beta_update(self):
        """
            Update the beta value for importance sampling
        """
        if self.beta < 1.0:
            self.beta += 0.00001
        

class DQNAgent:
    def __init__(self, args=None):
        self.env = gym.make(args.env_name, render_mode="rgb_array")
        self.test_env = gym.make(args.env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)


        self.use_preprocessor = True
        if args.dualing:
            self.q_net = DuelingDQN(self.num_actions).to(self.device)
        else:
            self.q_net = DQN_task2(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        if args.dualing:
            self.target_net = DuelingDQN(self.num_actions).to(self.device)
        else:
            self.target_net = DQN_task2(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.best_reward = -21
        
        # self.q_net = DQN(self.num_actions, input_channels).to(self.device)
        # self.q_net.apply(init_weights)
        # self.target_net = DQN(self.num_actions).to(self.device)
        # self.target_net.load_state_dict(self.q_net.state_dict())
        # self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.epsilon_start = args.epsilon_start
        self.epsilon_min = args.epsilon_min
        self.n_step = args.n_step
        self.beta_start = 0.4
        self.beta_anneal_steps = args.beta_anneal_steps

        self.env_count = 0
        self.train_count = 0
        
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        self.restart_ep = 0
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize replay buffer
        self.memory = PrioritizedReplayBuffer(args.memory_size, alpha=0.6, beta_start=0.4, n_step=args.n_step, gamma=self.gamma)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        self.resume()
        for ep in range(self.restart_ep, episodes):
            obs, _ = self.env.reset()
            if self.use_preprocessor:
                state = self.preprocessor.reset(obs)
            else:
                state = obs
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                if self.use_preprocessor:
                    next_state = self.preprocessor.step(next_obs)
                else:
                    next_state = next_obs
                initial_error = 1.0  
                self.memory.add((state, action, reward, next_state, done), error=initial_error)

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            # print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_{self.env_count}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward >= self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, f"best_model_{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        if self.use_preprocessor:
            state = self.preprocessor.reset(obs)
        else:
            state = obs
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            if self.use_preprocessor:
                state = self.preprocessor.step(next_obs)
            else:
                state = next_obs

        return total_reward


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        # if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
            # self.epsilon -= 0.00000099
        frac = min(1.0, self.env_count / self.epsilon_decay_steps)
        self.epsilon = max(self.epsilon_min, self.epsilon_start - frac * (self.epsilon_start - self.epsilon_min))
        beta_frac = min(1.0, self.env_count / self.beta_anneal_steps)
        self.memory.beta = self.beta_start + beta_frac * (1.0 - self.beta_start)
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        
        with torch.no_grad():
            # next_q_values = self.target_net(next_states).max(1)[0]
            # target_q_values = rewards + (self.gamma ** args.n_step) * next_q_values * (1 - dones)
            # Task3
            next_actions = self.q_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (self.gamma ** self.n_step) * next_q_values * (1 - dones)


        
        # losses = (q_values - target_q_values).pow(2)
        losses = torch.nn.functional.smooth_l1_loss(q_values, target_q_values, reduction="none")
        loss = (losses * weights).mean()

        errors = (q_values - target_q_values).detach().abs().cpu().numpy()
        self.memory.update_priorities(indices, errors)
        
        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)

        self.optimizer.step()
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
           print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
    
    def resume(self):
        state_dict = torch.load("task3_results_dual/best_model_1220703.pt", map_location=self.device)
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.00001)
        self.env_count = 1220703
        self.train_count = 1226000
        self.memory.beta = 1.0
        self.epsilon = 0.01
        self.restart_ep = 541

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="task3-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay-steps", type=int, default=500000)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=4)
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--beta-anneal-steps", type=int, default=1_000_000)
    parser.add_argument("--dualing", action="store_true", help="Use Dueling DQN architecture")
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-Atari", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run(args.episodes)
    

#--discount-factor 0.9 --epsilon-decay 0.99 --replay-start-size 10000 --epsilon-min 0.2 --max-episode-steps 200 --lr 0.0025 --episodes 1000
# --env-name ALE/Pong-v5 --train-per-step 4 --episodes 1500 --memory-size 500000 --replay-start-size 80000 --target-update-frequency 5000

# --env-name ALE/Pong-v5 --episodes 1500 --memory-size 500000 --replay-start-size 80000 --target-update-frequency 8000 --lr 0.0000625 --save-dir ./task3_results