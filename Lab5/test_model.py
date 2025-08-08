import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse
from dqn import DQN_task1, DQN_task2, AtariPreprocessor



# class AtariPreprocessor:
#     def __init__(self, frame_stack=4):
#         self.frame_stack = frame_stack
#         self.frames = deque(maxlen=frame_stack)

#     def preprocess(self, obs):
#         if len(obs.shape) == 3 and obs.shape[2] == 3:
#             gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
#         else:
#             gray = obs
#         resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
#         return resized

#     def reset(self, obs):
#         frame = self.preprocess(obs)
#         self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
#         return np.stack(self.frames, axis=0)

#     def step(self, obs):
#         frame = self.preprocess(obs)
#         self.frames.append(frame.copy())
#         stacked = np.stack(self.frames, axis=0)
#         return stacked
        
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.env_name == "CartPole-v1":
        input_channels = 4
        use_preprocessor = False
    else:
        use_preprocessor = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env_name, render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    if args.env_name == "CartPole-v1":
        model = DQN_task1(num_actions).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        model = DQN_task2(num_actions).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        if use_preprocessor:
            state = preprocessor.reset(obs)
        else:
            state = obs
        done = False
        total_reward = 0
        frames = []
        frame_idx = 0

        while not done:
            frame = env.render()
            if use_preprocessor:
                frame = cv2.resize(frame, (608, 400))
            else:
                frame = cv2.resize(frame, (160, 224))
            frames.append(frame)

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if use_preprocessor:
                state = preprocessor.step(next_obs)
            else:
                state = next_obs
            frame_idx += 1

        out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30) as video:
            for f in frames:
                video.append_data(f)
        print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")
        rewards.append(total_reward)
    
    average_reward = sum(rewards) / args.episodes
    print(f"Average reward over {args.episodes} episodes: {average_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=313551076, help="Random seed for evaluation")
    parser.add_argument("--env-name", type=str, default="CartPole-v1")
    args = parser.parse_args()
    evaluate(args)

# --env-name=ALE/Pong-v5 --model-path=results/best_model.pt