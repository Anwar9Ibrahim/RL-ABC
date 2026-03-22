import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from rl_framework.Utils import setLogger, set_seed
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

class GaussianNoise:
    def __init__(self, action_dim, mu=0.0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        self.action_dim = action_dim

    def sample(self):
        return np.random.normal(self.mu, self.sigma, self.action_dim)

    def reset(self):
        pass  # for compatibility with OU noise

class Actor(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=800, fc2_dims=600, fc3_dims=512, fc4_dims=256):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        f3 = 1. / np.sqrt(self.fc3.weight.data.size()[0])
        nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        self.bn3 = nn.LayerNorm(fc3_dims)

        self.fc4 = nn.Linear(fc3_dims, fc4_dims)
        f4 = 1. / np.sqrt(self.fc4.weight.data.size()[0])
        nn.init.uniform_(self.fc4.weight.data, -f4, f4)
        nn.init.uniform_(self.fc4.bias.data, -f4, f4)
        self.bn4 = nn.LayerNorm(fc4_dims)

        self.mu = nn.Linear(fc4_dims, n_actions)
        f_mu = 0.003
        nn.init.uniform_(self.mu.weight.data, -f_mu, f_mu)
        nn.init.uniform_(self.mu.bias.data, -f_mu, f_mu)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        actions = torch.tanh(self.mu(x))  # Output bounded in [-1, 1]
        return actions

class Critic(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=800, fc2_dims=600, fc3_dims=512, fc4_dims=256):
        super(Critic, self).__init__()

        # STATE pathway
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        f3 = 1. / np.sqrt(self.fc3.weight.data.size()[0])
        nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        self.bn3 = nn.LayerNorm(fc3_dims)

        self.fc4 = nn.Linear(fc3_dims, fc4_dims)
        f4 = 1. / np.sqrt(self.fc4.weight.data.size()[0])
        nn.init.uniform_(self.fc4.weight.data, -f4, f4)
        nn.init.uniform_(self.fc4.bias.data, -f4, f4)
        self.bn4 = nn.LayerNorm(fc4_dims)

        # ACTION pathway (injected late)
        self.action_value = nn.Linear(n_actions, fc4_dims)

        # Final output: Q-value
        self.q = nn.Linear(fc4_dims, 1)
        f_q = 0.003
        nn.init.uniform_(self.q.weight.data, -f_q, f_q)
        nn.init.uniform_(self.q.bias.data, -f_q, f_q)

    def forward(self, state, action):
        s = F.relu(self.bn1(self.fc1(state)))
        s = F.relu(self.bn2(self.fc2(s)))
        s = F.relu(self.bn3(self.fc3(s)))
        s = (self.bn4(self.fc4(s)))

        a = F.relu(self.action_value(action))

        # Combine state and action features
        x = F.relu(s + a)
        q_value = self.q(x)

        return q_value

class ReplayBuffer:
    def __init__(self, capacity=int(1e6)):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            Pickled_buffer = pickle.load(f)
        
        #print("############# debiging in ddpg memory ############# ")
        #print("Type Pickled_buffer: ", type(Pickled_buffer))
        #self.buffer= Pickled_buffer.buffer depends if we picled te whole Replay buffer or only the deque
        self.buffer= Pickled_buffer
        
        #print("Type Pickled_buffer.buffer:  ", Pickled_buffer.buffer)
        #print("Type buffer: ", type(self.buffer))
        #mprint("############# debiging in ddpg memory ############# ")
            

import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class DDPGAgent:
    def __init__(self, env, convert= True, alpha=1e-4, beta=1e-3, batch_size=128, gamma=0.99, tau=0.005, max_size=1000000, noise_type='gaussian',
                 log_interval=100, eval_interval=1000, num_steps=1000000, seed=0, exp="", load=False):
        set_seed(seed)
        self.seed = int(seed)
        self.device= device
        
        self.env = env
        self.convert= convert
        self.input_dims = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]
        self.actor = Actor(self.input_dims, self.n_actions).to(device)
        self.actor_target = Actor(self.input_dims, self.n_actions).to(device)
        self.critic = Critic(self.input_dims, self.n_actions).to(device)
        self.critic_target = Critic(self.input_dims, self.n_actions).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=beta)

        self.buffer = ReplayBuffer(capacity=max_size)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        if noise_type == 'ou':
            self.noise = OUNoise(self.n_actions)
        elif noise_type == 'gaussian':
            self.noise = GaussianNoise(self.n_actions)
        else:
            raise ValueError("Invalid noise type. Choose 'ou' or 'gaussian'.")

        # Logging and evaluation
        self.steps = 0
        self.episodes = 0
        self.num_steps = num_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.train_rewards = []
        self.logs = []
        self.exp = exp
        self.load = load

        # TensorBoard
        self.summary_dir = os.path.join(os.getcwd(), 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=f"{self.summary_dir}/{self.exp}")

        # CSV logger
        self.logger, self.file_handler = setLogger(False, 'ddpg_logger.csv',
                                                  ('episode', 'epsilon'))

        # Debug logger
        debug_logger_headers = ('step', 'action', 'noise', 'source')
        self.logger_debug, self.file_handler_debug = setLogger(False, 'ddpg_debug_logger.csv', debug_logger_headers)

        # For evaluation
        self.plots_dir = f"plots_{self.seed}"
        self.logs_dir = f"logs_{self.seed}"
        self.models_dir = f"models_{self.seed}"
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.best_eval_particles = -1
        self.best_eval_log = None
        self.recent_eval_logs = deque(maxlen=10)

        if self.load:
            self.load_models()

    '''def choose_action(self, state, greedy=0.5, log_source="train"):
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        noise = greedy * self.noise.sample()
        action_noisy = action + noise
        action_noisy = np.clip(action_noisy, -1, 1)
        # Rescale to env action space if needed
        low = self.env.action_space.low
        high = self.env.action_space.high
        action_rescaled = low + (0.5 * (action_noisy + 1.0) * (high - low))
        action_rescaled = np.clip(action_rescaled, low, high)
        # Debug log
        return action_noisy'''
   

    def choose_action(self, state, greedy=0.5, log_source="train"):
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        noise = greedy * self.noise.sample()
        action_noisy = action + noise
        action_noisy = np.clip(action_noisy, -1, 1)
        return action_noisy

    def get_greedy(self, episode, n_episodes, greedy_start=0.5, greedy_end=0.05):
        """Linearly decay greedy (noise scale) from greedy_start to greedy_end over n_episodes."""
        greedy = greedy_start - (greedy_start - greedy_end) * (episode / max(1, n_episodes - 1))
        return max(greedy, greedy_end)

    def train(self, n_episodes=1000, max_steps=1000, greedy=0.5, start=0):
        scores = []
        for ep in range(start, n_episodes,1):
            # Decay greediness as training progresses
            greedy_ep = self.get_greedy(ep, n_episodes, greedy_start=greedy, greedy_end=0.05)
            state, _ = self.env.reset() if isinstance(self.env.reset(), tuple) else (self.env.reset(), {})
            self.reset_noise()
            score = 0
            episode_actions = []
            done = False
            while not done:
                if type(state) == tuple:
                    state = state[0]
                action = self.choose_action(state, greedy=greedy_ep, log_source="train")
                next_state, reward, done, _, info = self.env.step(action, self.convert)
                self.remember(state, action, reward, next_state, done)
                self.learn()
                state = next_state
                score += reward
                self.steps += 1
                episode_actions.append(action)
                if self.steps % self.eval_interval == 0:
                    self.evaluate()
                if done:
                    self.writer.add_scalar('Rewards/Reward', reward, self.steps)
                    self.writer.add_scalar('Rewards/NumberOfParticles', info['number_of_particles'], self.steps)
                    break
            scores.append(score)
            self.train_rewards.append(score)
            self.logger.writerow({'episode': ep+1, 'epsilon': greedy_ep})
            #self.logger.writerow({'episode': ep+1, 'reward': reward, 'actions': episode_actions, 'output_file': info.get('output_file', '')})
            self.file_handler.flush()
        return scores
    

    def _log_debug(self, step, action, noise, source):
        ep_info = {
            'step': step,
            'action': action,
            'noise': noise,
            'source': source
        }
        self.logger_debug.writerow(ep_info)
        self.file_handler_debug.flush()

    def reset_noise(self):
        self.noise.reset()

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            y = reward + (1 - done) * self.gamma * target_q

        q = self.critic(state, action)
        critic_loss = F.mse_loss(q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        # TensorBoard logging
        self.writer.add_scalar('loss/critic', critic_loss.item(), self.steps)
        self.writer.add_scalar('loss/actor', actor_loss.item(), self.steps)

        if self.steps % self.log_interval == 0:
            print(f"Step {self.steps}: Critic loss {critic_loss.item():.4f}, Actor loss {actor_loss.item():.4f}")

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def remember(self, s, a, r, s2, d):
        self.buffer.push(s, a, r, s2, d)


    def evaluate(self, episodes=1):
        eval_log_file = os.path.join(self.logs_dir, f'ddpg_eval_{self.steps}.csv')
        eval_plot_file = os.path.join(self.plots_dir, f'ddpg_eval_{self.steps}.png')
        eval_logger_headers = ('episode', 'rewards', 'number_of_particles', 'output_file', 'episode_reward', 'actions')
        eval_logger, eval_file_handler = setLogger(False, eval_log_file, eval_logger_headers)

        self.actor.eval()
        print("####### DDPG EVALUATING #########")
        best_particles_this_eval = -1
        for i in range(episodes):
            avg_reward = 0.0
            episode_rewards = []
            episode_reward = 0.0
            done = False
            state = self.env.reset()
            episode_actions = []
            t = 0
            while not done:
                if type(state)== tuple:
                    state= state[0]
                action = self.choose_action(state, greedy=0.0, log_source="eval")
                next_state, reward, done, _, info = self.env.step(action, self.convert) 
                episode_reward += reward
                state = next_state
                episode_rewards.append(reward)
                episode_actions.append(action)
                t += 1
                eval_info = {
                    'episode': i + 1,
                    'rewards': reward,
                    'output_file': info.get("output_file", ""),
                    'number_of_particles': info.get("number_of_particles", 0),
                    'episode_reward': episode_reward,
                    'actions': info["dict_vars"] 
                }
                eval_logger.writerow(eval_info)
                eval_file_handler.flush()

            final_particles = info.get("number_of_particles", 0)
            best_particles_this_eval = max(best_particles_this_eval, final_particles)

            print("Episode:", i + 1)
            print('rewards:', reward)
            avg_reward += episode_reward

            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, t+1), episode_rewards, 'r-o', label='Sum Episode Reward')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title(f'DDPG Evaluation at Training Step {self.steps}')
            plt.legend()
            plt.grid(True)
            plt.savefig(eval_plot_file)
            plt.close()

        eval_file_handler.close()

        self.save_models(tag="last")

        if best_particles_this_eval > self.best_eval_particles:
            old_best_log = self.best_eval_log
            self.best_eval_particles = best_particles_this_eval
            self.best_eval_log = (eval_log_file, eval_plot_file)
            self.save_models(tag="best")
            if old_best_log:
                self._remove_eval_file_if_not_recent(old_best_log[0])
                self._remove_eval_file_if_not_recent(old_best_log[1])
            print(f'New best model! number_of_particles: {best_particles_this_eval} at step {self.steps}')

        is_best = self.best_eval_log and self.best_eval_log[0] == eval_log_file
        if not is_best:
            if len(self.recent_eval_logs) == self.recent_eval_logs.maxlen:
                evicted_log, evicted_plot = self.recent_eval_logs[0]
                self._safe_remove(evicted_log)
                self._safe_remove(evicted_plot)
            self.recent_eval_logs.append((eval_log_file, eval_plot_file))

        print(f'DDPG Evaluation completed at step {self.steps}')
        print(f'Average reward: {avg_reward:.2f} over {episodes} episodes')
        print(f'Best number_of_particles so far: {self.best_eval_particles}')
        print("@@@@@@@ DONE DDPG EVALUATING @@@@@@@")
        self.actor.train()

    def _remove_eval_file_if_not_recent(self, filepath):
        """Remove a file only if it's not in the recent logs deque."""
        for log, plot in self.recent_eval_logs:
            if filepath in (log, plot):
                return
        self._safe_remove(filepath)

    def _safe_remove(self, filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass

    def save_models(self, tag="last"):
        os.makedirs(self.models_dir, exist_ok=True)
        if tag == "best":
            torch.save(self.actor.state_dict(), os.path.join(self.models_dir, "ddpg_actor_best.pth"))
        else:
            torch.save({
                'steps': self.steps,
                'best_eval_particles': self.best_eval_particles,
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'actor_target': self.actor_target.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
            }, os.path.join(self.models_dir, "ddpg_checkpoint_last.pth"))
        print(f"DDPG Models saved successfully ({tag}) at step {self.steps}.")

    def load_models(self, tag="last"):
        if tag == "best":
            self.actor.load_state_dict(torch.load(os.path.join(self.models_dir, "ddpg_actor_best.pth"), map_location=self.device))
            print("Loaded best actor for evaluation.")
        else:
            checkpoint = torch.load(os.path.join(self.models_dir, "ddpg_checkpoint_last.pth"), map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.steps = checkpoint['steps']
            self.best_eval_particles = checkpoint['best_eval_particles']
            print(f"Resumed training from step {self.steps} (best particles: {self.best_eval_particles}).")