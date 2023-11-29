import re
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from routers.calvin.bnn_trader import model
import pickle
from llm_debug import llm_debugger
import dotenv
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Specify the device for computation




class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 2048000),
            nn.ReLU(),
            nn.Linear(2048000, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 2048000),
            nn.ReLU(),
            nn.Linear(2048000, 1)
        )
    
    def forward(self, state):
        return self.network(state)


class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate=0.0001, model_dir="path_to_save_model", optimization_epochs=10, mini_batch_size=256):
        super(PPOAgent, self).__init__()
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.clip_param = 0.2
        self.model_dir = model_dir
        self.gamma = 0.99
        self.gae_lambda = 0.95  # Lambda for GAE
        self.optimization_epochs = optimization_epochs
        self.mini_batch_size = mini_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_grad_norm = 1.0
        self.mini_batch_size = 6
        self.to(self.device)  # Move the whole model to the chosen device
        with open(r"C:\dev\\CalvinBrains\\Py\\bnnscalar.p", 'rb') as f:
            self.scaler = pickle.load(f)
        with open(r"C:\dev\\CalvinBrains\\Py\\bnnsom.p", 'rb') as f:
            self.som = pickle.load(f)
        with open(r"C:\\calvin\\es\\eslowstate\\somvalue.p", 'rb') as f:
            self.value_som = pickle.load(f)
        with open(r"C:\\calvin\\es\\eslowstate\\scalarvalue.p", 'rb') as f:
            self.value_scalar = pickle.load(f)


    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value
    @llm_debugger(reflections=1, output = "act.md")
    def act(self, state, trade_open):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        #state = torch.nn.functional.one_hot(torch.tensor([state]), num_classes=100).float().to(device)

        action_probs, value = self.forward(state)

        # Ensure action_probs is 2D: [1, num_actions]
        action_probs = action_probs.squeeze(0)

        # Limit the values of action_probs to avoid extreme values
        action_probs = torch.clamp(action_probs, -100, 100)

        # Replace NaN and Inf values with large finite numbers
        action_probs = torch.nan_to_num(action_probs, nan=1e-10, posinf=1.0, neginf=-1.0)

        # Apply softmax to convert to probability distribution
        action_probs = torch.softmax(action_probs, dim=-1)

        # Check for any NaN, Inf, or unexpected negative values
        if not ((action_probs >= 0).all() and torch.isfinite(action_probs).all()):
            raise ValueError(f"Invalid action probabilities: {action_probs}")
        if trade_open:
            # Assuming the last two actions in your action space are 'hold' and 'close trade'
            # Zero out probabilities of all actions except 'hold' and 'close trade'
            action_probs[:-2] = 0
            action_probs = action_probs / action_probs.sum()  # Renormalize probabilities
        else:
            # If no trade is open, zero out the probability of the 'close trade' action
            # Assuming 'close trade' is the last action in your action space
            action_probs[-1] = 0  # Set the probability of 'close trade' to 0
            action_probs = action_probs / action_probs.sum()  # Renormalize probabilities
        # Sample an action from the probability distribution
        action = action_probs.multinomial(num_samples=1).detach()
        #action = torch.argmax(action_probs, dim=-1).unsqueeze(-1)
        print(value)
        print(action_probs)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(action)
        # Convert tensor to Python int for the action ID
        return action.item(), state, log_probs, value, action_probs
        
    @llm_debugger(reflections=1, output = "gae.md")
    def compute_gae(self, next_observations, rewards, dones, values_old):
        # Ensure values_old is a tensor
        if not isinstance(values_old, torch.Tensor):
            values_old = torch.tensor(values_old, dtype=torch.float32)

        # Convert rewards and dones to tensors if they are not already
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)
        if not isinstance(dones, torch.Tensor):
            dones = torch.tensor(dones, dtype=torch.float32)

        # Append 0 to values_old for the terminal state
        # Use torch.cat for concatenation
        zero_tensor = torch.tensor([0], dtype=torch.float32)
        values = torch.cat((values_old, zero_tensor))

        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])

        return returns
    @llm_debugger(reflections=1, output = "update.md")
    def update(self, observations, actions, log_probs_old, rewards, next_observations, dones, values_old):

        advantages = torch.tensor(self.compute_gae(next_observations, rewards, dones, values_old)).to(device)
        
        observations = torch.tensor(observations).to(device)
        actions = torch.tensor(actions).to(device)
        log_probs_old = torch.tensor(log_probs_old).to(device)
        values_old = torch.tensor(values_old).to(device)
        
        for _ in range(self.optimization_epochs):
            # Get current policies and value
            action_probs, state_values = self.forward(observations)
            dist = Categorical(action_probs)
            
            # New log probabilities for the actions taken
            log_probs_new = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Probability ratios
            ratios = torch.exp(log_probs_new - log_probs_old.detach())
            
            # Compute clipped objective function
            surr1 = ratios * advantages.detach()
            surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages.detach()
            actor_loss = - torch.min(surr1, surr2).mean()
            
            # Critic loss as mean squared error
            #critic_loss = F.mse_loss(state_values.squeeze(-1), values_old + advantages.detach())
            critic_loss = F.mse_loss(state_values.squeeze(), values_old + advantages.detach())

            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # Gradient descent step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Save the models after the update
            self.save_models()

        # Save the models after the update
    def save_models(self):
    #    torch.save(self.actor.state_dict(), self.model_dir)
    #    torch.save(self.critic.state_dict(), self.model_dir)
        # Check if the directory exists, if not, create it
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

        try:
            torch.save(self.actor.state_dict(), os.path.join(self.model_dir, 'actor.pth'))
            torch.save(self.critic.state_dict(), os.path.join(self.model_dir, 'critic.pth'))
            # Similarly save other parts of the model if necessary
        except Exception as e:
            print("An exception occurred while saving the model:", e)
    def load_models(self):
        self.actor.load_state_dict(torch.load(os.path.join(self.model_dir, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(self.model_dir, 'critic.pth')))

# Compute Generalized Advantage Estimation (GAE)
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        gae = gae.detach()  # detach from graph, not to backprop later
        returns.insert(0, gae + values[step])
    return returns
# Usage example
# state_dim = ...
# action_dim = ...
# agent = PPOAgent(state_dim, action_dim)
# observations, actions, rewards, next_observations, dones = ...
# next_value = agent.critic(torch.FloatTensor(next_observations).unsqueeze(0)).detach()
# returns = compute_gae(next_value, rewards, dones, values
