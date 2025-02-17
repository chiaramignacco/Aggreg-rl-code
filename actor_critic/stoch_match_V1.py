import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import product
import random
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)


class StochasticDynamicMatching:
    def __init__(self, num_nodes, graph, arrival_rates, departure_rates, rewards, priority, experts, costs_dep, costs_queue, max_queue_length=5): 
        self.num_nodes = num_nodes
        self.graph = graph
        self.arrival_rates = arrival_rates
        self.departure_rates = departure_rates
        self.rewards = rewards
        self.max_queue_length = max_queue_length
        self.priority = priority
        self.experts = experts
        self.K = len(experts)
        self.costs_dep = costs_dep
        self.costs_queue = costs_queue

    def calculate_reward(self, match, state, cost_dep):
        cost = np.sum(np.array(state) * self.costs_dep) 
        if match:
            u = match[0]
            v = match[1]
            return self.rewards[u, v] - cost - cost_dep
        else:
            return -cost - cost_dep
        

    def match_greedy(self, state):
        edges = [(u, v, self.rewards[u, v]) for u, v in self.graph.edges() if state[u] > 0 and state[v] > 0]
        state_copy = state[:] 
        if edges:
            u, v, _ = max(edges, key=lambda x: x[2])
            state_copy[u] -= 1
            state_copy[v] -= 1
            return (u, v), tuple(state_copy)
        else:
            return None, tuple(state_copy)
    
    def match_longest(self, state):
        edges = [(u, v) for u, v in self.graph.edges() if state[u] > 0 and state[v] > 0]
        state_copy = state[:] 
        if edges:
            u, v = max(edges, key=lambda x: min(state_copy[x[0]], state_copy[x[1]]))
            state_copy[u] -= 1
            state_copy[v] -= 1
            return (u, v), tuple(state_copy)
        else:
            return None, tuple(state_copy)
            
    def match_priority(self, state):
        edges = [(u, v, self.priority[u, v]) for u, v in self.graph.edges() if state[u] > 0 and state[v] > 0]
        state_copy = state[:] 
        if edges:
            u, v = max(edges[::1], key=lambda x: x[2])[:2]
            state_copy[u] -= 1
            state_copy[v] -= 1
            # print((u, v), tuple(state))
            return (u, v), tuple(state_copy)
        else:
            return None, tuple(state_copy)

    def match_combination(self, state, expert):
        state = list(state)
        #print(expert)
        if expert == 'greedy':
            match, new_state = self.match_greedy(state)
        elif expert == 'match_longest':
            match, new_state = self.match_longest(state)
        elif expert == 'match_priority':
            # print(state)
            match, new_state = self.match_priority(state)
        else:
            print('ERROR: Unknown expert policy')
        return match, new_state

    def step(self, state):
        new_state = list(state)
        # Arrivals
        v = np.random.choice(np.array(range(self.num_nodes)), p=self.arrival_rates) 
        #print('arrival: ', v)
        if new_state[v] < self.max_queue_length:
            new_state[v] = min(new_state[v] + 1, self.max_queue_length)
            
        # Departures
        # num_dep = 0
        cost_dep = 0
        for i in range(self.num_nodes):
            if np.random.rand() < self.departure_rates[i]:
                if new_state[i] > 0:
                    new_state[i] = max(new_state[i] - 1, 0)
                    cost_dep += self.costs_dep[i]
                    #print('departure: ', i)
        
        return tuple(new_state), cost_dep

    def run_simulation(self, policy_net, discount_factor, initial_state=None, num_steps=500):
        # self.states = state_space
        # self.state_index = {state: idx for idx, state in enumerate(self.states)}
        
        if initial_state is None:
            initial_state = tuple([0] * self.num_nodes)
        
        state = initial_state
        state_history = [state]
        reward_history = []
        reward_total = 0.0
        
        policy_count = {expert: 0 for expert in self.experts}

        for step in range(num_steps):
            state_curr, cost_dep = self.step(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                policy_probs = policy_net(state_tensor)  # Softmax probabilities
            action = torch.multinomial(policy_probs, num_samples=1).item()
            expert = self.experts[action]
            policy_count[expert] += 1
            
            match, next_state = self.match_combination(state_curr, expert)
            reward = self.calculate_reward(match, next_state, cost_dep)
            reward_total += discount_factor**step * reward
            state_history.append(next_state)
            reward_history.append(reward_total)
            state = next_state

        return state_history, reward_history, reward_total, policy_count
    
    def run_simulation_expert(self, expert, discount_factor, initial_state=None, num_steps=500):
        
        if initial_state is None:
            initial_state = tuple([0] * self.num_nodes)
        
        state = initial_state
        state_history = [state]
        reward_history = []
        reward_total = 0.0

        for step in range(num_steps):
            state_curr, cost_dep = self.step(state)
            match, next_state = self.match_combination(state_curr, expert)
            reward = self.calculate_reward(match, next_state, cost_dep)
            reward_total += discount_factor**step * reward
            state_history.append(next_state)
            reward_history.append(reward_total)
            state = next_state

        return state_history, reward_history, reward_total


# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # Outputs logits

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        policy = torch.softmax(logits, dim=-1)
        return policy

# Define the Critic Network
class AdvantageNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(AdvantageNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # Outputs advantage values

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        advantages = self.fc3(x)
        return advantages

# def initialize_weights(module):
#     if isinstance(module, nn.Linear):
#         nn.init.xavier_uniform_(module.weight)
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)
            
            
def aggregation_update_exp(self, discount, learning_rate, model, H, k, num_repeat_est): 
    self.TD(discount_factor=discount)
    self.weights = self.weights * np.exp(learning_rate * (self.A) / (1 - discount))
    self.weights = self.weights / (np.sum(self.weights, axis=1)).reshape(self.weights.shape[0], 1) 
    return self.weights
    

# Training loop
def train_advantage(sdm, episodes=1000, learning_rate=0.00005, lr_decay_rate=0.9, discount_factor=0.9):
    state_dim = sdm.num_nodes
    action_dim = len(sdm.experts)
    
    # Initialize policy and critic networks
    policy_net = PolicyNetwork(state_dim, action_dim)
    critic_net = AdvantageNetwork(state_dim, action_dim)
    
    policy_optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate)
    critic_optimizer = optim.AdamW(critic_net.parameters(), lr=learning_rate)
    
    scheduler_policy = torch.optim.lr_scheduler.ExponentialLR(policy_optimizer, gamma=lr_decay_rate)
    scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(critic_optimizer, gamma=lr_decay_rate)
    
    eps = 0.3  # Initial exploration probability
    decay_rate = 0.96  # Epsilon decay rate for exploration
    
    policy_collection = []
    res = episodes / 20

    for episode in tqdm(range(episodes), desc="Training Progress"):
        initial_state = tuple([0] * sdm.num_nodes)
        state, cost_dep = sdm.step(initial_state)
        
        if episode % res == 0:
            policy_collection.append(copy.deepcopy(policy_net))

        for _ in range(1000):  # Interactions within one episode
            state_tensor = torch.FloatTensor(state).unsqueeze(0).requires_grad_()  # Ensure it tracks gradients
            
            # Get the current policy and sample action
            with torch.no_grad():
                policy_probs = policy_net(state_tensor)
            # policy_probs = torch.softmax(logits, dim=-1)
            
            if torch.isnan(policy_probs).any() or torch.isinf(policy_probs).any():
                print("Found NaN or Inf in policy_probs")
                
            action = torch.multinomial(policy_probs, num_samples=1).item()

            # Execute action in the environment
            match, next_state_temp = sdm.match_combination(state, sdm.experts[action])
            reward = sdm.calculate_reward(match, next_state_temp, cost_dep)
            next_state, cost_dep = sdm.step(next_state_temp)
            
            # Evaluate advantage function using the critic
            next_state_tensor = torch.FloatTensor(np.array(next_state)).unsqueeze(0)
            advantage_values = critic_net(state_tensor)
            
            # Compute TD target for critic
            target = reward + 0.9 * torch.max(critic_net(next_state_tensor)).item()  # Bellman backup
            
            # Update the critic network
            critic_loss = (advantage_values[0, action] - target) ** 2  # Mean squared error between advantage and target
            critic_optimizer.zero_grad()
            critic_loss.backward()  # Retain graph to allow another backward pass
            critic_optimizer.step()

            # Update the policy using the advantage values
            advantage = advantage_values[0].detach() # Don't detach here; use as is for gradient tracking
            # print(advantage)
            
            new_policy_logits = torch.log(policy_probs) + 1 / torch.sqrt(torch.tensor(float(episode + 1))) * advantage / (1 - discount_factor)
            new_policy = torch.softmax(new_policy_logits, dim=-1)
           
            kl_div = torch.sum(policy_probs * (torch.log(policy_probs) - new_policy_logits), dim=-1)
            policy_loss = kl_div.mean()
            
            # print(policy_loss)

            # Zero gradients for the policy network
            policy_optimizer.zero_grad()
            
            policy_loss.requires_grad = True

            policy_loss.backward()  
            policy_optimizer.step()

            # Move to the next state
            state = next_state

        # Decay exploration probability
        eps = max(0.01, eps * decay_rate)

        # Adjust learning rates periodically
        # if episode % 100 == 0:
        scheduler_policy.step()
        scheduler_critic.step()

    return policy_net, critic_net, policy_collection





