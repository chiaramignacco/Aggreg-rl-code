import random
import itertools
import numpy as np
from itertools import permutations
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm
from collections import deque
from scipy.special import logsumexp

# In thiss call we implement the model and the experts
class StochasticMatching:
    def __init__(self, graph, arrival_rates, queue_max, queue=None):
        self.graph = graph
        self.arrival_rates = arrival_rates
        if queue == None:
            self.queue = {vertex: 0 for vertex in graph}
        else:
            self.queue = queue 
        self.total_reward = 0
        self.queue_max = queue_max        
            
        prob_arrival = []
        N =np.sum(np.array(list(arrival_rates.values())))
        
        for rate in arrival_rates.values():
            prob_arrival.append(rate/N) 
            
        self.prob_arrival = np.array(prob_arrival)
        # Convert graph to a list of edges with weights
        self.edges = [(u, v, w) for u in graph for v, w in graph[u]]
        self.state_space = list(itertools.product(np.array(range(self.queue_max + 1)), repeat=len(self.graph.keys())))
        
        n = 0
        self.state_space_ind = {}
        for s in self.state_space:
            self.state_space_ind[str(n)] = s
            n += 1 
        
        self.key_list = list(self.state_space_ind.keys())
        self.val_list = list(self.state_space_ind.values())
        
        self.edges_list = []   
        self.rewards_ind = {}
        self.verteces_ind = {'A': 0,
               'B': 1,
               'C': 2,
               'D': 3
               }
        
        self.verteces = {'A': 0, 'B': 1, 'C':2, 'D':3}
        
        for edge in self.edges:
            e=[]
            e.append(self.verteces_ind[np.array(edge)[0]])
            e.append(self.verteces_ind[np.array(edge)[1]])
            if [self.verteces_ind[np.array(edge)[1]], self.verteces_ind[np.array(edge)[0]]] not in self.edges_list:
                self.edges_list.append(e)
                self.rewards_ind[str(e)] = edge[2]
                
        self.matching_rates = {edge: [0] for edge in self.edges}
        
    def arrivals(self, print_v=False):
        stop = False
        vertex = np.random.choice(list(self.arrival_rates.keys()), p=self.prob_arrival) 
        #print(vertex)
        if print_v:
            print(vertex)
        if self.queue[vertex] < self.queue_max:           
            self.queue[vertex] += 1
            if self.queue[vertex] == self.queue_max:
                stop = True
        return stop
    
    # Experts
    def match_the_longest(self):
        matching = []
        reward = 0
        
        selected_edges = []
        edge_max = 0
        for edge in self.edges: 
            if (self.queue[edge[0]] > 0)  and (self.queue[edge[1]] > 0):
                if self.queue[edge[0]] + self.queue[edge[1]] > edge_max: 
                    edge_max = self.queue[edge[0]] + self.queue[edge[1]]
                    selected_edges = []
                    selected_edges.append(edge)
                #elif self.queue[edge[0]] + self.queue[edge[1]] == edge_max:
                #    selected_edges.append(edge)
        if selected_edges:
            i = np.random.choice(np.array(range(len(selected_edges)))) 
            matched_edge = selected_edges[i]
            u, v, w = matched_edge
            matching.append(matched_edge)
            self.matching_rates[matched_edge].append(self.matching_rates[matched_edge][-1] + 1)           
            self.queue[u] -= 1
            self.queue[v] -= 1
            self.total_reward += w
            reward +=w
            
        for e in self.edges:
            if e not in matching:
                self.matching_rates[e].append(self.matching_rates[e][-1])

        return matching, reward
    
    def edge_priority_match_random(self):
        matched = set()
        matching = []
        reward = 0      
        # Sort edges based on the arrival rates
        edges = self.edges.copy()
        random.seed(11)
        random.shuffle(edges)
        
        for u, v, w, r in edges:
            if u not in matched and v not in matched and self.queue[u] > 0 and self.queue[v] > 0:
                matching.append((u, v, w))
                self.matching_rates[(u, v, w)].append(self.matching_rates[(u, v, w)][-1] + 1)
                matched.add(u)
                matched.add(v)
                self.queue[u] -= 1
                self.queue[v] -= 1
                self.total_reward += w    
                reward +=w 
                
        for e in self.edges:
            if e not in matching:
                self.matching_rates[e].append(self.matching_rates[e][-1])                
        return matching, reward
    
    def random_match(self):
        matching = []
        reward = 0
        
        selected_edges = []
        edge_max = 0
        edges = self.edges.copy()
        random.shuffle(edges)
        
        for edge in edges: 
            if (self.queue[edge[0]] > 0)  and (self.queue[edge[1]] > 0):
                selected_edges.append(edge)
                
        if selected_edges:
            i = np.random.choice(np.array(range(len(selected_edges)))) 
            matched_edge = selected_edges[i]
            u, v, w = matched_edge
            matching.append(matched_edge)
            self.matching_rates[matched_edge].append(self.matching_rates[matched_edge][-1] + 1)           
            self.queue[u] -= 1
            self.queue[v] -= 1
            self.total_reward += w
            reward += w
            
        for e in self.edges:
            if e not in matching:
                self.matching_rates[e].append(self.matching_rates[e][-1])

        return matching, reward

    def edge_priority_match_reward(self):
        matched = set()
        matching = []
        reward = 0
        edges = self.edges.copy()
        # Sort edges based on rewards 
        edges.sort(key=lambda x: x[2], reverse=True)
        for u, v, w in edges:
            if u not in matched and v not in matched and self.queue[u] > 0 and self.queue[v] > 0:
                matching.append((u, v, w))
                self.matching_rates[(u, v, w)].append(self.matching_rates[(u, v, w)][-1] + 1)
                matched.add(u)
                matched.add(v)
                self.queue[u] -= 1
                self.queue[v] -= 1
                self.total_reward += w    
                reward +=w 
                
        for e in self.edges:
            if e not in matching:
                self.matching_rates[e].append(self.matching_rates[e][-1])               
        return matching, reward
    
    def edge_priority_match_arrival_rate_high(self):
        matched = set()
        matching = []
        reward = 0      
        # Sort edges based on the arrival rates
        edges = [(u, v, w, ru + rv) for u in self.graph for v, w in self.graph[u] for rv in [self.arrival_rates[v]] for ru in [self.arrival_rates[u]]]
        edges.sort(key=lambda x: x[3], reverse=True)
        
        for u, v, w, r in edges:
            if u not in matched and v not in matched and self.queue[u] > 0 and self.queue[v] > 0:
                matching.append((u, v, w))
                self.matching_rates[(u, v, w)].append(self.matching_rates[(u, v, w)][-1] + 1)
                matched.add(u)
                matched.add(v)
                self.queue[u] -= 1
                self.queue[v] -= 1
                self.total_reward += w    
                reward +=w 
                
        for e in self.edges:
            if e not in matching:
                self.matching_rates[e].append(self.matching_rates[e][-1])                
        return matching, reward
    
    def edge_priority_match_arrival_rate_low(self):
        matched = set()
        matching = []
        reward = 0      
        # Sort edges based on the arrival rates
        edges = [(u, v, w, ru + rv) for u in self.graph for v, w in self.graph[u] for rv in [self.arrival_rates[v]] for ru in [self.arrival_rates[u]]]
        edges.sort(key=lambda x: x[3])
        
        for u, v, w, r in edges:
            if u not in matched and v not in matched and self.queue[u] > 0 and self.queue[v] > 0:
                matching.append((u, v, w))
                self.matching_rates[(u, v, w)].append(self.matching_rates[(u, v, w)][-1] + 1)
                matched.add(u)
                matched.add(v)
                self.queue[u] -= 1
                self.queue[v] -= 1
                self.total_reward += w    
                reward +=w 
                
        for e in self.edges:
            if e not in matching:
                self.matching_rates[e].append(self.matching_rates[e][-1])                
        return matching, reward
    
    def edge_priority_match_arrival_rate_low(self):
        matched = set()
        matching = []
        reward = 0      
        # Sort edges based on the arrival rates
        edges = [(u, v, w, ru + rv) for u in self.graph for v, w in self.graph[u] for rv in [self.arrival_rates[v]] for ru in [self.arrival_rates[u]]]
        edges.sort(key=lambda x: x[3])
        
        for u, v, w, r in edges:
            if u not in matched and v not in matched and self.queue[u] > 0 and self.queue[v] > 0:
                matching.append((u, v, w))
                self.matching_rates[(u, v, w)].append(self.matching_rates[(u, v, w)][-1] + 1)
                matched.add(u)
                matched.add(v)
                self.queue[u] -= 1
                self.queue[v] -= 1
                self.total_reward += w    
                reward +=w 
                
        for e in self.edges:
            if e not in matching:
                self.matching_rates[e].append(self.matching_rates[e][-1])                
        return matching, reward
    
    def probability_match_05(self):
        matched = set()
        matching = []
        reward = 0      
        # Sort edges based on the arrival rates
        edges = [(u, v, w) for u in self.graph for v, w in self.graph[u]]  
        if np.random.uniform() < 0.5:
            for u, v, w in edges:
                if u not in matched and v not in matched and self.queue[u] > 0 and self.queue[v] > 0:
                    matching.append((u, v, w))
                    self.matching_rates[(u, v, w)].append(self.matching_rates[(u, v, w)][-1] + 1)
                    matched.add(u)
                    matched.add(v)
                    self.queue[u] -= 1
                    self.queue[v] -= 1
                    self.total_reward += w    
                    reward +=w            
        for e in self.edges:
            if e not in matching:
                self.matching_rates[e].append(self.matching_rates[e][-1])                
        return matching, reward
    
    def edge_match(self, edge):
        reward = 0
        matching = []
        # if edge != 'no_move':
        if (self.queue[edge[0]] > 0) and (self.queue[edge[1]] > 0):
            self.queue[edge[0]] -= 1
            self.queue[edge[1]] -= 1    
            e =[]
            e.append(self.verteces_ind[np.array(edge)[0]])
            e.append(self.verteces_ind[np.array(edge)[1]])
            reward += self.rewards_ind[str(e)]
            matching.append(edge)
        return matching, reward
        
    
    def step(self, algorithm):          
        if algorithm == 'match_the_longest':
            matching, reward = self.match_the_longest()
        elif algorithm == 'edge_priority_match_reward':
            matching, reward = self.edge_priority_match_reward()             
        elif algorithm == 'edge_priority_match_arrival_rate_high':
            matching, reward = self.edge_priority_match_arrival_rate_high()             
        elif algorithm == 'edge_priority_match_arrival_rate_low':
            matching, reward = self.edge_priority_match_arrival_rate_low()
        elif algorithm == 'probability_match_05':
            matching, reward = self.probability_match_05()
        elif algorithm == 'random_match':
            matching, reward = self.random_match()
        elif algorithm == 'edge_priority_match_random':
            matching, reward = self.edge_priority_match_random()
        elif algorithm == 'edge_priority_match_random':
            matching, reward = self.edge_priority_match_random()
        else:
            print('Error: unknown expert')
        return matching, reward   
    
    def run_simulation(self, policy, experts, eta, discount, cLI, num_MC=10000, num_steps=200):
        
        rewards_list = []
        K = len(experts)
    
        for _ in range(num_MC):
            
            total_reward = 0
            
            self.queue = {vertex: 0 for vertex in self.graph}
            
            
            for step in range(num_steps):
                
                _ = self.arrivals()
                state = tuple(self.queue.values())
                state_tensor = torch.FloatTensor(state).unsqueeze(0)  
                
                with torch.no_grad():
                    action_probs = policy(state_tensor) 
                    action_probs = torch.softmax(action_probs, dim=-1)
                        
                action = np.random.choice(K, p=action_probs.numpy()[0])
                # action = agent.select_action(state, epsilon, self.weights)
                
                expert = experts[action]
                _, reward = self.step(expert)  
                reward_norm = reward - eta * sum(self.queue.values()) + cLI
                
                
                # next_state = tuple(self.queue.values())
                
                # state = next_state
                total_reward += discount**step * reward_norm
                
            rewards_list.append(total_reward)
                
        return np.mean(np.array(rewards_list))
    
        
        
    
class Aggregation: 
    def __init__(self, graph, arrival_rates, experts, eta, queue_max, lambda_l2, lambda_l1, lr=1e-2, Q=None):
        self.graph = graph
        self.arrival_rates = arrival_rates
        self.algorithm = experts
        self.eta = eta
        self.queue_max = queue_max        
        
        self.experts = experts
        self.state_space = list(itertools.product(np.array(range(self.queue_max + 1)), repeat=len(self.graph.keys())))
        self.state_space_tensor = torch.FloatTensor(self.state_space).unsqueeze(0)
        
        self.K = len(experts)            
        self.weights = np.ones([len(self.state_space), self.K]) / self.K
        if Q is None:
            self.Q = np.zeros([len(self.state_space), self.K])
        else:
            self.Q = Q
            
        self.A = np.zeros([len(self.state_space), self.K])
        self.A_sum = np.zeros([len(self.state_space), self.K]) 
        self.A_sum_squared = np.zeros([len(self.state_space), self.K])
        self.cLI = (self.eta * self.queue_max * self.K)
        
        
        self.state_dim = len(self.graph)
        self.action_dim = self.K
        self.p = 3
        
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim)
        
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=lambda_l2)# 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.99)
        
        # milestones1 = [6] * 8 + [2] * 50
        # milestones2 = [6] * 8 #+ [4] * 4 + [2] * 50
#         milestones3 = [2] * 8 + [1] * 50
        
        # milestones = np.cumsum(milestones1)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.99)
        
        
        self.criterion = nn.MSELoss()
        
        self.agent = DoubleDQNAgent(state_dim=self.state_dim, action_dim=self.action_dim)
        
        
    def aggregation_step(self, weights, model):     
        state = tuple(model.queue.values())
        expert = np.random.choice(self.experts, p=weights[state])  
        #print(expert, self.weights[state], state)
        matching, reward = model.step(expert)                    
        #new_state = tuple(model.queue.values())      
        return matching, reward
    
    def aggregation_update_pol_potNN(self, discount, p):     
        self.train_double_dqn(self.agent, num_episodes=40)
        self.A_sum +=  self.A
        weights = np.maximum(self.A_sum, np.zeros(self.A_sum.shape))**(p-1)
        if (weights == 0).all():
            print('W 0')
        weights_sum = (np.sum(weights, axis=1)).reshape(weights.shape[0], 1)
        indx = np.where(weights_sum != 0)[0]
        self.weights[indx] = weights[indx] / weights_sum[indx]
        return self.weights
    
    def aggregation_update_exp_NN(self, discount, learning_rate): 
        # self.train_double_dqn(self.agent, num_episodes=40)
        self.TD(5000, 0.9, discount)
        self.weights = self.weights * np.exp(learning_rate * (self.A) / (1 - discount))
        self.weights = self.weights / (np.sum(self.weights, axis=1)).reshape(self.weights.shape[0], 1)
        return self.weights

    
    def train_actor_critic(self, num_episodes, lr, epsilon_start=0.4, epsilon_end=0.01, epsilon_decay=0.995, gamma=0.9):
    
        self.policy_net.eval()
        
        with torch.no_grad():
            weights = self.policy_net(self.state_space_tensor) 
            
        self.weights = np.array(weights).reshape(len(self.state_space), self.K)
        
        for episode in range(num_episodes):
            
            self.train_double_dqn(self.agent, 15)
            # self.TD(5000, 0.4, gamma) # 0.4
            
            curr_weights = self.policy_net(self.state_space_tensor).squeeze(0)
            
            scaled_A = lr * self.A / (1 - gamma) 
            
            # Exp update with log
            
            log_weights = np.log(self.weights + 1e-12)  # 1e-12 to avoid log(0)
            log_new_weights_unnormalized = log_weights + scaled_A

            # Step 3: Normalize in log-space via log-sum-exp
            # logsumexp computes log( sum( exp(x) ) ) in a numerically stable way
            log_norm = logsumexp(log_new_weights_unnormalized, axis=1, keepdims=True)
            log_new_weights = log_new_weights_unnormalized - log_norm
            
            new_weights = np.exp(log_new_weights)
            
            # PARTE CON EXP UPDATE
#             new_weights = self.weights * np.exp(scaled_A)
#             new_weights = new_weights / (np.sum(new_weights, axis=1)).reshape(self.weights.shape[0], 1)
            
            # POL POT UPDATE
            
            # self.A_sum +=  self.A
            # new_weights = np.maximum(self.A_sum, np.zeros(self.A_sum.shape))**(self.p-1)
            # if (weights == 0).all():
            #     print('W 0')
            # weights_sum = (np.sum(new_weights, axis=1)).reshape(new_weights.shape[0], 1)
            # indx = np.where(weights_sum != 0)[0]
            # new_weights[indx] = new_weights[indx] / weights_sum[indx]
            # print(new_weights)
            
            new_weights_tensor = torch.tensor(new_weights, dtype=torch.float32)
            
            #Compute the actor loss (KL divergence)
            
            l1_norm = sum(torch.abs(param).sum() for param in self.policy_net.parameters())
            
            actor_loss = self.kl_divergence(new_weights_tensor, curr_weights)
            # actor_loss = actor_loss + self.lambda_l1 * l1_norm
            # actor_loss = self.criterion(curr_weights, new_weights_tensor)
            
            # Update actor
            self.optimizer.zero_grad()
            actor_loss.backward()
            self.optimizer.step()
            
            self.scheduler.step()
            
            self.policy_net.eval()
        
            with torch.no_grad():
                weights = self.policy_net(self.state_space_tensor) 
                
            self.weights = np.array(weights).reshape(len(self.state_space), self.K)
            
        return self.weights
    
    def kl_divergence(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Computes KL(P || Q) for each sample (row) in the batch, then returns the mean.

        KL(P || Q) = sum_x P(x) * log [P(x) / Q(x)]

        Arguments:
            p  -- Tensor of shape (batch_size, action_dim), where each row is a distribution.
            q  -- Tensor of shape (batch_size, action_dim), where each row is a distribution.
            eps -- Small constant to avoid log(0) and division by zero.

        Returns:
            A scalar Tensor (the mean KL divergence over the batch).
        """
        # 1) Clamp so we never take log of zero or negative numbers
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)
        
        entropy = - (p * torch.log(p + eps)).sum(dim=-1).mean()
        
        # 2) (Optional) Re-normalize each row so it sums to 1
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)

        # 3) Compute KL(P || Q) per row, then average over the batch
        kl = p * (torch.log(p) - torch.log(q))
        
        return kl.sum(dim=-1).mean() - 0 * entropy
    
    
    def train_double_dqn(self, agent, num_episodes, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update_freq=10):
        epsilon = epsilon_start
        for episode in range(num_episodes):
            
            
            model = StochasticMatching(self.graph, self.arrival_rates, self.queue_max)                 
            state = 0
    
            total_reward = 0
    
            for _ in range(100):
                
                curr_state = self.state_space[state]
                # state_tensor = torch.FloatTensor(curr_state).unsqueeze(0)  
                
                action = agent.select_action(state, epsilon, self.weights)
                
                expert = self.experts[action]
                _, reward = model.step(expert)  
                reward_norm = reward - self.eta * sum(model.queue.values()) + self.cLI
                
                _ = model.arrivals()
                next_s = tuple(model.queue.values())
                next_state = int(model.val_list.index(next_s))
                
                agent.store_transition(curr_state, action, reward_norm, next_s)
    
                state = next_state
                total_reward += reward
    
                agent.update(self.weights, model)
    
            # Decay epsilon
            epsilon = max(epsilon * epsilon_decay, epsilon_end)
    
            # Update target network
            if episode % target_update_freq == 0:
                agent.update_target_network()
                
        agent.q_network.eval()
        
        with torch.no_grad():
            output = agent.q_network(self.state_space_tensor)  # Shape: (num_states, m)
            
        Q = np.array(output).reshape(len(self.state_space), self.K)
        
        V = (np.sum(Q * self.weights, axis=1)).reshape(Q.shape[0], 1)       
        A = Q - V
        
        self.A = A
        
    def TD(self, num_steps, learning_rate, discount):
        model = StochasticMatching(self.graph, self.arrival_rates, self.queue_max)  
        model.queue = {vertex: 0 for vertex in self.graph}
        state = 0
        start_learning_rate_decay = 1
        end_learning_rate_decay = num_steps 
        learning_rate_decay_value = learning_rate / (end_learning_rate_decay - start_learning_rate_decay)
        
        # states, actions, rewards, states_idx = [], [], [], []
                        
        for step in range(num_steps):
            curr_state = self.state_space[state]
            
            if np.random.rand() < 0.1:
                e = np.random.randint(self.K)
            else:
                e = np.random.choice(np.arange(self.K), p=self.weights[state]) 
#                 with torch.no_grad():
#                     e = self.weights(torch.tensor(curr_state, dtype=torch.float32).unsqueeze(0) ).multinomial(1).item()
                
            expert = self.experts[e]
            _, reward = model.step(expert)  
            reward_norm = reward - self.eta * sum(model.queue.values()) + self.cLI
            
            # states.append(curr_state)
            # actions.append(e)
            # rewards.append(reward_norm)
            # states_idx.append(state)
            
            _ = model.arrivals()
            s = tuple(model.queue.values())
            next_state = int(model.val_list.index(s))
            e_next = np.random.choice(np.arange(self.K), p=self.weights[next_state])
             
            # Update the Q-value using the update rule
            self.Q[state, e] = (1 - learning_rate) * self.Q[state, e] + learning_rate * (reward_norm + discount * self.Q[next_state, e_next])

            state = next_state
            if end_learning_rate_decay >= step >= start_learning_rate_decay:
                learning_rate -= learning_rate_decay_value   
                     
        V = (np.sum(self.Q * self.weights, axis=1)).reshape(self.Q.shape[0], 1)       
        A = self.Q - V
        
        # if (A == self.A).all():
            # print(True)
            
        self.A = A
        
        # return states, states_idx


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        logits = self.fc3(x)
        return self.softmax(logits)


   
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        return self.fc3(x)
    
    
class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.9, lr=1e-3, batch_size=64, buffer_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.loss_fn = nn.MSELoss()
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.99)

    def select_action(self, state, epsilon, weights):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        # state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # with torch.no_grad():
        #     q_values = self.q_network(state_tensor)
        return np.random.choice(np.arange(self.action_dim), p=weights[state]) #torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def update(self, weights, model):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of transitions
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        state_ind = [int(model.val_list.index(state)) for state in states]

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)

        # Compute current Q values
        current_q_values = self.q_network(states_tensor).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            # next_actions1 = torch.argmax(self.q_network(next_states_tensor), dim=1, keepdim=True)
            next_actions = [np.random.choice(np.arange(self.action_dim), p=weights[state]) for state in state_ind] 
            next_actions = torch.LongTensor(next_actions).unsqueeze(1).to(self.device)
            next_q_values = self.target_network(next_states_tensor).gather(1, next_actions)
            target_q_values = rewards + self.gamma * next_q_values

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # self.scheduler.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        
    
    
    
    
    
    
    
    
    
    
    
    

    
    