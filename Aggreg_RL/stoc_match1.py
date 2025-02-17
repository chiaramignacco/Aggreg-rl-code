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
    
# This call is used for aggregation 
class Aggregation: 
    def __init__(self, graph, arrival_rates, experts, eta, queue_max, Q=None):
        self.graph = graph
        self.arrival_rates = arrival_rates
        self.algorithm = experts
        self.eta = eta
        self.queue_max = queue_max        
        
        self.experts = experts
        self.state_space = list(itertools.product(np.array(range(self.queue_max + 1)), repeat=len(self.graph.keys())))
        
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
        
        
        self.agent = DoubleDQNAgent(state_dim=self.state_dim, action_dim=self.action_dim)
        
        
    def aggregation_step(self, weights, model):     
        state = tuple(model.queue.values())
        expert = np.random.choice(self.experts, p=weights[state])  
        #print(expert, self.weights[state], state)
        matching, reward = model.step(expert)                    
        #new_state = tuple(model.queue.values())      
        return matching, reward

    # Update with NN
    def aggregation_update_exp_NN(self, discount, learning_rate, model, H, k, num_repeat_est): 
        self.train_double_dqn(self.agent, num_episodes=50)
        self.weights = self.weights * np.exp(learning_rate * (self.A) / (1 - discount))
        self.weights = self.weights / (np.sum(self.weights, axis=1)).reshape(self.weights.shape[0], 1)
        return self.weights

    
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
    
        state_space_tensor = torch.FloatTensor(self.state_space).unsqueeze(0)
                
        agent.q_network.eval()
        
        with torch.no_grad():
            output = agent.q_network(state_space_tensor)  # Shape: (num_states, m)
            
        Q = np.array(output).reshape(len(self.state_space), self.K)
        
        V = (np.sum(Q * self.weights, axis=1)).reshape(Q.shape[0], 1)       
        A = Q - V
        
        self.A = A
   

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
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=64, buffer_size=10000):
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

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        
    
    
    
    
    
    
    
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    