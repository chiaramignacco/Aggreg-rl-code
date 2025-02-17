import random
import itertools
import numpy as np
from itertools import permutations
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

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
        
        # Initialize the neural network and optimizer
        self.q_net = QNetwork(self.state_dim, self.action_dim, self.cLI)
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=0.009) # 0.001 Changed HERE
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96)
        self.loss_fn = nn.MSELoss() # nn.SmoothL1Loss()
        
        # q_tensor = torch.tensor(self.Q, dtype=torch.float32)
        
#         with torch.no_grad():
#             # Assuming q_layer is the final layer of the network with the correct output size
#             assert self.q_net.q_layer.weight.shape == q_tensor.shape, (
#                 f"Shape mismatch: q_layer weight shape is {self.q_net.q_layer.weight.shape}, "
#                 f"but Q shape is {q_tensor.shape}"
#             )

#         # Directly assign Q values to the weights
#             self.q_net.q_layer.weight.copy_(q_tensor)
            
            # Optional: Add noise or scaling factor to weights for more fine-tuned behavior
            # scaling_factor = 0.1
            # self.q_net.q_layer.weight.data += scaling_factor * torch.randn_like(self.q_net.q_layer.weight.data)
        
        
    def aggregation_step(self, weights, model):     
        state = tuple(model.queue.values())
        expert = np.random.choice(self.experts, p=weights[state])  
        #print(expert, self.weights[state], state)
        matching, reward = model.step(expert)                    
        #new_state = tuple(model.queue.values())      
        return matching, reward
                
    def project_onto_probability_simplex(self, p, tol=1e-5):
        """Projects a vector onto the probability simplex."""
        p_sorted = np.sort(p)[::-1]
        n = len(p)
        cum_sum = np.cumsum(p_sorted)
        
        # Find the largest t such that p_sorted - t >= 0
        t = (cum_sum - 1) / np.arange(1, n + 1)
        t = np.maximum(t, 0)
        
        # Calculate the projected vector
        projected_vector = np.maximum(p - t[-1], 0)
        
        # Ensure the sum is exactly 1
        projected_vector /= sum(projected_vector)
        return projected_vector
    
    # Updates with TD
    def aggregation_update_pol_pot(self, discount, p, model, H, k, num_repeat_est):                
        self.TD(5000, 0.9, discount)
        self.A_sum +=  self.A
        weights = np.maximum(self.A_sum, np.zeros(self.A_sum.shape))**(p-1)
        if (weights == 0).all():
            print('W 0')
        weights_sum = (np.sum(weights, axis=1)).reshape(weights.shape[0], 1)
        indx = np.where(weights_sum != 0)[0]
        
        self.weights[indx] = weights[indx] / weights_sum[indx]
        return self.weights
    
    # Update with TD
    def aggregation_update_exp(self, discount, learning_rate, model, H, k, num_repeat_est): 
        self.TD(5000, 0.9, discount)
        self.weights = self.weights * np.exp(learning_rate * (self.A) / (1 - discount))
        self.weights = self.weights / (np.sum(self.weights, axis=1)).reshape(self.weights.shape[0], 1) 
        return self.weights
    
    # Update with NN
    def aggregation_update_exp_NN(self, discount, learning_rate, model, H, k, num_repeat_est): 
        self.train_advantage_network(self.weights)
        self.weights = self.weights * np.exp(learning_rate * (self.A) / (1 - discount))
        self.weights = self.weights / (np.sum(self.weights, axis=1)).reshape(self.weights.shape[0], 1)
        return self.weights
    
    def aggregation_update_exp_NN_direct(self, discount, learning_rate, model, H, k, num_repeat_est): 
        self.train_advantage_network_direct(self.weights)
        self.weights = self.weights * np.exp(learning_rate * (self.A) / (1 - discount))
        self.weights = self.weights / (np.sum(self.weights, axis=1)).reshape(self.weights.shape[0], 1)
        return self.weights
    
    def TD(self, num_steps, learning_rate, discount):
        model = StochasticMatching(self.graph, self.arrival_rates, self.queue_max)                 
        state = 0
        start_learning_rate_decay = 1
        end_learning_rate_decay = num_steps 
        learning_rate_decay_value = learning_rate / (end_learning_rate_decay - start_learning_rate_decay)
                        
        for step in range(num_steps):
            if np.random.rand() < 0.1:
                e = np.random.randint(self.K)
            else:
                e = np.random.choice(np.arange(self.K), p=self.weights[state]) 
            expert = self.experts[e]
            _, reward = model.step(expert)  
            reward_norm = reward - self.eta * sum(model.queue.values()) + self.cLI
            
            _ = model.arrivals()
            s = tuple(model.queue.values())
            next_state = int(model.val_list.index(s))
            e_next = np.random.choice(np.arange(self.K), p=self.weights[next_state])
             
            # Update the Q-value using the update rule
            self.Q[state, e] = (1 - learning_rate) * self.Q[state, e] + learning_rate * (reward_norm + discount * self.Q[next_state, e_next])
            # print(self.state_space[state], self.state_space[next_state], reward)
            state = next_state
            if end_learning_rate_decay >= step >= start_learning_rate_decay:
                learning_rate -= learning_rate_decay_value   
                     
        V = (np.sum(self.Q * self.weights, axis=1)).reshape(self.Q.shape[0], 1)       
        A = self.Q - V
        
        # if (A == self.A).all():
            # print(True)
            
        # self.A = A
        # return A

    
    def train_advantage_network(self, weights, episodes=500, entropy_param=0.1): 
        
        eps = 0.3
        decay_rate = 0.96
        
        
        for episode in range(episodes):
        
            model = StochasticMatching(self.graph, self.arrival_rates, self.queue_max)                 
            state = 0
            
            for _ in range(200):
                curr_state = self.state_space[state]
                state_tensor = torch.FloatTensor(curr_state).unsqueeze(0)  # Add batch dimension
                
                # with torch.no_grad():
                q_values = self.q_net(state_tensor)
                
                if np.random.random() < eps:
                    action = np.random.randint(0, self.action_dim)
                else:
                    action = np.random.choice(np.arange(self.K), p=self.weights[state]) 
                
                expert = self.experts[action]
                _, reward = model.step(expert)  
                reward_norm = reward - self.eta * sum(model.queue.values()) + self.cLI
                
                _ = model.arrivals()
                s = tuple(model.queue.values())
                next_state = int(model.val_list.index(s))
                action_next = np.random.choice(np.arange(self.K), p=self.weights[next_state])
    
                
                # Compute target for Q(s, a)
                next_state_tensor = torch.FloatTensor(np.array(s)).unsqueeze(0)
                
                next_q_value = self.q_net(next_state_tensor)[0][action_next]
                
                target = reward + 0.9 * (next_q_value - entropy_param * np.log(self.weights[next_state, action_next]))
                # print(next_q_value, target)
                
                loss = self.loss_fn(q_values[0, action], target)
                self.optimizer.zero_grad()

                loss.backward()
                
                self.optimizer.step()
                
                state = next_state
                
            eps = max(0.01, eps * decay_rate)
            self.scheduler.step()
            
            
        state_space_tensor = torch.FloatTensor(self.state_space).unsqueeze(0)
            
        self.q_net.eval()
        
        with torch.no_grad():
            output = self.q_net(state_space_tensor)  # Shape: (num_states, m)
            
        Q = np.array(output).reshape(len(self.state_space), self.K)
        
        V = (np.sum(Q * self.weights, axis=1)).reshape(Q.shape[0], 1)       
        A_ = Q - V
        
        # print(Q, A_)
    
        self.A = A_
        
        
    def train_advantage_network_direct(self, weights, episodes=500, entropy_param=0.1): 
        
        eps = 0.3
        decay_rate = 0.96
        
        
        for episode in range(episodes):
        
            model = StochasticMatching(self.graph, self.arrival_rates, self.queue_max)                 
            state = 0
            
            for _ in range(200):
                curr_state = self.state_space[state]
                state_tensor = torch.FloatTensor(curr_state).unsqueeze(0)  # Add batch dimension
                with torch.no_grad():
                    q_values = self.q_net(state_tensor)
                
                # Choose action (e.g., based on policy or epsilon-greedy strategy)
                if np.random.random() < eps:
                    action = np.random.randint(0, self.action_dim)
                else:
                    action = np.random.choice(np.arange(self.K), p=self.weights[state]) 
                
                expert = self.experts[action]
                
                _, reward = model.edge_match(expert)  
                reward_norm = reward - self.eta * sum(model.queue.values()) + self.cLI
                
                _ = model.arrivals()
                s = tuple(model.queue.values())
                next_state = int(model.val_list.index(s))
                action_next = np.random.choice(np.arange(self.K), p=self.weights[next_state])
    
                
                # Compute target for Q(s, a)
                next_state_tensor = torch.FloatTensor(np.array(s)).unsqueeze(0)
                
                next_q_value = self.q_net(next_state_tensor)[0][action_next]
                
                target = reward + 0.9 * (next_q_value - entropy_param * np.log(self.weights[next_state, action_next]))
                
                
                # requires_grad=True
                loss = self.loss_fn(q_values[0, action], target)
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                
                state = next_state
                
            eps = max(0.01, eps * decay_rate)
            
            self.scheduler.step()
            
            
        state_space_tensor = torch.FloatTensor(self.state_space).unsqueeze(0)
            
        self.q_net.eval()
        
        with torch.no_grad():
            output = self.q_net(state_space_tensor)  # Shape: (num_states, m)
            
        Q = np.array(output).reshape(len(self.state_space), self.K)
        
        V = (np.sum(Q * self.weights, axis=1)).reshape(Q.shape[0], 1)       
        A_ = Q - V
        
    
        self.A = A_

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, cLI, hidden_dim=16):
        super(QNetwork, self).__init__()
        self.cLI = cLI
        # Neural network architecture
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layers for Q(s, a) 
        self.q_layer = nn.Linear(hidden_dim, action_dim)  # Q-values for each action
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        q_values = self.q_layer(x)   # Q(s, a) for each action
        
        q_values_clipped = torch.clamp(q_values, min=-self.cLI, max=self.cLI + 200)
        
        return q_values_clipped
    
    
    
    
    
    
    
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    