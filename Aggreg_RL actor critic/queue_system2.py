import numpy as np
import itertools
import random
import networkx as nx


seed = random.seed(10)
seed_np = np.random.seed(10)

# Setting

graph = nx.Graph()

num_nodes = 16

verteces = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
verteces_ind = {k: v for k, v in zip(verteces, verteces)}
 
# Define nodes with attributes
nodes = {
    0: {'group': '0', 'urgency': 'donor'},
    1: {'group': 'A', 'urgency': 'donor'},
    2: {'group': 'B', 'urgency': 'donor'},
    3: {'group': 'AB', 'urgency': 'donor'},
    4: {'group': '0', 'urgency': 'high'},
    5: {'group': '0', 'urgency': 'medium'},
    6: {'group': '0', 'urgency': 'low'},
    7: {'group': 'A', 'urgency': 'high'},
    8: {'group': 'A', 'urgency': 'medium'},
    9: {'group': 'A', 'urgency': 'low'},
    10: {'group': 'B', 'urgency': 'high'},
    11: {'group': 'B', 'urgency': 'medium'},
    12: {'group': 'B', 'urgency': 'low'},
    13: {'group': 'AB', 'urgency': 'high'},
    14: {'group': 'AB', 'urgency': 'medium'},
    15: {'group': 'AB', 'urgency': 'low'}
}

# Add nodes to the graph
for node, attr in nodes.items():
    graph.add_node(node, **attr)

# Define edges
edges_ = [
    (0, 4), (0, 5), (0, 6), 
    (1, 7),  (1, 9), 
    (2, 10), (2, 12), 
    (3, 13), (3, 15),
    (9, 0),  (7, 0), 
    (13, 0), (13, 1), (13, 2),
    (15, 0), (15, 1), (15, 2), 
    (12, 0), (10, 0), (11, 0), (2, 11), 
    (1, 8), (8, 0), (14, 0), (14, 1), 
    (14, 2), (3, 14)
]

graph.add_edges_from(edges_)

# Define rewards for urgency levels
reward_scheme = {
    'donor': 0,
    'low': 50,
    'medium': 200,
    'high': 500
}

# Initialize the reward matrix
rewards = np.zeros((num_nodes, num_nodes))
costs_dep, costs_queue = np.zeros(num_nodes),  np.zeros(num_nodes)

for n in range(num_nodes):
    if nodes[n]['urgency'] == 'high':
        costs_dep[n] = 1
    if nodes[n]['urgency'] == 'medium':
        costs_dep[n] = 40
    if nodes[n]['urgency'] == 'low':
        costs_dep[n] = 50
    if nodes[n]['urgency'] == 'donor':
        costs_dep[n] = 1
        
for n in range(num_nodes):
    if nodes[n]['urgency'] == 'high':
        costs_queue[n] = 1
    if nodes[n]['urgency'] == 'medium':
        costs_queue[n] = 2
    if nodes[n]['urgency'] == 'low':
        costs_queue[n] = 4
    if nodes[n]['urgency'] == 'donor':
        costs_queue[n] = 0
        
costs_dep = np.array(costs_dep)
costs_queue = np.array(costs_queue)

# Assign rewards based on node urgency levels
for edge in edges_:
    node1, node2 = edge
    urgency1 = nodes[node1]['urgency']
    urgency2 = nodes[node2]['urgency']
    r = max(reward_scheme[urgency1], reward_scheme[urgency2])
    rewards[node1, node2] = r
    rewards[node2, node1] = r


num_edges = len(graph.edges())
max_queue_length = 5
arrival_rates_values = np.random.rand(num_nodes)
arrival_rates_values = arrival_rates_values / np.sum(arrival_rates_values)
departure_rates = np.random.rand(num_nodes)
departure_rates = departure_rates / np.sum(departure_rates)

edges = [(u, v, rewards[u, v]) for u, v in edges_]

arrival_rates = {k: v for k, v in zip(verteces, arrival_rates_values)}
arrival_rates_ind = {ind: rate for ind, (vertex, rate) in enumerate(arrival_rates.items())}


graph_dict = {}
for node in graph.nodes():
    adjacency_list = []
    for neighbor in graph.neighbors(node):
        reward_value = rewards[node, neighbor]
        adjacency_list.append((neighbor, reward_value))
    graph_dict[node] = adjacency_list

print("Dictionary representation:")
for key, neighbors in graph_dict.items():
    print(f"{key}: {neighbors}")

graph = graph_dict
edges = [(u, v, rewards[u, v]) for u, v in edges_]
edges_list = []


rewards_ind = {}
for edge in edges:
    u, v, w = edge
    e = [verteces_ind[u], verteces_ind[v]]
    if e[::-1] not in edges_list:
        edges_list.append(e)
        rewards_ind[str(e)] = w
        
        
print(rewards_ind.keys())


#####################################
eta = 0.5
discount = 0.8
H = 50
k = 10
queue_max = 1


# experts = edges

experts = ['match_the_longest', 'edge_priority_match_reward', 'random_match']
K = len(experts)

queues = {v: [] for v in verteces}

state_space = list(itertools.product(range(queue_max + 1), repeat=len(graph)))

state_space_ind = {str(i): s for i, s in enumerate(state_space)}

exp_ind = {str(i): exp for i, exp in enumerate(experts)}

prob_arrival = [rate / sum(arrival_rates.values()) for rate in arrival_rates.values()]

num_states = len(state_space)
num_exp = len(experts)
num_actions = len(edges_list) + 1

state_action_space_ind = {
    str(i): [s, a]
    for i, (s, a) in enumerate(itertools.product(state_space, edges_list))
}

# edges_ind = {i: edge for i, edge in enumerate(edges)}
# edges_ind[len(edges)] = 'no_move'

key_list = list(state_space_ind.keys())
val_list = list(state_space_ind.values())

cLI = eta * queue_max * len(verteces) #10


# Expert Policies
def match_the_longest(edges_list, state):
    selected_edges = []
    edge_max = -np.inf
    for edge in edges_list:
        q1, q2 = edge
        if state[q1] > 0 and state[q2] > 0:
            edge_length = state[q1] + state[q2]
            if edge_length > edge_max:
                edge_max = edge_length
                selected_edges = [edge]
    return selected_edges


def edge_priority_match_reward(edges_list, state):
    selected_edges = []
    edge_max = -np.inf
    for edge in edges_list:
        q1, q2 = edge
        if state[q1] > 0 and state[q2] > 0:
            reward = rewards_ind[str(edge)]
            if reward > edge_max:
                edge_max = reward
                selected_edges = [edge]
    return selected_edges


def random_match(edges_list, state):
    return [edge for edge in edges_list if state[edge[0]] > 0 and state[edge[1]] > 0]


def edge_priority_match_arrival_rate_low(edges_list, state):
    selected_edges = []
    edge_min = np.inf
    for edge in edges_list:
        q1, q2 = edge
        if state[q1] > 0 and state[q2] > 0:
            arrival_sum = arrival_rates_ind[q1] + arrival_rates_ind[q2]
            if arrival_sum < edge_min:
                edge_min = arrival_sum
                selected_edges = [edge]
    return selected_edges


def edge_priority_match_arrival_rate_high(edges_list, state):
    selected_edges = []
    edge_max = -np.inf
    for edge in edges_list:
        q1, q2 = edge
        if state[q1] > 0 and state[q2] > 0:
            arrival_sum = arrival_rates_ind[q1] + arrival_rates_ind[q2]
            if arrival_sum > edge_max:
                edge_max = arrival_sum
                selected_edges = [edge]
    return selected_edges


def edge_priority_match_random(edges_list, state):
    random.seed(11)
    shuffled_edges = edges_list.copy()
    random.shuffle(shuffled_edges)
    for edge in shuffled_edges:
        if state[edge[0]] > 0 and state[edge[1]] > 0:
            return [edge]
    return []


def edge_match(edge, state):
    q1, q2 = edge
    if state[q1] > 0 and state[q2] > 0:
        return [edge]
    return []

experts_dict = {'match_the_longest': match_the_longest,
                'edge_priority_match_reward' : edge_priority_match_reward,
                'random_match': random_match,
                   } 

exp_ind = {'match_the_longest': 0,
         'edge_priority_match_reward' : 1,
         'random_match': 2,
            } 

cLI = (eta * queue_max * len(verteces))


def compute_transitions_and_rewards_avg(weights):
    P = np.zeros((num_states, num_states))
    r = np.zeros(num_states)
                       
    for index, state in state_space_ind.items():
        selected_edges_experts = {exp: [] for exp in experts} 
        for exp in experts: 
            selected_edges_experts[exp] = experts_dict[exp](edges_list, state)
            if selected_edges_experts[exp]: 
                num_edges = len(selected_edges_experts[exp])
                for edge in selected_edges_experts[exp]:
                    q1 = edge[0]
                    q2 = edge[1]
                    new_state = np.array(state).copy()
                    new_state[q1] -= 1
                    new_state[q2] -= 1
                    sum_queues = np.sum(new_state)
                    for q in range(len(new_state)):
                        if np.array(new_state)[q] < queue_max:
                            new_state_after_arr = new_state.copy()
                            new_state_after_arr[q] += 1
                            new_state_after_arr = tuple(new_state_after_arr)
                            index_new = int(val_list.index(new_state_after_arr))
                            P[int(index), int(index_new)] += weights[int(index)][exp_ind[exp]] * (1 / num_edges) * prob_arrival[q]
                            r[int(index)] += prob_arrival[q] * weights[int(index)][exp_ind[exp]] * (1 / num_edges) * (rewards_ind[str(edge)] - eta * sum_queues)
                        else:           
                            new_state_after_arr = new_state.copy()
                            index_new = int(val_list.index(tuple(new_state_after_arr)))
                            P[int(index), int(index_new)] += weights[int(index)][exp_ind[exp]] * (1 / num_edges) * prob_arrival[q]
                            r[int(index)] += weights[int(index)][exp_ind[exp]] * (rewards_ind[str(edge)] - eta * sum_queues) * prob_arrival[q]* (1 / num_edges) 
            else:
                for q in range(len(state)):
                    if np.array(state)[q] < queue_max:
                        new_state = np.array(state) 
                        new_state[q] += 1
                        sum_queues = np.sum(state)
                        new_state = tuple(new_state)
                        index_new = int(val_list.index(new_state))
                        P[int(index), index_new] += weights[int(index)][exp_ind[exp]]  * prob_arrival[q]
                        r[int(index)] += weights[int(index)][exp_ind[exp]] * (- eta * sum_queues) * prob_arrival[q]
                    else:           
                        sum_queues = np.sum(state)
                        P[int(index), int(index)] += weights[int(index)][exp_ind[exp]] * prob_arrival[q]
                        r[int(index)] += weights[int(index)][exp_ind[exp]] * (- eta * sum_queues) * prob_arrival[q] 
    #normalise rewards
    r_norm = r + cLI
    # check
    # for s in range(len(state_space)):
    #     if np.abs(np.sum(P[s,:]) - 1) > 1e-5:
    #         print('Error: transition probabilities do not some to 1', np.sum(P[s,:]), s)
    return P, r_norm
    

def compute_transitions_and_rewards_experts():
    P = np.zeros((num_states, num_exp, num_states))
    r = np.zeros((num_states, num_exp, num_states))
    
    for index, state in state_space_ind.items():
        for exp in experts: 
            selected_edges_experts = experts_dict[exp](edges_list, state)
            if selected_edges_experts: 
                num_edges = len(selected_edges_experts)
                for edge in selected_edges_experts:
                    q1 = edge[0]
                    q2 = edge[1]
                    new_state = np.array(state).copy()
                    new_state[q1] -= 1
                    new_state[q2] -= 1
                    sum_queues = np.sum(new_state)
                    for q in range(len(new_state)):
                        if np.array(new_state)[q] < queue_max:
                            new_state_after_arr = new_state.copy()
                            new_state_after_arr[q] += 1
                            new_state_after_arr = tuple(new_state_after_arr)
                            index_new = int(val_list.index(new_state_after_arr))
                            P[int(index), exp_ind[exp], index_new] += (1 / num_edges) * prob_arrival[q]
                            r[int(index), exp_ind[exp], index_new] += (1 / num_edges) * (rewards_ind[str(edge)] - eta * sum_queues) 
                        else:            
                            new_state_after_arr = new_state.copy()
                            index_new = int(val_list.index(tuple(new_state_after_arr)))
                            P[int(index), exp_ind[exp], index_new] += 1 / num_edges * prob_arrival[q]
                            r[int(index), exp_ind[exp], index_new] += (1 / num_edges) *(rewards_ind[str(edge)] - eta * sum_queues)      
            else:
                for q in range(len(state)):
                    if np.array(state)[q] < queue_max:
                        new_state = np.array(state).copy()
                        new_state[q] += 1
                        sum_queues = np.sum(state)
                        new_state = tuple(new_state)
                        index_new = int(val_list.index(new_state))
                        P[int(index),  exp_ind[exp], index_new] += prob_arrival[q]
                        r[int(index),  exp_ind[exp], index_new] += - eta * sum_queues 
                    else:           
                        sum_queues = np.sum(state)
                        P[int(index), exp_ind[exp], int(index)] += prob_arrival[q]
                        r[int(index), exp_ind[exp], int(index)] = (- eta * sum_queues)
                        
    #normalise rewards
    r_norm = r + cLI
    # check
    # for s in range(num_states):
    #     for e in range(num_exp):
    #         if np.abs(np.sum(P[s,e, :]) - 1) > 1e-5:
    #             print('Error: transition probabilities do not some to 1', np.sum(P[s,:]), s)
    return P, r_norm

def compute_transitions_and_rewards_actions():
    P = np.zeros((num_states, num_actions, num_states))
    r = np.zeros((num_states, num_actions, num_states))
    
    for index, state in state_space_ind.items():
        potential_matches = 0
        for ind, edge in edges_ind.items():
            if edge != 'no_move':
                if (np.array(state)[edge[0]] > 0)  and (np.array(state)[edge[1]] > 0):
                    potential_matches += 1
                    q1 = edge[0]
                    q2 = edge[1]
                    new_state = np.array(state).copy()
                    new_state[q1] -= 1
                    new_state[q2] -= 1
                    sum_queues = np.sum(new_state)
                    for q in range(len(new_state)):
                        if np.array(new_state)[q] < queue_max:
                            new_state_after_arr = new_state.copy()
                            new_state_after_arr[q] += 1
                            new_state_after_arr = tuple(new_state_after_arr)
                            index_new = int(val_list.index(new_state_after_arr))
                            P[int(index),  ind, index_new] += prob_arrival[q]
                            r[int(index), ind, index_new] +=  rewards_ind[str(edge)] - eta * sum_queues 
                        else:           
                            index_new = int(val_list.index(tuple(new_state)))
                            P[int(index), ind, index_new] += prob_arrival[q]
                            r[int(index), ind, index_new] = rewards_ind[str(edge)] - eta * sum_queues
    #check
            elif edge == 'no_move':
                if potential_matches==0:
                    for q in range(len(state)):
                        if np.array(state)[q] < queue_max:
                            new_state = np.array(state) 
                            new_state[q] += 1
                            sum_queues = np.sum(state)
                            new_state = tuple(new_state)
                            index_new = int(val_list.index(new_state))
                            P[int(index),  ind, index_new] += prob_arrival[q]
                            r[int(index),  ind, index_new] += - eta * sum_queues 
                        else:           
                            sum_queues = np.sum(state)
                            P[int(index), ind, int(index)] += prob_arrival[q]
                            r[int(index), ind, int(index)] = - eta * sum_queues 
                        
    #normalise rewards
    r_norm = r + cLI
    #check
    # for s in range(num_states):
    #     for e in range(num_exp):
    #         if (np.abs(np.sum(P[s,e, :]) - 1) > 1e-5) and (np.sum(P[s,e, :]) != 0):
    #             print('Error: transition probabilities do not some to 1', np.sum(P[s,:]), s)
    
    return P, r_norm
                        
def compute_value_bellman(P, r):
    I = np.eye(num_states)
    inv = np.linalg.inv(I - discount*P)
    V = np.matmul(inv, r)
    return V