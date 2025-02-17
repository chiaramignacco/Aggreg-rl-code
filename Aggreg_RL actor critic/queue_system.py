import numpy as np
import itertools
import random

# Setting
verteces = ['A', 'B', 'C', 'D']

graph = {
    'A': [('B', 200), ('C', 10), ('D', 50)],
    'B': [('A', 200), ('D', 20)],
    'C': [('A', 10), ('D', 1)],
    'D': [('A', 50), ('B', 20), ('C', 1)]
}

arrival_rates = {'A': 0.9, 'B': 0.2, 'C': 0.5, 'D': 0.6}

arrival_rates_ind = {ind: rate for ind, (vertex, rate) in enumerate(arrival_rates.items())}

verteces_ind = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

eta = 0.5
discount = 0.8
H = 50
k = 10
queue_max = 5

edges = [(u, v, w) for u in graph for v, w in graph[u]]
edges_list = []

rewards_ind = {}
for edge in edges:
    u, v, w = edge
    e = [verteces_ind[u], verteces_ind[v]]
    if e[::-1] not in edges_list:
        edges_list.append(e)
        rewards_ind[str(e)] = w

# experts = edges_list

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

edges_ind = {i: edge for i, edge in enumerate(edges_list)}
edges_ind[len(edges_list)] = 'no_move'

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