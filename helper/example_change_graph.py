from tqdm import tqdm
import numpy as np
import csv
import sys
import random

# parameters
num_relations = 10000
num_relation_change = 1000
np.random.seed(1234)
random.seed(1234)

# maps of entities and relations to IDs
entity_ID_map = dict()
relation_type_ID_map = dict()
num_total_entity = 0
num_total_relation = 0
num_total_relation_type = 1

total_adj_list = []
total_relation_list = []

### build the total graph of the input data ###
print("Processing input data...")
with open('../data/FB15K-237/train.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    for row in tqdm(reader):
        (head, relation, tail) = row
        ## give IDs to the head and tail entities if never seen
        if (head not in entity_ID_map):
            entity_ID_map[head] = num_total_entity
            num_total_entity += 1
            total_adj_list.append(dict())
        
        if (tail not in entity_ID_map):
            entity_ID_map[tail] = num_total_entity
            num_total_entity += 1
            total_adj_list.append(dict())
        
        ## give ID to the relation type if never seen
        if (relation not in relation_type_ID_map):
            relation_type_ID_map[relation] = num_total_relation_type
            num_total_relation_type += 1
            
        head_ID = entity_ID_map[head]
        tail_ID = entity_ID_map[tail]
        relation_type_ID = relation_type_ID_map[relation]
        
        ## add relation into the total adjacency list
        if (head_ID != tail_ID):
            total_adj_list[head_ID][tail_ID] = relation_type_ID
            total_adj_list[tail_ID][head_ID] = - relation_type_ID
            total_relation_list.append([head_ID, relation_type_ID, tail_ID])
            num_total_relation += 1

### form the graph before change ###
## randomly choose some relations to form the graph
graph_IDs = np.random.choice(num_total_relation, num_relations, replace=False)
adj_dict = dict()
for relation_ID in graph_IDs:
    head_ID, relation_type_ID, tail_ID = total_relation_list[relation_ID]
    if (head_ID not in adj_dict):
        adj_dict[head_ID] = dict()
        
    if (tail_ID not in adj_dict):
        adj_dict[tail_ID] = dict()

    adj_dict[head_ID][tail_ID] = relation_type_ID
    adj_dict[tail_ID][head_ID] = - relation_type_ID

### form the graph after change ###
new_adj_dict = adj_dict.copy()

## randomly choose some new relation_IDs to add
graph_IDs_to_add = np.array([], dtype='int')
count_relation_change = 0
while(True):
    x = np.random.choice(num_total_relation)
    if (x not in graph_IDs):
        graph_IDs_to_add = np.append(graph_IDs_to_add, x)
        count_relation_change += 1
        if (count_relation_change >= num_relation_change//2):
            break

## delete the original last num_relation_change/2 relations in the new graph
for relation_ID in graph_IDs[-(num_relation_change//2):]:
    head_ID, relation_type_ID, tail_ID = total_relation_list[relation_ID]
    new_adj_dict[head_ID].pop(tail_ID, None)
    if (len(new_adj_dict[head_ID]) == 0):
        new_adj_dict.pop(head_ID, None)
        
    new_adj_dict[tail_ID].pop(head_ID, None)
    if (len(new_adj_dict[tail_ID]) == 0):
        new_adj_dict.pop(tail_ID, None)    
    
## add new relations into the new graph
for relation_ID in graph_IDs_to_add:
    head_ID, relation_type_ID, tail_ID = total_relation_list[relation_ID]
    if (head_ID not in new_adj_dict):
        new_adj_dict[head_ID] = dict()
        
    if (tail_ID not in new_adj_dict):
        new_adj_dict[tail_ID] = dict()

    new_adj_dict[head_ID][tail_ID] = relation_type_ID
    new_adj_dict[tail_ID][head_ID] = - relation_type_ID

## finalize the new graph_IDs
new_graph_IDs = np.append(graph_IDs[:len(graph_IDs) - (num_relation_change//2)], graph_IDs_to_add)