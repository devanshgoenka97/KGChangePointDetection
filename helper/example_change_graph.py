from tqdm import tqdm
import numpy as np
import csv
import sys

# parameters
num_relations = 10000
num_relation_change = 1000
np.random.seed(1234)

# maps of entities and relations to IDs
entity_ID_map = dict()
relation_type_ID_map = dict()
num_total_entity = 0
num_total_relation = 0
num_total_relation_type = 1

total_adj_list = []
total_relation_list = []
'''
mylist = []
def add_node(node):
  if node not in mylist:
    mylist.append(node)
  else:
    print("Node ",node," already exists!")
 
def add_edge(node1, node2, weight):
  temp = []
  if node1 in mylist and node2 in mylist:
    if node1 not in adj_list:
      temp.append([node2,weight])
      adj_list[node1] = temp
   
    elif node1 in adj_list:
      temp.extend(adj_list[node1])
      temp.append([node2,weight])
      adj_list[node1] = temp
       
  else:
    print("Nodes don't exist!")
'''

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
        
        ## give IDs to the relations if never seen
        if (relation not in relation_type_ID_map):
            relation_type_ID_map[relation] = num_total_relation_type
            num_total_relation_type += 1
            
        head_ID = entity_ID_map[head]
        tail_ID = entity_ID_map[tail]
        relation_type_ID = relation_type_ID_map[relation]
        
        ## add relation into the adjacency list
        total_adj_list[head_ID][tail_ID] = relation_type_ID
        total_adj_list[tail_ID][head_ID] = - relation_type_ID
        total_relation_list.append([head_ID, relation_type_ID, tail_ID])
        num_total_relation += 1

### randomly choose some entities to form the sub-graph ###
subgraph_IDs = np.random.choice(num_total_relation, num_relations, replace=False)
adj_dict = dict()
# count_num_entity = 0
for relation_ID in subgraph_IDs:
    head_ID, relation_type_ID, tail_ID = total_relation_list[relation_ID]
    if (head_ID not in adj_dict):
        adj_dict[head_ID] = dict()
        
    if (tail_ID not in adj_dict):
        adj_dict[tail_ID] = dict()

    adj_dict[head_ID][tail_ID] = relation_type_ID
    adj_dict[tail_ID][head_ID] = - relation_type_ID
    
#     for v_ID in adj_dict

# to make a change, delete and add some entites into the subgraph_IDs
# count_num_relation_change = 0
# to_delete_IDs = subgraph_IDs[-int(num_entity_change/2):]



'''
print("Done processing input data.")

no_entities = len(ENTITIES)
no_relations = len(RELATIONS)
no_triplets = len(TRIPLETS)

# Converting to list to make these data structures subscriptable
ENTITIES = list(ENTITIES)
RELATIONS = list(RELATIONS)

# Get the number of change points to generate
change_points = int(sys.argv[1])

# Get a rough estimate of the number of unique entities and relations in each subgraph
entities_in_subgraph = no_entities // change_points
relations_in_subgraph = no_relations // change_points


print("Generating synthetic KGs for change points...")

# For each change point, sample indices uniformly
for i in range(change_points):
    entity_indices = np.random.randint(no_entities, size=entities_in_subgraph)
    relation_indices = np.random.randint(no_relations, size=relations_in_subgraph)

    # Generate the entity and relation subsets 
    entity_subgraph = [ENTITIES[idx] for idx in entity_indices]
    relation_subgraph = [RELATIONS[idx] for idx in relation_indices]

    print(f"Generating KG for change point {i}...")

    for (head, relation, tail) in tqdm(TRIPLETS):
        # All entities and relations in this triplet belong to the 
        if head in entity_subgraph and tail in entity_subgraph and relation in relation_subgraph:
            pass
            # Save this triplet to a new file for the dataset

print("Done generating synthetic KGs.")
'''