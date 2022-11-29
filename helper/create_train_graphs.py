# -*- coding: utf-8 -*-
"""
Python program to create a series of changing graphs in triplet format
as training data

@author: Bo-Hsun Chen
"""


from tqdm import tqdm
import numpy as np
import csv
import sys
import random

### static parameter setting ###
NumRelations = 1000 # number of relations in each sub-graph
SubgraphChangePercent = 4 # [%], percentage of relations to change in the sub-graph
NumSubgraphs = 100 # number of sub-graphs in series
SaveType = "train" # "train", "valid", or "test"
DataDelimiter = '\t'
dataset_name = "FB15K-237"
total_train_path = "../data/" + dataset_name + "/original_train.txt"

seed_number = 1234
folder_name = "../data/" + dataset_name + "/" + SaveType + "data/"

# global parameter setting
NumTotalEntity = 0
NumRelationType = 0
NumTotalRelations = None
NumRelationChange = int(NumRelations * (SubgraphChangePercent/100.0))

np.random.seed(seed_number)
random.seed(seed_number)

#############################
#### Function Set Starts ####
#############################

### build the total graph from the total train data ###
def read_total_graph(total_train_path):
    global NumTotalEntity, NumRelationType, NumTotalRelations, EntityID_Map
    global RelationTypeID_Map
    print("Reading the total graph...")
    
    ## parameter initialize
    entity_ID_map = dict() # map of entities to IDs
    relation_type_ID_map = dict() # map of relation types to IDs
    total_triplets = []
    
    with open(total_train_path, 'r') as f:
        reader = csv.reader(f, delimiter=DataDelimiter)
    
        for row in tqdm(reader):
            (head, relation, tail) = row
            ## give IDs to the head and tail entities if never seen
            if (head not in entity_ID_map):
                entity_ID_map[head] = NumTotalEntity
                NumTotalEntity += 1
            
            if (tail not in entity_ID_map):
                entity_ID_map[tail] = NumTotalEntity
                NumTotalEntity += 1
            
            ## give ID to the relation type if never seen
            if (relation not in relation_type_ID_map):
                relation_type_ID_map[relation] = NumRelationType
                NumRelationType += 1
                
            head_ID = entity_ID_map[head]
            tail_ID = entity_ID_map[tail]
            relation_type_ID = relation_type_ID_map[relation]
            
            ## add relation into the triplet list
            total_triplets.append([head_ID, relation_type_ID, tail_ID])
    
    NumTotalRelations = len(total_triplets)
    total_triplets = np.vstack(total_triplets)
    return total_triplets, entity_ID_map, relation_type_ID_map


### form the list of relation IDs of a sub-graph ###
def form_relation_IDs():
    global NumRelations, NumTotalRelations
    
    ## randomly choose some relations to form the graph
    relation_IDs = np.random.choice(NumTotalRelations, NumRelations, replace=False)
    
    return relation_IDs


### form the graph after change ###
def change_relation_IDs(relation_IDs, removed_relation_IDs):
    global NumTotalRelations, NumRelationChange
    ## randomly choose some new relation_IDs to add
    relation_IDs_to_add = np.array([], dtype='int')
    count_relation_change = 0
    while(True):
        x = np.random.choice(NumTotalRelations)
        if ((x not in relation_IDs) and (x not in removed_relation_IDs)):
            relation_IDs_to_add = np.append(relation_IDs_to_add, x)
            count_relation_change += 1
            if (count_relation_change >= NumRelationChange//2):
                break
    
    ## finalize the new relation_IDs
    # new_relation_IDs = np.append(relation_IDs_to_add, relation_IDs[0:len(relation_IDs) - (NumRelationChange//2)])
    # return new_relation_IDs, relation_IDs[len(relation_IDs) - (NumRelationChange//2):]
    new_relation_IDs = np.append(relation_IDs[(NumRelationChange//2):], relation_IDs_to_add)
    return new_relation_IDs, relation_IDs[:(NumRelationChange//2)]


### convert relation IDs of a sub-graph to the triplet list ###
# def graph_ID_to_triplet_list(relation_IDs, total_triplets):
#     triplet_arr = []
#     for relation_ID in relation_IDs:
#         triplet_list.append(total_triplet_list[relation_ID])
    
#     triplet_list
#     return triplet_arr


### save triplet list of a sub-graph to disk ###
def save_triplets(triplets, folder_name, timestep=None):
    global SubgraphChangePercent, SaveType
    timestep_name = f"timestep_{timestep}_change_{SubgraphChangePercent}" if (timestep is not None) else "total"
    file_name = f"{SaveType}_{timestep_name}.csv"
    np.savetxt(folder_name + file_name, triplets, delimiter=",", fmt="%d")
    

### save ID maps to disk in csv file ###
def save_ID_map(ID_map, folder_name, map_name):
    keys_sorted_by_ID = sorted(ID_map, key=ID_map.get)
    
    with open(folder_name + map_name + ".txt", 'w') as f:
        for key_name in keys_sorted_by_ID:
            f.write(f"{key_name}\n")

###########################
#### Function Set Ends ####
###########################

# if __name__ == 'main':
    
## get the total triplet list of the total graph
total_triplets, entity_ID_map, relation_type_ID_map = read_total_graph(total_train_path)

## save the entities/relation_types-to-ID maps and total triplets to disk
save_ID_map(entity_ID_map, folder_name, "entity_ID_map")
save_ID_map(relation_type_ID_map, folder_name, "relation_type_ID_map")
save_triplets(total_triplets, folder_name, timestep=None)

## build and save the initial graph in step 0
time_step = 0
relation_IDs = form_relation_IDs()
triplets = total_triplets[relation_IDs]
save_triplets(triplets, folder_name, timestep=time_step)

## build and save the initial graph in step 0
relation_IDs_list = [] # test
removed_relation_IDs = np.array([], dtype='int')
for time_step in range(1, NumSubgraphs):
    relation_IDs, removed_relation_IDs = change_relation_IDs(relation_IDs, removed_relation_IDs)
    relation_IDs_list.append(relation_IDs)
    triplets = total_triplets[relation_IDs]
    save_triplets(triplets, folder_name, timestep=time_step)

    
    
    
    
    
    
    
    
    