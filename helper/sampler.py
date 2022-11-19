from tqdm import tqdm
import numpy as np
import csv
import sys

# Initializing as set to get unique entities and relations
ENTITIES = set()
RELATIONS = set()
TRIPLETS = []
SUBGRAPH_CHANGE_PERCENT = 5.0

# Expect an argument as the number of timesteps and percentage
if len(sys.argv) <= 1:
    print("Number of timesteps and percentage size not specified")
    exit()

print("Processing input data...")
with open('./data/FB15K-237/train.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    for row in tqdm(reader):
        (head, relation, tail) = row
        ENTITIES.add(head)
        ENTITIES.add(tail)
        RELATIONS.add(relation)
        TRIPLETS.append((head, relation, tail))

print("Done processing input data.")

no_entities = len(ENTITIES)
no_relations = len(RELATIONS)
no_triplets = len(TRIPLETS)

# Converting to list to make these data structures subscriptable
ENTITIES = list(ENTITIES)
RELATIONS = list(RELATIONS)

global_entity_indices = [i for i in range(no_entities)]
global_relation_indices = [i for i in range(no_relations)]

# Get the number of timesteps to generate the data
timesteps = int(sys.argv[1])

# Get the approximate size of the sub-graph as percentage
subgraph_size = int(sys.argv[2])

# Get a rough estimate of the number of unique entities and relations in each subgraph
entities_in_subgraph = int((subgraph_size / 100.0) * no_entities)
relations_in_subgraph = int((subgraph_size / 100.0) * no_relations)

print("Generating synthetic KGs for the given timesteps...")

# Function to store triplets to disk
def store_triplets(triplets, is_major, timestep=None):
    timestep_name = f"timestep_{timestep}_{subgraph_size}_change" if not is_major else "major"
    filename = f"test_{timestep_name}.txt"

    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter ='\t',quotechar =',',quoting=csv.QUOTE_MINIMAL)
        for (head, relation, tail) in triplets:
            writer.writerow([head, relation, tail])


print(f"Generating KG for major task first...")

# First create major task dataset

entity_indices = list(np.random.choice(global_entity_indices, size=entities_in_subgraph))
relation_indices = list(np.random.choice(global_relation_indices, size=relations_in_subgraph))

# Generate the entity and relation subsets 
entity_subgraph = [ENTITIES[idx] for idx in entity_indices]
relation_subgraph = [RELATIONS[idx] for idx in relation_indices]

MAJOR_TASK_TRIPLETS = []

for (head, relation, tail) in tqdm(TRIPLETS):
        # All entities and relations in this triplet belong to the 
        if head in entity_subgraph and tail in entity_subgraph and relation in relation_subgraph:
            MAJOR_TASK_TRIPLETS.append((head, relation, tail))

store_triplets(MAJOR_TASK_TRIPLETS, True)

# Now for each timestep, perform bounded addition and deletion and create subgraphs
for i in range(timesteps):
    print(f"Generating KG for timestep {i}...")

    entities_add = int(entities_in_subgraph * (SUBGRAPH_CHANGE_PERCENT/100.0))
    relations_add = int(relations_in_subgraph * (SUBGRAPH_CHANGE_PERCENT/100.0))
    entities_sub = entities_add
    relations_sub = relations_add

    # Randomly drop entities_sub and relations_sub number from their respective subgraphs
    delta_entity_indices = list(np.random.choice(entity_indices, size = entities_in_subgraph - entities_sub))
    delta_relation_indices = list(np.random.choice(relation_indices, size = relations_in_subgraph - relations_sub))

    non_contributing_entity_indices = [idx for idx in global_entity_indices if idx not in entity_indices]
    non_contributing_relation_indices = [idx for idx in global_relation_indices if idx not in relation_indices]

    # Add some entities and relations from non-used ones
    delta_entity_indices.extend(np.random.choice(non_contributing_entity_indices, size = entities_add))
    delta_relation_indices.extend(np.random.choice(non_contributing_relation_indices, size = relations_add))

    delta_entity_subgraph = [ENTITIES[idx] for idx in delta_entity_indices]
    delta_relation_subgraph = [RELATIONS[idx] for idx in delta_relation_indices]

    MINOR_TASK_TRIPLETS = []
    for (head, relation, tail) in tqdm(TRIPLETS):
        # All entities and relations in this triplet belong to the 
        if head in delta_entity_subgraph and tail in delta_entity_subgraph and relation in delta_relation_subgraph:
            MINOR_TASK_TRIPLETS.append((head, relation, tail))

    store_triplets(MINOR_TASK_TRIPLETS, False, i+1)

print("Done generating synthetic KGs.")