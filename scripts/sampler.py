from tqdm import tqdm
import numpy as np
import csv
import sys

# Initializing as set to get unique entities and relations
ENTITIES = set()
RELATIONS = set()
TRIPLETS = []

# Expect an argument as the number of change points
if len(sys.argv) <= 1:
    print("Number of change points not specified")
    exit()

print("Processing input data...")
with open('../FB15K-237/train.txt', 'r') as f:
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