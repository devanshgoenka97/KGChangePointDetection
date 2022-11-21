from tqdm import tqdm
import numpy as np
import csv
import sys

# Initializing as set to get unique entities and relations
ENTITIES = set()
RELATIONS = set()
TRIPLETS = []

SUBGRAPH_CHANGE_PERCENT = 5.0
timesteps = int(sys.argv[1])

# Read original entities and relations in test set
with open('./data/FB15K-237/original_test.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    for row in tqdm(reader):
        (head, relation, tail) = row
        ENTITIES.add(head)
        ENTITIES.add(tail)
        RELATIONS.add(relation)
        TRIPLETS.append((head, relation, tail))

no_entities = len(ENTITIES)
no_relations = len(RELATIONS)
no_triplets = len(TRIPLETS)

# Converting to list to make these data structures subscriptable
ENTITIES = list(ENTITIES)
RELATIONS = list(RELATIONS)

MAJOR_ENTITIES = set()
MAJOR_RELATIONS = set()
MAJOR_TRIPLETS = []

# Read entities and relations from major task test set
with open('./data/FB15K-237/test.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    for row in tqdm(reader):
        (head, relation, tail) = row
        MAJOR_ENTITIES.add(head)
        MAJOR_ENTITIES.add(tail)
        MAJOR_RELATIONS.add(relation)
        MAJOR_TRIPLETS.append((head, relation, tail))

maj_entities = len(MAJOR_ENTITIES)
maj_relations = len(MAJOR_RELATIONS)
maj_triplets = len(MAJOR_TRIPLETS)

# Converting to list to make these data structures subscriptable
MAJOR_ENTITIES = list(MAJOR_ENTITIES)
MAJOR_RELATIONS = list(MAJOR_RELATIONS)


# Function to store triplets to disk
def store_triplets(triplets, is_major, timestep=None):
    timestep_name = f"timestep_{timestep}_{SUBGRAPH_CHANGE_PERCENT}_change" if not is_major else "major"
    filename = f"test_{timestep_name}.txt"

    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter ='\t',quotechar =',',quoting=csv.QUOTE_MINIMAL)
        for (head, relation, tail) in triplets:
            writer.writerow([head, relation, tail])

# Now for each timestep, perform bounded addition and deletion and create sub test graphs
for i in range(timesteps):
    print(f"Generating KG for timestep {i}...")
    entities_add = int(maj_entities * (SUBGRAPH_CHANGE_PERCENT/100.0))
    relations_add = int(maj_relations * (SUBGRAPH_CHANGE_PERCENT/100.0))
    entities_sub = entities_add
    relations_sub = relations_add

    # Randomly drop entities_sub and relations_sub number from their respective subgraphs
    MINOR_ENTITIES = list(np.random.choice(MAJOR_ENTITIES, size = maj_entities - entities_sub))
    MINOR_RELATIONS = list(np.random.choice(MAJOR_RELATIONS, size = maj_relations - relations_sub))

    non_contributing_entities = [e for e in ENTITIES if e not in MAJOR_ENTITIES]
    non_contributing_relations = [r for r in RELATIONS if r not in MAJOR_RELATIONS]

    # Add some entities and relations from non-used ones
    MINOR_ENTITIES.extend(np.random.choice(non_contributing_entities, size = entities_add))
    MINOR_RELATIONS.extend(np.random.choice(non_contributing_relations, size = relations_add))

    MINOR_TASK_TRIPLETS = []
    for (head, relation, tail) in tqdm(TRIPLETS):
        # All entities and relations in this triplet belong to the 
        if head in MINOR_ENTITIES and tail in MINOR_ENTITIES and relation in MINOR_RELATIONS:
            MINOR_TASK_TRIPLETS.append((head, relation, tail))

    store_triplets(MINOR_TASK_TRIPLETS, False, i+1)

print("Done generating synthetic KGs.")