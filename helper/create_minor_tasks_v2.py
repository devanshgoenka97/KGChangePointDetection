from tqdm import tqdm
import random
import numpy as np
import csv
import sys

# Set seed to create/re-create shuffled data
np.random.seed(1234)
random.seed(1234)

# Initializing as set to get unique entities and relations
ENTITIES = set()
RELATIONS = set()
TRIPLETS = dict()

DEFAULT_ADD_PERCENT = 1.1
DEFAULT_DROP_PERCENT = 1.0

INITIAL_COOLDOWN = 20
CHANGE_MULTIPLIERS = [4.0, 5.0, 6.0]

if len(sys.argv) < 3:
    print("Error: Expected number of timesteps and train/test in argument")
    exit()

timesteps = int(sys.argv[1])
dataset = sys.argv[2]
DATASET = sys.argv[3] if len(sys.argv) > 3 else 'FB15K-237'

# Read original entities and relations in original set
with open(f'./data/{DATASET}/original_{dataset}.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    for row in tqdm(reader):
        (head, relation, tail) = row
        ENTITIES.add(head)
        ENTITIES.add(tail)
        RELATIONS.add(relation)
        # TRIPLETS.append((head, relation, tail))
        if (head not in TRIPLETS):
            TRIPLETS[head] = dict()
        
        if (tail not in TRIPLETS):
            TRIPLETS[tail] = dict()
        
        TRIPLETS[head][tail] = relation

# Converting to list to make these data structures subscriptable
ENTITIES = list(ENTITIES)
RELATIONS = list(RELATIONS)

MAJOR_ENTITIES = set()
MAJOR_RELATIONS = set()
MAJOR_TRIPLETS = []

# Read entities and relations from major task set
with open(f'./data/{DATASET}/{dataset}.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    for row in tqdm(reader):
        (head, relation, tail) = row
        MAJOR_ENTITIES.add(head)
        MAJOR_ENTITIES.add(tail)
        MAJOR_RELATIONS.add(relation)
        MAJOR_TRIPLETS.append((head, relation, tail))

# Converting to list to make these data structures subscriptable
MAJOR_ENTITIES = list(MAJOR_ENTITIES)
MAJOR_RELATIONS = list(MAJOR_RELATIONS)

num_maj_ents = len(MAJOR_ENTITIES)

# Function to store triplets to disk
def store_triplets(triplets, change, timestep=None):
    timestep_name = f"timestep_{timestep}_{change}_change"
    filename = f"{dataset}_{timestep_name}.txt"

    with open(f'./{dataset}data/' + filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter ='\t',quotechar =',',quoting=csv.QUOTE_MINIMAL)
        for (head, relation, tail) in triplets:
            writer.writerow([head, relation, tail])

# Keep track of the entitities in the previous timestep (at t=0)
CURRENT_ENTITIES = MAJOR_ENTITIES

# Now for each timestep, perform bounded addition and deletion and create sub test graphs
for i in range(timesteps):
    print(f"Generating KG for timestep {i+1}...")
    change_mult = 1.0

    # Pick a lottery for a change multiplier
    if i + 1 > INITIAL_COOLDOWN and np.random.uniform() <= 0.20:
        print(f"Picking timestep: {i+1} as a change point")
        change_mult = random.sample(CHANGE_MULTIPLIERS, 1)[0]

    entities_add = int(len(CURRENT_ENTITIES) * ((DEFAULT_ADD_PERCENT * change_mult)/100.0))
    entities_sub = int(len(CURRENT_ENTITIES) * ((DEFAULT_DROP_PERCENT * change_mult)/100.0))

    # Randomly drop entities_sub number of entities from subgraph
    MINOR_ENTITIES = []

    contributing_entities = [e for e in CURRENT_ENTITIES]
    non_contributing_entities = [e for e in ENTITIES if e not in CURRENT_ENTITIES]

    # Pick all but entities_sub entities from the current list
    MINOR_ENTITIES.extend(random.sample(contributing_entities, len(CURRENT_ENTITIES) - entities_sub))

    # Add some entities from non-used ones
    MINOR_ENTITIES.extend(random.sample(non_contributing_entities, min(entities_add, len(non_contributing_entities))))

    MINOR_TASK_TRIPLETS = []
    '''
    for (head, relation, tail) in tqdm(TRIPLETS):
        # All entities and relations in this triplet belong to the 
        if head in MINOR_ENTITIES and tail in MINOR_ENTITIES:
            MINOR_TASK_TRIPLETS.append((head, relation, tail))
    '''
    for head in MINOR_ENTITIES:
        for tail in TRIPLETS[head]:
            if (tail in MINOR_ENTITIES):
                MINOR_TASK_TRIPLETS.append((head, TRIPLETS[head][tail], tail))

    store_triplets(MINOR_TASK_TRIPLETS, change_mult, i+1)
    CURRENT_ENTITIES = MINOR_ENTITIES

print("Done generating synthetic KGs.")
