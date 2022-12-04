from tqdm import tqdm
import numpy as np
import csv
import sys

# Initializing as set to get unique entities and relations
ENTITIES = set()
RELATIONS = set()
TRIPLETS = []

if len(sys.argv) < 2:
    print("Error: Need argument as the subgraph change percent for the major task")
    exit()

SUBGRAPH_CHANGE_PERCENT = int(sys.argv[1])
DATASET = sys.argv[2] if len(sys.argv) > 2 else 'FB15K-237'

# Creating major task train and test sets
print(f"Processing input data for the dataset: {DATASET}")

with open(f'./data/{DATASET}/original_train.txt', 'r') as f:
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

# Get a rough estimate of the number of unique entities and relations in each subgraph
entities_in_subgraph = int((SUBGRAPH_CHANGE_PERCENT / 100.0) * no_entities)
relations_in_subgraph = no_relations

print(f"Generating KG for major task first...")

# Create major task dataset
MAJOR_ENTITIES = list(np.random.choice(ENTITIES, size=entities_in_subgraph))
MAJOR_RELATIONS = list(RELATIONS)

TRAIN_TRIPLETS = []

for (head, relation, tail) in tqdm(TRIPLETS):
        # All entities and relations in this triplet belong to the 
        if head in MAJOR_ENTITIES and tail in MAJOR_ENTITIES and relation in MAJOR_RELATIONS:
            TRAIN_TRIPLETS.append((head, relation, tail))

with open('new_train.txt', 'w') as f:
        writer = csv.writer(f, delimiter ='\t',quotechar =',',quoting=csv.QUOTE_MINIMAL)
        for (head, relation, tail) in TRAIN_TRIPLETS:
            writer.writerow([head, relation, tail])

# Now create validation and test set

VALID_TRIPLETS = []
with open(f'./data/{DATASET}/original_valid.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    for row in tqdm(reader):
        (head, relation, tail) = row
        if head in MAJOR_ENTITIES and relation in MAJOR_RELATIONS and tail in MAJOR_ENTITIES:
            VALID_TRIPLETS.append((head, relation, tail))

with open('new_valid.txt', 'w') as f:
        writer = csv.writer(f, delimiter ='\t',quotechar =',',quoting=csv.QUOTE_MINIMAL)
        for (head, relation, tail) in VALID_TRIPLETS:
            writer.writerow([head, relation, tail])

TEST_TRIPLETS = []
with open(f'./data/{DATASET}/original_test.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    for row in tqdm(reader):
        (head, relation, tail) = row
        if head in MAJOR_ENTITIES and relation in MAJOR_RELATIONS and tail in MAJOR_ENTITIES:
            TEST_TRIPLETS.append((head, relation, tail))

with open('new_test.txt', 'w') as f:
        writer = csv.writer(f, delimiter ='\t',quotechar =',',quoting=csv.QUOTE_MINIMAL)
        for (head, relation, tail) in TEST_TRIPLETS:
            writer.writerow([head, relation, tail])