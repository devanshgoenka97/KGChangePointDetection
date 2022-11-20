from tqdm import tqdm
import numpy as np
import csv
import sys

# Initializing as set to get unique entities and relations
TRAIN_ENTITIES = set()
TRAIN_RELATIONS = set()
TRAIN_TRIPLETS = []

with open('./data/FB15K-237/train.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    for row in tqdm(reader):
        (head, relation, tail) = row
        TRAIN_ENTITIES.add(head)
        TRAIN_ENTITIES.add(tail)
        TRAIN_RELATIONS.add(relation)
        TRAIN_TRIPLETS.append((head, relation, tail))

no_entities = len(TRAIN_ENTITIES)
no_relations = len(TRAIN_RELATIONS)
no_triplets = len(TRAIN_TRIPLETS)

# Converting to list to make these data structures subscriptable
ENTITIES = list(TRAIN_ENTITIES)
RELATIONS = list(TRAIN_RELATIONS)

VALID_ENTITIES = set()
VALID_RELATIONS = set()
VALID_TRIPLETS = []
with open('./data/FB15K-237/valid.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    for row in tqdm(reader):
        (head, relation, tail) = row
        if head in TRAIN_ENTITIES and relation in TRAIN_RELATIONS and tail in TRAIN_ENTITIES:
            VALID_ENTITIES.add(head)
            VALID_ENTITIES.add(tail)
            VALID_RELATIONS.add(relation)
            VALID_TRIPLETS.append((head, relation, tail))

with open('new_valid.txt', 'w') as f:
        writer = csv.writer(f, delimiter ='\t',quotechar =',',quoting=csv.QUOTE_MINIMAL)
        for (head, relation, tail) in VALID_TRIPLETS:
            writer.writerow([head, relation, tail])

TEST_ENTITIES = set()
TEST_RELATIONS = set()
TEST_TRIPLETS = []
with open('./data/FB15K-237/test.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    for row in tqdm(reader):
        (head, relation, tail) = row
        if head in TRAIN_ENTITIES and relation in TRAIN_RELATIONS and tail in TRAIN_ENTITIES:
            TEST_ENTITIES.add(head)
            TEST_ENTITIES.add(tail)
            TEST_RELATIONS.add(relation)
            TEST_TRIPLETS.append((head, relation, tail))

with open('new_test.txt', 'w') as f:
        writer = csv.writer(f, delimiter ='\t',quotechar =',',quoting=csv.QUOTE_MINIMAL)
        for (head, relation, tail) in TEST_TRIPLETS:
            writer.writerow([head, relation, tail])