import numpy as np
import csv
import sys

ENTITIES = set()
RELATIONS = set()
TRIPLETS = []

# Expect an argument as the number of change points
if len(sys.argv) <= 1:
    print("Number of change points not specified")
    exit()

with open('../FB15K-237/train.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    for row in reader:
        (head, relation, tail) = row
        ENTITIES.add(head)
        ENTITIES.add(tail)
        RELATIONS.add(relation)
        TRIPLETS.append((head, relation, tail))

no_entities = len(ENTITIES)
no_relations = len(RELATIONS)
no_triplets = len(TRIPLETS)

# Get the number of change points to generate
change_points = sys.argv[1]

# Get a rough estimate of the number of unique entities and relations in each subgraph
entities_in_subgraph = no_entities // change_points
relations_in_subgraph = no_relations // change_points
