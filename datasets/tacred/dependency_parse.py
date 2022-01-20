"""
-------------------------------------------------------------------------------------
Exploring Linguistically Enriched Transformers for Low-Resource Relation Extraction:
    --Enriched Attention on PLM
    
    by Pedro G. Bascoy  (Bosch Center for Artificial Intelligence (BCAI)),
    
    with the supervision of
    
    Prof. Dr. Sebastian Padó (Institut für Machinelle Sprachverarbeitung (IMS)),
    and Dr. Heike Adel-Vu  (BCAI).
-------------------------------------------------------------------------------------
"""
from typing import List

import networkx as nx

"""
Functions in this file are from https://github.com/yuhaozhang/tacred-relation.
They extract information from the dependency parse heads of the TACRED dataset.
"""

import networkx as nw

def get_info_from_dependency(e1_idx, e2_idx, tree, context, enhanced_dependencies):
    # for each word: distance to entity 1 and entity 2 in dependency tree
    e1_upmost_node = e1_idx[1]  # heuristic for now
    e2_upmost_node = e2_idx[1]
    path_e1_root = get_path_to_root(e1_upmost_node, tree)
    path_e2_root = get_path_to_root(e2_upmost_node, tree)
    distances_to_e1 = []
    distances_to_e2 = []
    head_dict = tree[0]

    # Pedro: highest distance (for embeddings)
    highest_dependency_distance = 0

    for idx in range(len(head_dict)):

        de1 = get_distance_to_entity(idx, e1_idx, path_e1_root, head_dict)
        de2 = get_distance_to_entity(idx, e2_idx, path_e2_root, head_dict)

        if de1 > highest_dependency_distance:
            highest_dependency_distance = de1

        if de2 > highest_dependency_distance:
            highest_dependency_distance = de2

        distances_to_e1.append(de1)
        distances_to_e2.append(de2)

    shortest_path_idx: List[int]
    shortest_path: List[str]
    # shortest path between e1 and e2
    if enhanced_dependencies == None:
        shortest_path_idx = []
        for e1_indices in range(e1_idx[0], e1_idx[1] + 1):
            shortest_path_idx.append(e1_indices)
        for e1_walker in path_e1_root:
            shortest_path_idx.append(e1_walker)
            if e1_walker in path_e2_root:
                index_on_path_e2 = path_e2_root.index(e1_walker)
                break
        for e2_walker in range(index_on_path_e2 - 1, -1, -1):
            shortest_path_idx.append(path_e2_root[e2_walker])
        for e2_indices in range(e2_idx[0], e2_idx[1] + 1):
            shortest_path_idx.append(e2_indices)
        shortest_path = [context[idx - 1] for idx in shortest_path_idx]

    else:
        G = nw.Graph()
        G.add_edges_from([(i+1,j if j!=0 else 'ROOT') for i, list in enumerate(enhanced_dependencies) for j in list])
        shortest_path_idx = nx.shortest_path(G, source=e1_idx[1]+1, target=e2_idx[1]+1)

        # include entity mentions
        for e1_indices in range(e1_idx[0], e1_idx[1] + 1):
            try:
                shortest_path_idx.remove(e1_indices + 1)
            except ValueError:
                pass
        for e2_indices in range(e2_idx[0], e2_idx[1] + 1):
            try:
                shortest_path_idx.remove(e2_indices + 1)
            except ValueError:
                pass

        shortest_path_idx =  [i+1 for i in range(e1_idx[0], e1_idx[1] + 1)] + shortest_path_idx
        shortest_path_idx.extend([i+1 for i in range(e2_idx[0], e2_idx[1] + 1)])

        shortest_path_idx = [i-1 for i in shortest_path_idx]

        shortest_path = [context[idx] for idx in shortest_path_idx]

    # for each word: whether it is on shortest path between e1 and e2 or not
    flag_on_shortest_path = []
    for idx in range(len(head_dict)):
        if idx in shortest_path_idx:
            flag_on_shortest_path.append(1)
        else:
            flag_on_shortest_path.append(-1)

    return distances_to_e1, distances_to_e2, shortest_path, flag_on_shortest_path, highest_dependency_distance

def get_path_to_root(index, tree):
    head_dict = tree[0]
    path = []
    idx = index
    while idx != "ROOT":
        idx = head_dict[idx]
        path.append(idx)
    if path == ["ROOT"]:
        # do not allow only root on path - that will not work with other parts
        path.insert(0, index)
    return path

def get_distance_to_entity(idx, e_idx, path_ent_root, head_dict):
    distance_to_ent = -1
    if idx >= e_idx[0] and idx <= e_idx[1]:  # it is the entity
        distance_to_ent = 0
    else:
        idx_running = idx
        path_length = 0
        while idx_running != "ROOT":
            if idx_running in path_ent_root:
                distance_to_ent = path_ent_root.index(idx_running) + 1 + path_length
                break
            idx_running = head_dict[idx_running]
            path_length += 1
        if path_ent_root == ['ROOT']:
            distance_to_ent = path_length
    assert(distance_to_ent >= 0)
    return distance_to_ent

def read_dependency_tree(instance):
    heads = instance['stanford_head']
    # rels = instance['stanford_deprel']
    head_dict = {}
    # rel_dict = {}
    for idx, head in enumerate(heads):
        head_dict[idx] = int(head) - 1
        if int(head) == 0:
            head_dict[idx] = "ROOT"
        # rel_dict[idx] = rels[idx]

    # ACE05 does not provide de rel_dict data, so we don't use it.
    # return (head_dict, rel_dict)
    return (head_dict, None)