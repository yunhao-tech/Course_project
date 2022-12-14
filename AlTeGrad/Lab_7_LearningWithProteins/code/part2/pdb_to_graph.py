"""
Learning on Sets / Learning with Proteins - ALTEGRAD - Dec 2022
"""

import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.visualisation import plot_protein_structure_graph
from graphein.protein.analysis import plot_degree_by_residue_type, plot_edge_type_distribution, plot_residue_composition
from graphein.protein.edges.distance import add_peptide_bonds, add_hydrogen_bond_interactions, add_disulfide_interactions, add_ionic_interactions, add_aromatic_interactions, add_aromatic_sulphur_interactions, add_cation_pi_interactions, add_distance_threshold, add_k_nn_edges
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot, expasy_protein_scale, meiler_embedding
from graphein.protein.utils import download_alphafold_structure

# Configuration object for graph construction
config = ProteinGraphConfig(**{"node_metadata_functions": [amino_acid_one_hot, 
                                                           expasy_protein_scale,
                                                           meiler_embedding],
                               "edge_construction_functions": [add_peptide_bonds,
                                                  add_aromatic_interactions,
                                                  add_hydrogen_bond_interactions,
                                                  add_disulfide_interactions,
                                                  add_ionic_interactions,
                                                  add_aromatic_sulphur_interactions,
                                                  add_cation_pi_interactions,
                                                  partial(add_distance_threshold, long_interaction_threshold=5, threshold=10.),
                                                  partial(add_k_nn_edges, k=3, long_interaction_threshold=2)],
                               })
# we add edges between nodes, in multiple cases. 
PDB_CODE = "Q5VSL9"


############## Task 8
    
##################
# your code here #
protein_path = download_alphafold_structure(PDB_CODE, aligned_score=False)
G = construct_graph(protein_path, config=config)
##################

# Print number of nodes and number of edges
print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())


############## Task 9

##################
# your code here #
seq_degree = [G.degree(node) for node in G.nodes()]
print(f"Max degree: {np.max(seq_degree)}")
print(f"Min degree: {np.min(seq_degree)}")
print(f"Mean degree: {np.mean(seq_degree)}")

p = plot_degree_by_residue_type(G)
p.write_image('degree_by_residue_type')

p2 = plot_edge_type_distribution(G)
p2.write_image('residue_composition')


p4 = plot_residue_composition(G)
p4.write_image('residue_composition')

p3 = plot_protein_structure_graph(G)
p3.write_image('protein_structure')


##################