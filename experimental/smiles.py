from rdkit import Chem
import networkx as nx
smiles = 'CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1'
mol = Chem.MolFromSmiles(smiles)
adjacency_matrix = Chem.GetAdjacencyMatrix(mol, useBO = True)

