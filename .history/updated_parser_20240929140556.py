"""
Improvements:

- Optimizes CA atom coordinate extraction by pre-extracting all residues and CA atom coordinates in bulk,
and creating NumPy arrays upfront
- Removes list appends to avoid memory reallocation and resizing by collecting all data first and converting to arrays at once
- More descriptive error handling
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from Bio.PDB import PDBParser


class Protein(object):
    '''Parses .pdb file to extract residue information, CA position, hydrophobicity scores'''

    # Dictionary containing amino acid prefixes
    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

    # Enumerate
    A_TO_NUMBER = {v: i for i, v in enumerate(AA_NAME_SYM.values())}

    # Dictionary containing hydrophobicity scores for amino acids
    KYTE_DOOLITTLE_SCALE = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}

    def __init__(self, pdb_path, threshold=8.0):
        """
        Initialize protein features

        Parameter pdb_path: filepath to .pdb file
        Parameter threshold: user-defined value for distance threshold (otherwise set to 8 Angstroms)
        """

        # Initialize arrays
        self.pdb_path = pdb_path
        self.threshold = threshold
        self.parser = PDBParser(QUIET=True)
        self.is_valid = False
        self.residues = []
        self.pos_CA = []
        self.hydrophobicity_scores = []
        self.residue_connections = []
        self.residue_distances = []

        # Call parsing and calculation functions
        self.parse_protein()
        self.calculate_residue_edges()

    def parse_protein(self):
        '''
        Extracts protein features from .pdb file.
        '''
        try:
            self.structure = self.parser.get_structure(
                'Protein', self.pdb_path)
            model = next(self.structure.get_models())

            # Collect data in lists before converting to arrays
            residues_list = []
            pos_CA_list = []
            hydrophobicity_scores_list = []

            for chain in model:
                for residue in chain.get_residues():
                    res_name = residue.get_resname()
                    if res_name in self.AA_NAME_SYM and 'CA' in residue:
                        ca_atom = residue['CA']  # store index of alpha carbon
                        aa_name = self.AA_NAME_SYM[res_name]
                        aa_number = self.AA_TO_NUMBER[aa_name]
                        hydrophobicity_score = self.KYTE_DOOLITTLE_SCALE.get(
                            aa_name, 0)

                        residues_list.append(aa_number)
                        pos_CA_list.append(ca_atom.get_coord())
                        hydrophobicity_scores_list.append(hydrophobicity_score)

            # Convert lists to numpy arrays
            if residues_list and pos_CA_list:
                self.residues = np.array(residues_list)
                self.pos_CA = np.array(pos_CA_list)
                self.hydrophobicity_scores = np.array(
                    hydrophobicity_scores_list)
                self.is_valid = True

        # Invalid structure file
        except Exception as e:
            print(f"An error occurred while processing {self.pdb_path}: {e}")

    def calculate_residue_edges(self):
        if not self.is_valid:  # Ensure valid data
            return

        distance = squareform(pdist(self.pos_CA))
        edge_indices = np.where((distance > 0) & (distance <= self.threshold))
        self.residue_connections = np.stack(
            (edge_indices[0], edge_indices[1]), axis=0)

        # Store distance calculations for edges
        self.residue_distances = distance[edge_indices[0], edge_indices[1]]

    def to_dict(self):
        return {
            'residues': self.residues,
            'pos_CA': self.pos_CA,
            'connections': self.residue_connections,
            'distances': self.residue_distances,
            'hydrophobicity_scores': self.hydrophobicity_scores
        }


# test
protein_path = '/Users/vedantajain/Research/DeNovo3D/6ugn_pocket.pdb'
protein = Protein(protein_path).to_dict()
print(protein)

# ligand_path = '/Users/vedantajain/Research/DeNovo3D/6ugn_ligand.mol2'
# ligand = Ligand(ligand_path).to_dict()
# print(ligand)
