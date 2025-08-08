from rdkit.Chem import AllChem, ChemicalFeatures, RDConfig
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform

from Bio.PDB import PDBParser

from rdkit import Chem
from rdkit.Chem.rdchem import BondType



class Protein(object):
    '''Parses .pdb files to extract residues, alpha carbon positions, hydrophobicity scores, '''

    # Dict containing amino acid prefixes
    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

    # Enumerate amino acids
    AA_TO_NUMBER = {v: i for i, v in enumerate(AA_NAME_SYM.values())}

    # Dictionary containing hydrophobicity scores for each amino acid
    KYTE_DOOLITTLE_SCALE = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}

    def __init__(self, pdb_path, threshold=8.0):
        '''
        Initialize protein features

        Parameter pdb_path: filepath to .pdb file
        Parameter threshold: user-defined value for distance threshold (otherwise set to 8 Angstroms)
        '''
        self.pdb_path = pdb_path
        self.threshold = threshold
        self.parser = PDBParser(QUIET=True)
        self.is_valid = False # Introduce validity check
        self.residues = []
        self.pos_CA = []
        self.hydrophobicity_scores = []
        self.residue_connections = []  # For edge_index
        self.residue_distances = []    # For edge_attr
        self.parse_protein()
        self.calculate_residue_edges()

    def parse_protein(self):
        '''
        Extracts previously defined protein features from .pdb file
        '''

        # Check if .pdb contains valid protein structure
        try:
            self.structure = self.parser.get_structure('Protein', self.pdb_path)
            model = next(self.structure.get_models())

            for chain in model:
                for residue in chain.get_residues():
                    res_name = residue.get_resname()
                    if res_name in self.AA_NAME_SYM and residue.has_id('CA'):
                        ca_atom = residue['CA']
                        aa_name = self.AA_NAME_SYM[res_name]
                        aa_number = self.AA_TO_NUMBER[aa_name]
                        hydro_score = self.KYTE_DOOLITTLE_SCALE.get(aa_name, 0)

                        self.residues.append(aa_number)
                        self.pos_CA.append(ca_atom.get_coord())
                        self.hydrophobicity_scores.append(hydro_score)

            if self.residues and self.pos_CA:
                self.is_valid = True
        except Exception as e:
            print(f"An error occurred while processing {self.pdb_path}: {e}")

    def calculate_residue_edges(self):
        if not self.is_valid:
            return

        distances = squareform(pdist(self.pos_CA))
        edge_indices = np.where((distances > 0) & (distances <= self.threshold))
        self.residue_connections = np.stack((edge_indices[0], edge_indices[1]), axis=0)
        self.residue_distances = distances[edge_indices[0], edge_indices[1]]


    def to_dict(self):
        return {
            'residues': self.residues,
            'pos_CA': self.pos_CA,
            'connections': self.residue_connections,
            'distances': self.residue_distances,
            'hydrophobicity_scores': self.hydrophobicity_scores
        }


class Ligand(object):
    ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe',
                     'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
    ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
    BOND_TYPES = {
        BondType.UNSPECIFIED: 0,
        BondType.SINGLE: 1,
        BondType.DOUBLE: 2,
        BondType.TRIPLE: 3,
        BondType.AROMATIC: 4,
    }
    BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
    HYBRIDIZATION_TYPE = ['UNSPECIFIED', 'S', 'SP', 'SP2',
                          'SP2D', 'SP2D2', 'SP3', 'SP3D', 'SP3D2']
    HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}

    def __init__(self, mol2_path):
        self.mol2_path = mol2_path
        self.is_valid = False
        self.molecule = self.load_molecule(mol2_path)

        if self.molecule:
            self._parse_ligand()
            self.is_valid = True
        else:
            print(f"Unable to load or correctly parse: {mol2_path}. Skip/flag as needed.")

    def load_molecule(self, mol2_path):
        try:
            mol = Chem.MolFromMol2File(mol2_path, sanitize=False, removeHs=False)
            if mol is None:
                print(f"Error loading molecule from path: {mol2_path}")
                return None

            mol = Chem.AddHs(mol)
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                print(f"Sanitization or valence issue with {mol2_path}: {e}")
                return None
            return mol
        except Exception as e:
            print(f"Error processing {mol2_path}: {e}")
            return None

    def _parse_ligand(self):
        self.element = []
        self.pos = []
        self.bond_index = []
        self.bond_type = []
        self.atom_features = []
        self.hybridization = []

        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        feats = factory.GetFeaturesForMol(self.molecule)

        conf = self.molecule.GetConformer()
        for atom in self.molecule.GetAtoms():
            self.element.append(atom.GetAtomicNum())
            atom_idx = atom.GetIdx()
            self.pos.append(list(conf.GetAtomPosition(atom_idx)))
            # Use .name if available; otherwise, verify the string conversion
            self.hybridization.append(self.HYBRIDIZATION_TYPE_ID[str(atom.GetHybridization())])

            atom_feat = [0] * len(self.ATOM_FAMILIES)
            for feat in feats:
                if atom_idx in feat.GetAtomIds():
                    atom_feat[self.ATOM_FAMILIES_ID[feat.GetFamily()]] = 1
            self.atom_features.append(atom_feat)

        for bond in self.molecule.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_val = self.BOND_TYPES.get(bond.GetBondType(), 0)
            # Add both directions for undirected graphs
            self.bond_index.append([i, j])
            self.bond_index.append([j, i])
            self.bond_type.extend([bond_val, bond_val])

        self.element = np.array(self.element, dtype=np.int32)
        self.pos = np.array(self.pos, dtype=np.float32)
        self.bond_index = np.array(self.bond_index, dtype=np.int32).T
        self.bond_type = np.array(self.bond_type, dtype=np.int32)
        self.atom_features = np.array(self.atom_features, dtype=np.int32)
        self.hybridization = np.array(self.hybridization, dtype=np.int32)

    def to_dict(self):
        return {
            'element': self.element,
            'pos': self.pos,
            'bond_index': self.bond_index,
            'bond_type': self.bond_type,
            'atom_features': self.atom_features,
            'hybridization': self.hybridization,
        }

#if __name__ == "__main__":
protein_path = '/Users/vedantajain/Research/DeNovo3D/6ugn_pocket.pdb'
ligand_path = '/Users/vedantajain/Research/DeNovo3D/6ugn_ligand.mol2'

protein = Protein(protein_path).to_dict()
ligand = Ligand(ligand_path).to_dict()

print("Protein features:", protein)
print("Ligand features:", ligand)