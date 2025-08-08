import os
import logging
from typing import Dict, Any

import numpy as np
from scipy.spatial import KDTree
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Protein:
    """
    Parses .pdb files to extract residue-level features for graph construction:
      - Residue index, type encoding, alpha-carbon coordinates
      - Sequential backbone edges and spatial proximity edges with types
    """
    AA_NAME_SYM = { 'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F',
                    'GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L',
                    'MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R',
                    'SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y' }
    AA_TO_NUMBER = {v:i for i,v in enumerate(AA_NAME_SYM.values())}
    KYTE_DOOLITTLE = { 'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,
                       'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,
                       'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3 }

    def __init__(self, pdb_path: str, threshold: float = 8.0):
        self.pdb_path = pdb_path
        self.threshold = threshold
        self.parser = PDBParser(QUIET=True)
        self.is_valid = False

        # Raw lists
        self.residues = []
        self.pos_CA = []
        self.hydro_scores = []

        # Graph edges
        self.edges = None
        self.edge_distances = None
        self.edge_types = None

        try:
            self._parse()
            self._compute_edges()
            self._finalize()
            self.is_valid = True
            logger.info(f"Parsed protein {os.path.basename(pdb_path)}: {len(self.residues)} residues, {self.edges.shape[1]} edges")
        except Exception as e:
            logger.error(f"Failed to parse protein {pdb_path}: {e}")

    def _parse(self):
        structure = self.parser.get_structure('prot', self.pdb_path)
        model = next(structure.get_models())
        for chain in model:
            for residue in chain.get_residues():
                resname = residue.get_resname()
                if resname not in self.AA_NAME_SYM or 'CA' not in residue:
                    continue
                aa = self.AA_NAME_SYM[resname]
                idx = self.AA_TO_NUMBER[aa]
                coord = residue['CA'].get_coord()
                hydro = self.KYTE_DOOLITTLE.get(aa, 0.0)
                self.residues.append(idx)
                self.pos_CA.append(coord)
                self.hydro_scores.append(hydro)

        if not self.residues:
            raise ValueError("No valid residues extracted")

    def _compute_edges(self):
        coords = np.vstack(self.pos_CA)
        # Sequential backbone edges (type=0)
        n = coords.shape[0]
        backbone_i = np.arange(n - 1)
        backbone_j = backbone_i + 1
        backbone_pairs = np.vstack([backbone_i, backbone_j])
        backbone_dists = np.linalg.norm(coords[backbone_i] - coords[backbone_j], axis=1)
        backbone_types = np.zeros(len(backbone_dists), dtype=np.int64)

        # Spatial proximity edges (type=1)
        tree = KDTree(coords)
        prox_pairs_set = tree.query_pairs(r=self.threshold)
        prox_pairs = np.array(list(prox_pairs_set)).T
        if prox_pairs.size == 0:
            raise ValueError("No spatial edges within threshold")
        prox_i, prox_j = prox_pairs
        prox_dists = np.linalg.norm(coords[prox_i] - coords[prox_j], axis=1)
        prox_types = np.ones(len(prox_dists), dtype=np.int64)

        # Combine edges
        self.edges = np.hstack([backbone_pairs, prox_pairs])
        self.edge_distances = np.concatenate([backbone_dists, prox_dists])
        self.edge_types = np.concatenate([backbone_types, prox_types])

    def _finalize(self):
        # Convert lists to arrays and validate shapes
        self.residues = np.array(self.residues, dtype=np.int64)
        self.pos_CA = np.array(self.pos_CA, dtype=np.float32)
        self.hydro_scores = np.array(self.hydro_scores, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'residues': self.residues,
            'pos_CA': self.pos_CA,
            'hydrophobicity': self.hydro_scores,
            'edge_index': self.edges,
            'edge_distance': self.edge_distances,
            'edge_type': self.edge_types
        }

class Ligand:
    """
    Parses .mol2 (or SDF) ligand files to extract atom-level and bond-level features:
      - Atomic numbers, 3D coordinates, Gasteiger charges
      - Hybridization, aromaticity, ring membership, topological descriptors
      - Bond indices and types
    """
    def __init__(self, mol_path: str, add_charges: bool = True):
        self.mol_path = mol_path
        self.add_charges = add_charges
        self.is_valid = False
        try:
            self._load()
            self._parse()
            self._finalize()
            self.is_valid = True
            logger.info(f"Parsed ligand {os.path.basename(mol_path)}: {self.n_atoms} atoms, {self.n_bonds} bonds")
        except Exception as e:
            logger.error(f"Failed to parse ligand {mol_path}: {e}")

    def _load(self):
        ext = os.path.splitext(self.mol_path)[1]
        if ext.lower() == '.mol2':
            self.mol = Chem.MolFromMol2File(self.mol_path, sanitize=False, removeHs=False)
        else:
            self.mol = Chem.MolFromMolFile(self.mol_path, sanitize=False, removeHs=False)
        if self.mol is None:
            raise ValueError("RDKit failed to load molecule")
        self.mol = Chem.AddHs(self.mol)
        Chem.SanitizeMol(self.mol)
        if self.add_charges:
            AllChem.ComputeGasteigerCharges(self.mol)

    def _parse(self):
        conf = self.mol.GetConformer()
        self.elements = []
        self.coords = []
        self.charges = []
        self.hybrid = []
        self.aromatic = []
        for atom in self.mol.GetAtoms():
            idx = atom.GetIdx()
            self.elements.append(atom.GetAtomicNum())
            self.coords.append(conf.GetAtomPosition(idx))
            self.charges.append(float(atom.GetProp('_GasteigerCharge')) if self.add_charges else 0.0)
            self.hybrid.append(atom.GetHybridization())
            self.aromatic.append(atom.GetIsAromatic())

        # Bonds
        self.bond_index = []
        self.bond_type = []
        for bond in self.mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            order = bond.GetBondTypeAsDouble()
            self.bond_index.extend([[a, b], [b, a]])
            self.bond_type.extend([order, order])

    def _finalize(self):
        self.n_atoms = len(self.elements)
        self.n_bonds = len(self.bond_type) // 2
        # Convert to numpy arrays
        self.element = np.array(self.elements, dtype=np.int64)
        self.pos = np.array([[p.x, p.y, p.z] for p in self.coords], dtype=np.float32)
        self.charge = np.array(self.charges, dtype=np.float32)
        self.hybridization = np.array([int(h) for h in self.hybrid], dtype=np.int64)
        self.aromaticity = np.array(self.aromatic, dtype=np.int8)
        self.bond_index = np.array(self.bond_index, dtype=np.int64).T
        self.bond_type = np.array(self.bond_type, dtype=np.float32)
        # Additional descriptors
        self.ring_sizes = np.array([len(x) for x in Chem.GetSymmSSSR(self.mol)], dtype=np.int8)
        self.mol_wt = Descriptors.MolWt(self.mol)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'element': self.element,
            'pos': self.pos,
            'charge': self.charge,
            'hybridization': self.hybridization,
            'aromaticity': self.aromaticity,
            'bond_index': self.bond_index,
            'bond_type': self.bond_type,
            'mol_wt': float(self.mol_wt),
            'ring_sizes': self.ring_sizes
        }
