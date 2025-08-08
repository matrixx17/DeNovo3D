import os 
import logging
from typing import Dict, Any

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
from Bio.PDB import PDBParser, DSSP
from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalFeatures, rdMolDescriptors, Descriptors
from rdkit.Chem.rdchem import BondType

class Protein:

    AA_NAME_SYM = { 'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F',
                    'GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L',
                    'MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R',
                    'SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y' }
    AA_TO_NUMBER = {v:i for i,v in enumerate(AA_NAME_SYM.values())}
    KYTE_DOOLITTLE = { 'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,
                       'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,
                       'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3 }
    
    def __init__(self, path: str, threshold: float = 8.0):
        self.path = path
        self.threshold = threshold
        self.parser = PDBParser(QUIET=True)
        self.is_valid = False

        self.residues = []
        self.CA_pos = []
        self.hydro_scores = []

        self.edges = None
        self.edge_dists = None

        try:
            self.parse()
            self.compute_edges()
            self.finalize()
            self.is_valid = True

    def parse(self):
        struct = self.parser.get_structure('prot', self.path)
        model = next(struct.get_models())
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
                self.CA_pos.append(coord)
                self.hydro_scores.append(hydro)

        if not self.residues: 
            raise ValueError("No valid residues extracted")
    
    def compute_edges(self):
        coords = np.vstack(self.CA_pos)
        tree = KDTree(coords)
        pairs = tree.query 

