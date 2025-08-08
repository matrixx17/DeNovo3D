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