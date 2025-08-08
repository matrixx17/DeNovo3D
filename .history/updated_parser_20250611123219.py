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

    AA_NAME_SYM = {}