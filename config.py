"""
Central configuration for model and dataset vocabularies.
"""

from typing import Dict, List

# Fixed atom element vocabulary (atomic numbers) common in drug-like molecules
# Order matters; index in this list is used as class id. The last class is UNK.
ATOM_ELEMENTS: List[int] = [
    1,   # H
    6,   # C
    7,   # N
    8,   # O
    9,   # F
    15,  # P
    16,  # S
    17,  # Cl
    35,  # Br
    53,  # I
]

# Index mapping for elements; UNK goes to the last index
ATOM_ELEMENT_TO_INDEX: Dict[int, int] = {z: i for i, z in enumerate(ATOM_ELEMENTS)}
ATOM_CLASS_UNK: int = len(ATOM_ELEMENTS)  # unknown element class id
NUM_ATOM_CLASSES: int = len(ATOM_ELEMENTS) + 1

# Bond type classes (including no bond)
# We map RDKit bond orders (including aromatic ~1.5) to class ids
BOND_ORDER_TO_CLASS: Dict[float, int] = {
    0.0: 0,   # no bond (for dense adjacency labels)
    1.0: 1,   # single
    1.5: 4,   # aromatic
    2.0: 2,   # double
    3.0: 3,   # triple
}
NUM_BOND_CLASSES: int = 5  # 0..4

# Data and training defaults
DEFAULT_MAX_LIG_NODES: int = 64
DEFAULT_HIDDEN_DIM: int = 128
DEFAULT_LR: float = 1e-3
DEFAULT_EPOCHS: int = 20
DEFAULT_BATCH_SIZE: int = 8


