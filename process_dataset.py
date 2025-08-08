import os
import torch
from typing import List, Tuple

from updated_parser import Protein, Ligand
from graph_constructor import build_protein_graph, build_ligand_graph


def build_dataset(dataset_path: str) -> List[Tuple]:
    dataset = []
    for pdb_id in os.listdir(dataset_path):
        folder = os.path.join(dataset_path, pdb_id)
        if not os.path.isdir(folder):
            continue

        p = os.path.join(folder, f"{pdb_id}_pocket.pdb")
        l = os.path.join(folder, f"{pdb_id}_ligand.mol2")
        if not (os.path.exists(p) and os.path.exists(l)):
            print(f"Skipping {pdb_id}, missing files.")
            continue

        prot = Protein(p)
        lig = Ligand(l)
        if not (prot.is_valid and lig.is_valid):
            print(f"Skipping {pdb_id}, parse failed.")
            continue

        pdict = prot.to_dict()
        ldict = lig.to_dict()
        prot_graph = build_protein_graph(pdict)
        lig_graph = build_ligand_graph(ldict)
        dataset.append((prot_graph, lig_graph))
    return dataset


def build_example_dataset() -> List[Tuple]:
    """Fallback using bundled 6ugn example files."""
    protein_path = '/Users/vedantajain/ConiferPoint/DeNovo3D/6ugn_pocket.pdb'
    ligand_path = '/Users/vedantajain/ConiferPoint/DeNovo3D/6ugn_ligand.mol2'
    prot = Protein(protein_path)
    lig = Ligand(ligand_path)
    if not (prot.is_valid and lig.is_valid):
        raise RuntimeError('Bundled example structures failed to parse')
    prot_graph = build_protein_graph(prot.to_dict())
    lig_graph = build_ligand_graph(lig.to_dict())
    return [(prot_graph, lig_graph)]


if __name__ == '__main__':
    # Configure paths as needed. If DATASET_DIR is not set, save example dataset.
    dataset_dir = os.environ.get('DATASET_DIR', '').strip()
    output_path = os.environ.get('OUTPUT_PATH', 'dataset.pt')
    if dataset_dir and os.path.isdir(dataset_dir):
        ds = build_dataset(dataset_dir)
    else:
        print('DATASET_DIR not set or invalid. Building example dataset from bundled files...')
        ds = build_example_dataset()
    torch.save(ds, output_path)
    print(f'Saved dataset with {len(ds)} samples to {output_path}')