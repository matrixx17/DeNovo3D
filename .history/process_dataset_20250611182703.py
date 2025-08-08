from pdbbind_parser import Protein, Ligand
from graph_constructor import build_protein_graph, build_ligand_graph, build_complex_graph

dataset = []
for pdb_id in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, pdb_id)
    if not os.path.isdir(folder): continue

    p = os.path.join(folder, f"{pdb_id}_pocket.pdb")
    l = os.path.join(folder, f"{pdb_id}_ligand.mol2")
    if not (os.path.exists(p) and os.path.exists(l)):
        print(f"Skipping {pdb_id}, missing files.")
        continue

    prot = Protein(p); lig = Ligand(l)
    if not (prot.is_valid and lig.is_valid):
        print(f"Skipping {pdb_id}, parse failed.")
        continue

    pdict = prot.to_dict()
    ldict = lig.to_dict()
    prot_graph = build_protein_graph(pdict)
    lig_graph  = build_ligand_graph(ldict)

    # Option A: keep them separate
    dataset.append((prot_graph, lig_graph))

    # Option B: merge into one complex graph instead
    # complex_graph = build_complex_graph(prot_graph, lig_graph)
    # dataset.append(complex_graph)

torch.save(dataset, output_path)