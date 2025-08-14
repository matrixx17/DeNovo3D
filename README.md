This started in 2023 as my senior year project while I was an high school intern at the PA Biotechnology Center. I aimed to create a generative GNN-based deep learning model to create new potential drug molecules for user-inputted proteins. It is still a work in progress. So far, the completed functionality includes:
* A robust parsing system to extract structural and biochemical features of protein and ligand data files (.pdb and .sdf/.mol2 file types, respectively)
* A graph creator which takes the extracted coordinates and additional features, and turns them into structure preserving graphs (see below: example protein in red, ligand in blue). Uses PyTorch Geometric framework.
  <img width="1208" height="800" alt="image" src="https://github.com/user-attachments/assets/9b5319d4-7d54-4853-a971-21008a7aaa87" />
  <img width="1215" height="809" alt="image" src="https://github.com/user-attachments/assets/1067fd1e-83a4-406d-8756-2e756806385c" />


* A custom PyTorch dataset consisting of experimentally validated protein/ligand pairs as tuples

The model itself is still currently under development and I'm working out some bugs. 
