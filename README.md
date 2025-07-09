# GiG
A Graph-in-Graph Learning Framework for Drug-Target Interaction Prediction

Graph-in-Graph learning framework is a GNN-based model that represents graphs of drug and target molecular structures as meta-nodes in a drug-target interaction graph, enabling a detailed exploration of their intricate relationships.

## Model Overview
<img width="1370" alt="Screenshot 2025-07-08 at 3 10 44 PM" src="https://github.com/user-attachments/assets/b81dc2ad-ce9c-4f8f-bbb7-c602145ca92f" />

## Installation
python == 3.9.0

pytorch = 2.2.0

pytorch-cuda == 11.8

torch-geometric == 2.5.3

biopython == 1.84

## Download and Prepare Data

You can obtain the full dataset in **JSON** format from our [Google Drive](https://drive.google.com/file/d/1DHN98GNzy_8OQ9_Z1r62m9BSo6HkVETr/view?usp=sharing), or simply use the three plain-text files included in the `data/` folder:

* **`drug_smiles.txt`**
  Two-column TSV:

  1. Drug name
  2. SMILES string

* **`protein_sequences.txt`**
  Two-column TSV:

  1. Protein (target) name
  2. Amino-acid sequence

* **`mat_drug_protein.txt`**
  A binary interaction matrix (drugs × proteins):

  * `0` = no interaction
  * `1` = interaction

### Preparing Graph Inputs

1. **Drug graphs**
   Use [RDKit](https://www.rdkit.org/) to parse each SMILES string and generate its corresponding molecular graph.

2. **Protein graphs**

   1. Install and run [PconsC4](https://github.com/ElofssonLab/PconsC4) to predict residue–residue contact maps from each protein sequence.
   2. Convert each contact map into a graph representation (nodes = residues; edges = predicted contacts).

Once both drug and protein graphs are prepared, you can load them—together with the interaction matrix—to construct your drug–target interaction dataset.


