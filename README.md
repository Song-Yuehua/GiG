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

## Data Download
You can download the full dataset in **JSON** format from our [Google Drive](https://drive.google.com/file/d/1DHN98GNzy_8OQ9_Z1r62m9BSo6HkVETr/view?usp=sharing), or simply use the three files included in the `data/` folder:

* **`drug_smiles.txt`**
  Contains two columns:

  1. Drug name
  2. Corresponding SMILES string

* **`protein_sequences.txt`**
  Contains two columns:

  1. Protein (target) name
  2. Amino-acid sequence

* **`mat_drug_protein.txt`**
  A binary interaction matrix between drugs (rows) and proteins (columns):

  * `0` = no interaction
  * `1` = interaction

Just clone this repo and point your data loader at `data/` to get started!

