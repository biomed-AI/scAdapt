# scAdapt: Virtual adversarial domain adaptation network for single cell RNA-seq data classification across platforms and species
This repository contains the Python implementation for scAdapt. scAdapt is a virtual adversarial domain adaptation network to transfer single cell labels between datasets with batch effects. scAdapt used both the labeled source and unlabeled target data to train an enhanced classifier, and aligned the labeled source centroid and pseudo-labeled target centroid to generate a joint embedding.

<p align="center">
    <img src="scAdapt/model.png" width="638">
</p>

# Run the demo
python scAdapt/predict_example.py

# Tutorials
Coming soon...
