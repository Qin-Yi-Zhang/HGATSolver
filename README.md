# HGATSolver: A Heterogeneous Graph Attention Solver for Fluid–Structure Interaction (AAAI-2026 Oral)
[![HGATSolver](https://img.shields.io/badge/AAAI--2026-oral-blue)](https://aaai.org/conference/aaai/aaai-26/)

Implementation of the HGATSolver for Fluid–Structure Interaction (FSI).

## Dataset
Download the required datasets from Zenodo:

- FI-Valve.hdf5
- SI-Vessel.hdf5

Link: https://doi.org/10.5281/zenodo.17602345

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```
## Citation

```bibtex
@article{zhang2026hgatsolver,
  title     = {HGATSolver: A Heterogeneous Graph Attention Solver for Fluid--Structure Interaction},
  author    = {Zhang, Qin-Yi and Wang, Hong and Liu, Siyao and Lin, Haichuan and Cao, Linying and Zhou, Xiao-Hu and Chen, Chen and Wang, Shuangyi and Hou, Zeng-Guang},
  journal   = {arXiv preprint arXiv:2601.09251},
  year      = {2026},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi       = {10.48550/arXiv.2601.09251}
}
