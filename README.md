# SAMWOOD
### Segmentation and Quantitative Analysis of Wood Anatomical Sections (Modern & Fossil) Using SAM2

SamWood is an open-source Python package designed to automate the segmentation and measurement of wood cells in transverse sections of both modern and fossil samples.
It uses SAM2 (Segment Anything Model 2) from Ravi et al.2024 (https://arxiv.org/abs/2408.00714) , a state-of-the-art Vision Transformer for zero-shot segmentation, and a dedicated algorithms to reconstruct cell files and extract anatomical traits along growth gradients.

This repository accompanies the scientific article:
## SAMWOOD: An automated method to measure wood cells along growth orientation

## Installation

To install, you can use the following command

```
pip install .......
```


## Segment cells on an image

```
python src/segment_fossil.py
```

## Create connection graph and identify cell lines

```
python src/segment_fossil.py
```

## Export results to .csv

```
python src/segment_fossil.py
```


