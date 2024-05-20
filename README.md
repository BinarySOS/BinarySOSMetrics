<p align="center">
<!-- <a href="https://badge.fury.io/py/yapf"><img alt="PyPI Version" src="https://badge.fury.io/py/yapf.svg"></a> -->
<a href="https://github.com/google/yapf/actions/workflows/pre-commit.yml"><img alt="Actions Status" src="https://github.com/google/yapf/actions/workflows/pre-commit.yml/badge.svg"></a>

</p>

# Binary Small Object Segmentation Metrics
## SOS!!!
What should I evaluate about the Binary Small Object Segmentation(SOS) algorithm? Is there a simple and easy-to-use toolkit?

## Types
| **Pixel Level** | **Target Level** | **Both** |
|:---------------:|:----------------:|:--------:|
|AUC_ROC_PRMetric |                  |BinaryCenterMetric|
|Precision_Recall_F1_IoUMetric|                  |PD_FAMetric|
|                 |                  |          |




## Installation
```bash
cd BinarySOSMetrics
```
For developers(recommended, easy for debugging)
```bash
pip install -e .
```
Only use
```bash
pip install sosmetrics
```
## Features

## Tutorial