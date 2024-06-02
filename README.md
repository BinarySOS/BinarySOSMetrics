<p align="center">
<!-- <a href="https://badge.fury.io/py/yapf"><img alt="PyPI Version" src="https://badge.fury.io/py/yapf.svg"></a> -->
<a href="https://github.com/google/yapf/actions/workflows/pre-commit.yml"><img alt="Actions Status" src="https://github.com/google/yapf/actions/workflows/pre-commit.yml/badge.svg"></a>

</p>

# Binary Small Object Segmentation Metrics
## SOS!!!
What should I evaluate about the Binary Small Object Segmentation(SOS) algorithm? Is there a simple and easy-to-use toolkit?

## Overview of Metrics
| **Pixel Level** | **Target Level** | **Both** |
|:---------------:|:----------------:|:--------:|
|AUC_ROC_PR|                  |BinaryCenter|
|Precision_Recall_F1_IoU|                  |PD_FA|
|NormalizedIoU                 |           |HybridNormalizedIoU|


<div align="center">
  <b>Architectures</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Pixel Level</b>
      </td>
      <td>
        <b>Target Level</b>
      </td>
      <td>
        <b>Both</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="sosmetrics/metrics/auc_roc_pr_metric.py">AUC_ROC_PR</a></li>
            <li><a href="sosmetrics/metrics/pre_rec_f1_iou_metric.py">Precision Recall F1 IoU (DOI:10.1109/TAES.2023.3238703)</a></li>
            <li><a href="sosmetrics/metrics/normalized_iou_metric.py">NormalizedIoU (DOI:10.1109/WACV48630.2021.00099)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="sosmetrics/metrics/">Placeholder</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="sosmetrics/metrics/pd_fa_metric.py">Pd_Fa (DOI:10.1109/TIP.2022.3199107)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

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
For more details, please refer to `./notebook/tutorial.ipynb`
## Features

## Tutorial
