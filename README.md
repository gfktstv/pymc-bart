# PyMC-BART: Decision Tables Fork

This repository is a fork of [PyMC-BART](https://github.com/pymc-devs/pymc-bart). It extends the PyMC probabilistic programming framework to include a modified implementation of BART (Bayesian Additive Regression Trees) using decision tables.
> Custom implementation is on branch bart_on_tables
## Table of Contents
- [PyMC-BART: Decision Tables Fork](#pymc-bart-decision-tables-fork)
  - [Table of Contents](#table-of-contents)
  - [About Implementation](#about-implementation)
      - [Performance](#performance)
      - [Statistics](#statistics)
      - [Future Outlook](#future-outlook)
      - [Some details on implementation](#some-details-on-implementation)
  - [Contributors](#contributors)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Original Citation](#original-citation)
  - [License](#license)
## About Implementation
This fork represents a simple attempt to implement **decision tables** within the BART framework.
#### Performance
*   **Speed:** This approach allows for an acceleration of the implementation.
*   **Classification:** It maintains the same quality as the original BART implementation on classification tasks.
*   **Regression:** Currently, this implementation yields lower quality results on regression tasks compared to the standard BART, even considering exaggeration of number of trees (m parameter)

#### Statistics

| Dataset        | Metric        | m=50             | m=100            | m=150            |
| :------------- | :------------ | :--------------- | :--------------- | :--------------- |
| **friedman**   | RMSE          | -41.8% .. -22.5% | -25.1% .. -14.1% | -15.5% .. -7.2%  |
| **friedman**   | Training Time | +19.1% .. +42.2% | +22.0% .. +23.8% | +24.1% .. +27.4% |
| **sin**        | RMSE          | -5.3% .. +20.6%  | -19.6% .. +16.6% | +7.8% .. +18.5%  |
| **sin**        | Training Time | +35.5% .. +39.4% | +33.2% .. +41.4% | +42.2% .. +45.9% |
| **california** | RMSE          | -10.8% .. -5.6%  | -7.3% .. -4.5%   | -6.9% .. -3.1%   |
| **california** | Training Time | -4.8% .. +0.0%   | -5.5% .. -4.5%   | -20.5% .. -15.9% |
| **diabetes**   | RMSE          | -1.9% .. +1.3%   | -1.1% .. +1.4%   | -0.9% .. +0.7%   |
| **diabetes**   | Training Time | +8.8% .. +21.6%  | +23.1% .. +25.7% | +26.2% .. +29.0% |
| **moons**      | F1-score      | +0.0% .. +0.0%   | +0.0% .. +0.0%   | +0.0% .. +0.0%   |
| **moons**      | Training Time | +24.5% .. +27.0% | +30.3% .. +31.9% | +30.7% .. +35.6% |
| **raisin**     | F1-score      | -0.3% .. +1.4%   | -0.4% .. +0.5%   | -0.8% .. +0.4%   |
| **raisin**     | Training Time | +17.0% .. +25.2% | +28.1% .. +29.8% | +30.8% .. +33.4% |
| **cancer**     | F1-score      | -0.5% .. +0.9%   | -0.9% .. +0.5%   | -0.9% .. +0.5%   |
| **cancer**     | Training Time | +23.4% .. +29.3% | +29.3% .. +29.9% | +25.4% .. +33.7% |
| **csgo**       | F1-score      | -4.1% .. +7.3%   | -3.3% .. +4.0%   | -5.0% .. +4.2%   |
| **csgo**       | Training Time | +16.5% .. +26.8% | +21.8% .. +26.0% | +23.4% .. +28.9% |
* Positive percentages (+) denote a relative performance improvement of the BARTOnTables model compared to the baseline BART, whereas negative percentages (-) indicate a performance degradation.
* Training time refers to the computational duration required for the forest sampling process. All measurements were conducted on an Apple M3 architecture using the standard time library.
* Datasets description:
	- Friedman: A standard regression benchmark (Friedman #1) involving non-linear interactions between input variables, generated to evaluate the model's capability to capture complex dependencies.
	- Sin: A synthetic regression task modeling a sine wave function with added noise.
	- California (California Housing): A regression dataset (sklearn).
    - Diabetes: A regression dataset (sklearn).
    - Moons (Make Moons): A synthetic binary classification dataset (sklearn).
    - Raisin: A binary classification dataset (sourced from the UCI Machine Learning Repository).
    - Cancer (Breast Cancer Wisconsin Diagnostic): A classic binary classification dataset (sklearn).
    - CS:GO (CS:GO Round Winner): A binary classification dataset sourced from Kaggle.
- All tests were conducted using `benchmark.py` script on bart_on_tables_test branch

#### Future Outlook
It is worth noting that parallelizing the trees on a GPU could yield significantly higher speedups. Theoretically, by utilizing GPU parallelization, one could use more than **2x the number of trees** compared to standard BART. This might allow the model to achieve comparable regression quality to the original implementation while still remaining faster overall.

#### Some details on implementation
All changes from original pymc-bart repo are listed below:
* bart.py — adds a new distribution to connect with the PGBARTOnTables sampler.
- pgbart.py — adds the new class PGBARTOnTables (sampling differs only by using the Table class instead of Tree), a new method in the ParticleTree class (sample_level, which calls the custom function grow_table), and a new function grow_table (utilizing vectorized operations to work with the Table data structure).
- tree.py — adds the new class Table.

Acceleration is achieved by using NumPy arrays instead of trees to access leaves. Furthermore, the operations of (1) getting data points in each leaf and (2) predicting the leaf value are now vectorized. In a nutshell, the differences can be explained as follows:
1. In the **Tree** data structure, to split a node, we have to copy data points in this node and create new Nodes with divided data points. In the Table data structure, we simply skip this step. To understand which data point belongs to which leaf, we calculate a bit sequence representing the path to the leaf on each split.
2. In the Tree data structure, to predict the output of a tree for X, we have to traverse the whole tree. In the Table data structure, we simply go through levels: on each level, we get 0 or 1 depending on the predictor and split value; eventually, we get a bit sequence representing the path to the corresponding leaf.
## Contributors
This project was developed within the framework of the **Yandex Education Student Camp** under the supervision of **[Maksim Nikolaev](https://github.com/ponelprinel)** (Saint Petersburg State University).

**Team Members:**
- [Grigorii Feoktistov](https://github.com/gfktstv)
- [Mikhail Solonko](https://github.com/DlMarsh)
- [Artur Filatov](https://github.com/arfiev)
- [Dmitry Sidimekov](https://github.com/sidimekov)
- [Sergey Voitov](https://github.com/TastyButSadly)
## Installation

To install this specific fork, you can install directly from GitHub:

```bash
pip install git+https://github.com/gfktstv/pymc-bart.git@bart_on_tables
```

## Usage

Usage remains similar to the standard PyMC-BART workflow:

```python
import pymc as pm
import pymc_bart as pmb

X, y = ... # Your data replaces "..."
with pm.Model() as model:
    bart = pmb.BART('bart', X, y)
    ...
    idata = pm.sample()
```

To use BART on tables implementation, simply change model:

```python
import pymc as pm
import pymc_bart as pmb

X, y = ... # Your data replaces "..."
with pm.Model() as model:
    bart = pmb.BARTOnTables('bart', X, y)
    ...
    idata = pm.sample()
```
## Original Citation

This work is based on PyMC-BART. If you use this fork, please cite the original paper:

```
@misc{quiroga2023bayesian,
title={Bayesian additive regression trees for probabilistic programming},
author={Quiroga, Miriana and Garay, Pablo G and Alonso, Juan M. and Loyola, Juan Martin and Martin, Osvaldo A},
year={2023},
doi={10.48550/ARXIV.2206.03619},
archivePrefix={arXiv},
primaryClass={stat.CO}
}
```

## License

[Apache License, Version 2.0](https://github.com/pymc-devs/pymc-bart/blob/main/LICENSE)
