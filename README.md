# VPR-Pruning

Accompanying code for Part III project: "Structured Pruning of Convolutional Neural Networks in the Context of Visual Place Recognition"


## Installation

Note: This environment will only work on linux devices at the moment, due to some of the packages not being available for other operating systems. e.g faiss-gpu

```bash
conda env create -f environment.yml

git clone https://github.com/VainF/Torch-Pruning.git
pip install -e Torch-Pruning/.

git clone https://github.com/frgfm/torch-scan.git
pip install -e torch-scan/.
```

## Usage
```python main.py <insert arguments here>
```


