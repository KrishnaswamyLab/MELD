# Meld-convex

Use Tikhonov regularization to regress across the manifold.

## Installation

```
pip install --user git+git://github.com/KrishnaswamyLab/MELD.git#subdirectory=python
```

## Requirements
1. pygsp
	Install via
	`pip install pygsp`
2. Graphtools
	https://github.com/KrishnaswamyLab/graphtools
	Used for importing data, building graphs, and getting gradient matrices.

### Not required

1. pyunlocbox
	https://pyunlocbox.readthedocs.io/en/stable/
	Used for fast solving via proximal splitting
	Install via
	`pip install pyunlocbox`

## Usage

```
import meld
import graphtools
G = graphtools.Graph(data, use_pygsp=True)
meld_score = meld.meld(label, gamma=0.5, g=G)
```