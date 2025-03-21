# Spectral-MPGNN

This code explore the application of the spectral messaging passing graph neural network (Spectral-MPGNN) for modeling quasistatic problems. In quasistatic problems, information propagates at infite speed, i.e. all the points in the domain are aware of the boundary conditions instantly. In message passing graph neural network, information propagates at a finite speed constrained by the number of message passing layers. To overcome this problem, we employ an architecture ([Stachenfeld et al. (2020)](https://arxiv.org/abs/2101.00079)) that combines spectral graphs (complete graphs with nodes representing a few eigenmodes of the graph Laplacian) with the spatial graphs constructed by discretizing the domain. Message passing on spatial graphs allows learning local features (governing equations) while spectral graph learn global features. Eigenpooling and eigenbroadcasting operations facilitate exchange of information between the spatial and spectral networks.

## Getting started

clone the repository:

```bash
git clone git@github.com:m-manav/Spectral-MPGNN.git
cd Spectral-MPGNN
```

Install uv. Then create a virtual environment using the lock file by running the following command in a terminal:

```bash
uv sync
```

Activate the virtual environenment by running the following command in a terminal:

```bash
uv venv
```

To add an additional package to the project, use the add command. For example:

```bash
uv add numpy
```
