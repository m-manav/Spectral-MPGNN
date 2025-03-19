from .general_model_L import general_model_L
from .model import Spectral_MP_GNN


def construct_model(config):
    model = Spectral_MP_GNN(config.model)
    model_L = general_model_L(config.train, model)
    return model_L
