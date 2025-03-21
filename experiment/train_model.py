import lightning as L
from omegaconf import OmegaConf
from pathlib import Path


def train_model():
    """Setting up paths and configs"""
    PATH_ROOT = Path(__file__).parents[1]
    Experiment_dir = Path(__file__).parents[0]
    log_dir = Experiment_dir / Path("logs")
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir = Experiment_dir / Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    figures_dir = PATH_ROOT / Path("outputs/plots")
    figures_dir.mkdir(exist_ok=True)

    config = OmegaConf.load(PATH_ROOT / Path("config/config.yaml"))
    config["dirs"] = {
        "PATH_ROOT": PATH_ROOT,
        "Experiment_dir": Experiment_dir,
        "log_dir": log_dir,
        "checkpoint_dir": checkpoint_dir,
        "figures_dir": figures_dir,
    }

    L.seed_everything(config.model.seed)

    from spectralMPGNN.training.training import train

    train(config=config)
