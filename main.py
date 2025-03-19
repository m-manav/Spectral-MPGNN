from experiment.train_model import train_model
import wandb

wandb.login()


def main():
    train_model()


if __name__ == "__main__":
    main()
