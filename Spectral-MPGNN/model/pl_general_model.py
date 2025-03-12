import torch
import pytorch_lightning as pl
import torch.nn.functional as F

import wandb

from InputNormalize import normalize


class pl_general_model(pl.LightningModule):
    def __init__(
        self,
    ):
        super(pl_general_model, self).__init__()
        self.model = None
        self.energy_compute = None
        self.field_calculation = None
        self.num_fields = 1

        self.lr
        self.weight_decay
        self.scheduler_step
        self.scheduler_gamma

    def forward(self, input_coord, mat_A_section, train_trunk, train_branch):
        return self.model(input_coord, mat_A_section, train_trunk, train_branch)

    def loss(self, pred, inputs, loss_mask, mean_vec_y, std_vec_y):
        # Normalize labels with dataset statistics
        labels = normalize(inputs.y, mean_vec_y, std_vec_y)

        # Find sum of square errors
        error = torch.sum((labels - pred) ** 2, axis=1)

        # Root and mean the errors for the nodes we calculate loss for
        if loss_mask == torch.tensor([1], device=pred.device):
            loss = torch.sqrt(torch.mean(error))
        else:
            loss = torch.sqrt(torch.mean(error[loss_mask]))

        return loss

    def training_step(self, batch, batch_idx):
        # if batch_idx == 0:
        #     self._curr_epoch += 1
        input_coord, mat_A_section, train_trunk, train_branch, output_batch = batch

        output_pred_batch = self(input_coord, mat_A_section, train_trunk, train_branch)
        loss = 1

        return loss

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()
