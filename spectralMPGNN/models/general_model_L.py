import torch
import lightning as L


class general_model_L(L.LightningModule):
    def __init__(self, train_config, model):
        super(general_model_L, self).__init__()
        self.model = model

        self.lr = train_config.lr
        self.weight_decay = train_config.weight_decay
        self.scheduler_step = train_config.scheduler_step
        self.scheduler_gamma = train_config.scheduler_gamma

    def forward(self, spa_data, spc_data):
        return self.model(spa_data, spc_data)

    def loss_fn(self, gt_y, pred):
        error = torch.sum((gt_y - pred) ** 2, axis=1)
        loss = torch.sqrt(torch.mean(error))

        return loss

    def training_step(self, batch):
        spa_data, spc_data = batch
        output_pred_batch = self(spa_data, spc_data)
        # loss = self.loss_fn(spa_data.y, output_pred_batch)
        loss = self.model.loss(spa_data, output_pred_batch)
        self.log(
            "train_loss",
            loss,
            batch_size=len(spa_data),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch):
        spa_data, spc_data = batch
        output_pred_batch = self(spa_data, spc_data)
        # loss = self.loss_fn(spa_data.y, output_pred_batch)
        loss = self.model.loss(spa_data, output_pred_batch)
        self.log(
            "val_loss",
            loss,
            batch_size=len(spa_data),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch):
        spa_data, spc_data = batch
        output_pred_batch = self(spa_data, spc_data)
        loss = self.loss_fn(spa_data.y, output_pred_batch)
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def prediction_step(self, batch):
        spa_data, spc_data = batch
        output_pred_batch = self(spa_data, spc_data)
        return output_pred_batch

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
