import lightning as L
import pandas as pd
import torch
import torchmetrics
from transformers import (
    AdamW,
    BertForSequenceClassification,
    PretrainedConfig,
    get_linear_schedule_with_warmup,
)


class Bert(L.LightningModule):
    def __init__(
        self,
        num_classes=None,
        training_steps=None,
        from_checkpoint=False,
        model_config_json_filepath=None,
    ):
        super().__init__()
        if from_checkpoint:
            model_config = PretrainedConfig.from_json_file(model_config_json_filepath)
            self._model = BertForSequenceClassification(config=model_config)
        else:
            self._model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
                num_labels=num_classes,  # The number of output labels--2 for binary classification.
                # You can increase this for multi-class tasks.
                output_attentions=False,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
            )
        self.total_training_steps = training_steps
        self.f1 = torchmetrics.F1Score(task="binary")
        self.confmat = torchmetrics.ConfusionMatrix(task="binary", num_classes=2)

        # TODO remove
        self.preds = []
        self.labels = []

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=False,
    ):
        outputs = self._model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
        )
        return outputs

    def _common_step(self, batch, batch_idx, prefix):
        input_ids = batch[0]
        input_mask = batch[1]
        labels = batch[2]
        result = self(
            input_ids,
            token_type_ids=None,
            attention_mask=input_mask,
            labels=labels,
            return_dict=True,
        )
        loss = result.loss
        self.log(f"{prefix}_loss", loss.item(), prog_bar=True)

        logits = result.logits
        y_hat = torch.argmax(logits, dim=1)
        self.f1.update(y_hat, labels)
        self.confmat.update(y_hat, labels)

        return loss, logits, y_hat, labels

    def training_step(self, batch, batch_idx):
        loss, logits, y_hat, y = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y_hat, y = self._common_step(batch, batch_idx, "val")

        # TODO remove
        self.preds.extend(y_hat.cpu().numpy())
        self.labels.extend(y.cpu().numpy())

    def test_step(self, batch, batch_idx):
        loss, logits, y_hat, y = self._common_step(batch, batch_idx, "test")

    def _common_on_epoch_end(self, prefix):
        f1_score = self.f1.compute()
        self.log(f"{prefix}_f1", f1_score, prog_bar=True)
        self.f1.reset()

        confmat = self.confmat.compute()
        self.log(f"{prefix}_TN", confmat[0, 0], prog_bar=True)
        self.log(f"{prefix}_FP", confmat[0, 1], prog_bar=True)
        self.log(f"{prefix}_FN", confmat[1, 0], prog_bar=True)
        self.log(f"{prefix}_TP", confmat[1, 1], prog_bar=True)
        self.confmat.reset()

    def on_train_epoch_end(self):
        self._common_on_epoch_end("train")

    def on_validation_epoch_end(self):
        self._common_on_epoch_end("val")

        # Optionally log other metrics or average loss
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # self.log('val_loss', avg_loss, prog_bar=True)

        # TODO remove
        epoch = self.current_epoch
        df = pd.DataFrame({"actual_label": self.labels, "predicted_label": self.preds})
        df.to_csv(f"validation_predictions_epoch_{epoch}.csv", index=False)
        self.preds = []
        self.labels = []

    def on_test_epoch_end(self):
        self._common_on_epoch_end("test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5, eps=1e-8)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=self.total_training_steps
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
