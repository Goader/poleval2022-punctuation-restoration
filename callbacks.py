import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info


class MetricsLoggingCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        micro_recall = trainer.callback_metrics['rc_micro']
        micro_precision = trainer.callback_metrics['pr_micro']
        micro_f1 = trainer.callback_metrics['f1_micro']

        weighted_recall = trainer.callback_metrics['rc_micro']
        weighted_precision = trainer.callback_metrics['pr_micro']
        weighted_f1 = trainer.callback_metrics['f1_weighted']

        rank_zero_info(f'micro // precision: {100*micro_precision:.2f}, recall: {100*micro_recall:.2f}, f1: {100*micro_f1:.2f}')
        rank_zero_info(f'weighted // precision: {100*weighted_precision:.2f}, recall: {100*weighted_recall:.2f}, f1: {100*weighted_f1:.2f}')
