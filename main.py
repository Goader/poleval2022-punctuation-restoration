import logging

import hydra
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from datamodule import PunctuationRestorationDataModule
from model import RestorationModel
from callbacks import MetricsLoggingCallback


logger = logging.getLogger(__name__)


def train(cfg: DictConfig):
    datamodule = PunctuationRestorationDataModule(cfg)
    datamodule.setup()
    model = RestorationModel(cfg, num_classes=datamodule.num_classes)

#    trainer = pl.Trainer(accelerator=cfg.trainer.accelerator, devices=1, max_epochs=cfg.trainer.max_epochs)

#    lr_finder = trainer.tuner.lr_find(model)

    # Results can be found in
#    lr_finder.results

    # Plot with
#    fig = lr_finder.plot(suggest=True)
#    fig.show()

    # Pick point based on plot, or get suggestion
#    new_lr = lr_finder.suggestion()
#    print(new_lr)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        monitor="f1_weighted",
        mode="max",
        dirpath="output/",
        filename="punctuation-restoration-{epoch:03d}-{val_f1:.2f}",
    )
    wandb_logger = WandbLogger(project="punctuation-restoration")
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        logger=wandb_logger,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        # default_root_dir='output',
        callbacks=[MetricsLoggingCallback(), checkpoint_callback],
        # fast_dev_run=True,
        # overfit_batches=1,
    )
    trainer.fit(model, datamodule=datamodule)


def evaluate(cfg: DictConfig):
    datamodule = PunctuationRestorationDataModule(cfg)
    datamodule.setup('test')
    model = RestorationModel(cfg, num_classes=datamodule.num_classes)


def inference(cfg: DictConfig):
    datamodule = PunctuationRestorationDataModule(cfg)
    datamodule.setup('predict')
    model = RestorationModel(cfg, num_classes=datamodule.num_classes)


@hydra.main(config_path='configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if cfg.task == 'train':
        train(cfg)

    elif cfg.task == 'evaluate':
        evaluate(cfg)

    elif cfg.task == 'inference':
        inference(cfg)

    else:
        raise ValueError('unknown task, can be either `train` or `evaluate`')


if __name__ == '__main__':
    main()
