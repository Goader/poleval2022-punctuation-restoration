from omegaconf import DictConfig, OmegaConf
import hydra

import logging

logger = logging.getLogger(__name__)


def train():
    pass


def evaluate():
    pass


def inference():
    pass


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
