import os
import warnings
import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


def _is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _setup_ddp():
    if _is_dist():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _cleanup_ddp():
    if _is_dist():
        dist.destroy_process_group()

class _NullWriter:
    """empty writer"""
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop

@hydra.main(version_base=None, config_path="src/configs", config_name="train_clean")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    _setup_ddp()
    try:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        is_main_process = (rank == 0)

        # logger / writer
        project_config = OmegaConf.to_container(config)
        logger = setup_saving_and_logging(config)

        if not is_main_process:
            for h in logger.handlers:
                h.setLevel(50)  

        api_key = os.getenv("COMET_API_KEY")
        if is_main_process:
            writer = instantiate(config.writer, api_key, logger, project_config)
        else:
            writer = _NullWriter()
        

        # device
        if _is_dist():
            device = f"cuda:{local_rank}"
        else:
            if config.trainer.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = config.trainer.device

        # text encoder
        text_encoder = instantiate(config.text_encoder)

        # dataloaders + batch transforms
        dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

        # model
        model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)

        # loss
        loss_function = instantiate(config.loss_function).to(device)

        # metrics
        metrics = {"train": [], "inference": []}
        for metric_type in ["train", "inference"]:
            for metric_config in config.metrics.get(metric_type, []):
                metrics[metric_type].append(
                    instantiate(metric_config, text_encoder=text_encoder)
                )

        # optimizer & scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = instantiate(config.optimizer, params=trainable_params)
        lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer, steps_per_epoch=len(dataloaders['train']))

        # DDP wrap 
        if _is_dist():
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        # epoch_len
        epoch_len = config.trainer.get("epoch_len")

        trainer = Trainer(
            model=model,
            criterion=loss_function,
            metrics=metrics,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            text_encoder=text_encoder,
            config=config,
            device=device,
            dataloaders=dataloaders,
            epoch_len=epoch_len,
            logger=logger,
            writer=writer,               
            batch_transforms=batch_transforms,
            skip_oom=config.trainer.get("skip_oom", True),
        )

        trainer.train()

    finally:
        _cleanup_ddp()


if __name__ == "__main__":
    main()
