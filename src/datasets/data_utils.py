from itertools import repeat

import os
from hydra.utils import instantiate
from torch.utils.data import DistributedSampler

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, text_encoder, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.
    """
    # transforms / batch transforms
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    dataloaders = {}
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    use_ddp = world_size > 1

    print("DATASETS PARTITIONS:", list(config.datasets.keys()))
    for dataset_partition in config.datasets.keys():
        # dataset
        dataset = instantiate(
            config.datasets[dataset_partition],
            text_encoder=text_encoder,
            _convert_="all",
        )

        assert config.dataloader.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        # sampler
        sampler = None
        if use_ddp:
            sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=(dataset_partition == "train"),
                drop_last=(dataset_partition == "train"),
            )

        # dataloader
        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=False if sampler is not None else (dataset_partition == "train"),
            sampler=sampler,
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms