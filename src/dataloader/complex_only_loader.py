from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist



def build_complex_only_loader(
    dataset,
    batch_size,
    loader_num_workers,
    collate_fn_conformation_data,
    shuffle,
    drop_last,
    mode=None,
):
    # Use DistributedSampler in DDP.
    use_ddp = dist.is_available() and dist.is_initialized()
    sampler = None
    if use_ddp:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_conformation_data,
        num_workers=loader_num_workers,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=loader_num_workers > 0,
    )
    return loader
