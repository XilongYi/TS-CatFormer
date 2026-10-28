from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from lightning.pytorch.utilities.combined_loader import CombinedLoader



def _maybe_ddp_sampler(ds, shuffle: bool, drop_last: bool):
    use_ddp = dist.is_available() and dist.is_initialized()
    if not use_ddp:
        return None
    # drop_last is applied per shard in DistributedSampler.
    return DistributedSampler(ds, shuffle=shuffle, drop_last=drop_last)


def build_complex_pocket_seq_loader(dataset, batch_size, loader_num_workers, shuffle,
                         drop_last, collate_fn_conformation_data, collate_fn_no_struct,
                           mode="max_size_cycle"):
    dsA = dataset.dsA
    dsB = dataset.dsB

    samplerA = _maybe_ddp_sampler(dsA, shuffle=shuffle, drop_last=drop_last)
    samplerB = _maybe_ddp_sampler(dsB, shuffle=shuffle, drop_last=drop_last)

    loaderA = DataLoader(
        dsA,
        batch_size=batch_size,
        num_workers=loader_num_workers,
        shuffle=(shuffle if samplerA is None else False),
        sampler=samplerA,
        drop_last=drop_last,
        collate_fn=collate_fn_conformation_data,
        pin_memory=True,
        persistent_workers=loader_num_workers > 0,
    )

    loaderB = DataLoader(
        dsB,
        batch_size=batch_size,
        num_workers=loader_num_workers,
        shuffle=(shuffle if samplerB is None else False),
        sampler=samplerB,
        drop_last=drop_last,
        collate_fn=collate_fn_no_struct,
        pin_memory=True,
        persistent_workers=loader_num_workers > 0,
    )

    return CombinedLoader({"A": loaderA, "B": loaderB}, mode=mode)


