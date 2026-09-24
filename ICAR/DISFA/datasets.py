"""R8 reproducibility wrapper for DISFA.

This module intentionally reuses the original DISFADataset augmentation and split logic.
It only adds two diagnostic fields (global_idx, local_idx) and optionally gives the
DataLoader its own torch.Generator so shuffle order can be isolated from model RNG.
"""
import torch
from dataset_disfa import DISFADataset, AUBalanceSampler


class IndexedDISFADataset(DISFADataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        global_idx = int(self.original_indices[index])
        local_idx = int(index)
        return (*sample, global_idx, local_idx)


def get_data_loader(opts, mode='train', split='CCNN'):
    dataset = IndexedDISFADataset(opts, mode, split)

    if mode == 'train':
        batch_size = opts.batch_size_train
        sampler = None
        shuffle = True
    else:
        batch_size = opts.batch_size_valid
        sampler = None
        shuffle = False

    if mode == 'train' and opts.use_sampler:
        # Kept for API compatibility, but R8 reproducibility diagnosis is intended
        # for the natural loader used by the controlled experiment (use_sampler=0).
        sampler = AUBalanceSampler(dataset, opts.use_sampler)
        shuffle = False

    loader_rng = getattr(opts, 'loader_rng', 'global')
    generator = None
    if loader_rng == 'isolated':
        generator = torch.Generator()
        base_seed = int(getattr(opts, 'data_seed', getattr(opts, 'seed', 42)))
        # Validation does not shuffle, but give it a distinct state anyway.
        generator.manual_seed(base_seed if mode == 'train' else base_seed + 1000003)
    elif loader_rng != 'global':
        raise ValueError('loader_rng must be "global" or "isolated"')

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
    if generator is not None:
        kwargs['generator'] = generator

    if opts.snapshot == 'debug':
        kwargs.pop('num_workers', None)
        kwargs.pop('pin_memory', None)

    data_loader = torch.utils.data.DataLoader(**kwargs)
    return data_loader, len(dataset)
