import torch
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler


class EpochBaseLR(_LRScheduler):
    def __init__(self, optimizer, milestones, lrs, last_epoch=-1, ):
        if len(milestones) + 1 != len(lrs):
            raise ValueError('The length of milestones must equal to the '
                             ' length of lr + 1. Got {} and {} separately', len(milestones) + 1, len(lrs))
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)

        self.milestones = milestones
        self.lrs = lrs
        super(EpochBaseLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.lrs[bisect_right(self.milestones, self.last_epoch)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr = self.get_lr()

        for g in self.optimizer.param_groups:
            g['lr'] = lr


class ParamsControl(object):
    def __init__(self, opts, model):
        self.model = model
        self.opts = opts

        params_backbone = list(
            self.model.encoder.parameters()
        )

        params_new = list(
            self.model.new.parameters()
        )

        params_alignment = list(
            self.model.alignment.parameters()
        )

        params_patch = list(
            self.model.patch_proposal.parameters()
        )

        params_attention = list(
            self.model.self_attention.parameters()
        )

        params_temporal = list(
            self.model.temporal_head.parameters()
        )

        parameter_groups = [
            params_backbone,
            params_new,
            params_alignment,
            params_patch,
            params_attention,
            params_temporal,
        ]

        if opts.use_Adam:
            self.optimizers = [
                torch.optim.Adam(
                    group,
                    lr=0.0,
                    weight_decay=opts.weight_decay,
                )
                for group in parameter_groups
            ]
        else:
            self.optimizers = [
                torch.optim.SGD(
                    group,
                    lr=0.0,
                    momentum=opts.momentum,
                    weight_decay=opts.weight_decay,
                )
                for group in parameter_groups
            ]

        lr_backbone = self._parse_values(
            opts.lr_backbone
        )
        lr_new = self._parse_values(
            opts.lr_new
        )
        lr_alignment = self._parse_values(
            opts.lr_alignment
        )
        lr_patch = self._parse_values(
            opts.lr_patch
        )
        lr_attention = self._parse_values(
            opts.lr_attention
        )
        lr_temporal = self._parse_values(
            opts.lr_temporal
        )

        decay_backbone = self._parse_values(
            opts.decay_backbone
        )
        decay_new = self._parse_values(
            opts.decay_new
        )
        decay_alignment = self._parse_values(
            opts.decay_alignment
        )
        decay_patch = self._parse_values(
            opts.decay_patch
        )
        decay_attention = self._parse_values(
            opts.decay_attention
        )
        decay_temporal = self._parse_values(
            opts.decay_temporal
        )

        self.schedulers = [
            EpochBaseLR(
                self.optimizers[0],
                milestones=decay_backbone,
                lrs=lr_backbone,
            ),
            EpochBaseLR(
                self.optimizers[1],
                milestones=decay_new,
                lrs=lr_new,
            ),
            EpochBaseLR(
                self.optimizers[2],
                milestones=decay_alignment,
                lrs=lr_alignment,
            ),
            EpochBaseLR(
                self.optimizers[3],
                milestones=decay_patch,
                lrs=lr_patch,
            ),
            EpochBaseLR(
                self.optimizers[4],
                milestones=decay_attention,
                lrs=lr_attention,
            ),
            EpochBaseLR(
                self.optimizers[5],
                milestones=decay_temporal,
                lrs=lr_temporal,
            ),
        ]

    @staticmethod
    def _parse_values(value):
        return [
            float(item)
            for item in str(value).split(",")
            if str(item).strip()
        ]

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad(
                set_to_none=True
            )

    def back_grad(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def update(self, epoch):
        for scheduler in self.schedulers:
            scheduler.step(epoch)

    def get_lrs(self):
        names = [
            "backbone",
            "new",
            "alignment",
            "patch",
            "attention",
            "temporal",
        ]

        return {
            name: optimizer.param_groups[0]["lr"]
            for name, optimizer
            in zip(names, self.optimizers)
        }