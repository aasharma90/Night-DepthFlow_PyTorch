import torch.utils.data
from dataloader.unaligned_dataset import UnalignedDataset


class DataLoader():
    def name(self):
        return 'DataLoader'

    # Modified the function and added shuffle=True - AA, 15/10/18, 8:00pm
    def __init__(self, opt, dataname='Oxford', shuffle_data=True, val_set=False):
        self.opt = opt
        self.dataset = UnalignedDataset(opt, dataname, val_set)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            num_workers=int(opt.nThreads),
            shuffle=shuffle_data,
            drop_last=True)

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data

