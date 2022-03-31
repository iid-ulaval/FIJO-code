from torch.utils.data import DataLoader


class DataloaderFactory:
    def __init__(self, num_workers, collate_fn):
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def create(self,
               train_set,
               valid_set,
               test_set,
               batch_size,
               drop_last=True):
        train_generator = DataLoader(train_set,
                                     batch_size=batch_size,
                                     drop_last=drop_last,
                                     collate_fn=self.collate_fn,
                                     num_workers=self.num_workers)

        valid_generator = DataLoader(valid_set,
                                     batch_size=batch_size,
                                     drop_last=drop_last,
                                     collate_fn=self.collate_fn,
                                     num_workers=self.num_workers)

        test_generator = DataLoader(test_set,
                                    batch_size=batch_size,
                                    drop_last=drop_last,
                                    collate_fn=self.collate_fn,
                                    num_workers=self.num_workers)

        return train_generator, valid_generator, test_generator
