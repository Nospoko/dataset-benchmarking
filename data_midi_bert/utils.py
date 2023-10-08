import torch


class InfiniteDataLoader(torch.utils.data.DataLoader):
    """Super lazy copy-paste from https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()
        self.epoch_counter = 0

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.dataset.tokens_per_record

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            self.epoch_counter += 1
            if hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(self.epoch_counter)
            batch = next(self.dataset_iterator)
        return batch
