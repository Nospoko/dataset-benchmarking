import random

import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

from benchmarking.timer import timer
from data_masked.masking import AwesomeMasks
from data_masked.tokenizer import MidiEncoder
from data_masked.quantizer import MidiQuantizer
from data_masked.augmentation import pitch_shift, change_speed


class MidiDataset(Dataset):
    def __init__(
        self,
        dataset: HFDataset,
        quantizer: MidiQuantizer,
        tokenizer: MidiEncoder,
        pitch_shift_probability: float = 0.0,
        time_stretch_probability: float = 0.0,
        masking_probability: float = 0.15,
    ):
        super().__init__()

        self.dataset = dataset.with_format("numpy")
        self.quantizer = quantizer
        self.tokenizer = tokenizer
        self.masks = AwesomeMasks()

        self.pitch_shift_probability = pitch_shift_probability
        self.time_stretch_probability = time_stretch_probability
        self.masking_probability = masking_probability

    def __len__(self):
        return len(self.dataset)

    @timer
    def apply_augmentation(self, record: dict):
        # shift pitch augmentation
        if random.random() < self.pitch_shift_probability:
            shift = 7
            record["pitch"] = pitch_shift(pitch=record["pitch"], shift_threshold=shift)

        # change tempo augmentation
        if random.random() < self.time_stretch_probability:
            record["dstart"], record["duration"] = change_speed(dstart=record["dstart"], duration=record["duration"])

        return record

    @timer
    def apply_masking(self, token_ids: np.ndarray, record: dict):
        input_token_ids = token_ids.copy()
        tgt_token_ids = token_ids.copy()

        # masking, adds new key to record dict called masked
        masked, mask_type = self.masks.apply(record, p=self.masking_probability)

        # source token ids
        mask_idx = self.tokenizer.token_to_id["<mask>"]
        input_token_ids[masked] = mask_idx

        # tgt token ids, -100 means loss is not calculated on this token
        tgt_token_ids[~masked] = -100

        # add mask type token
        mask_type_idx = self.tokenizer.token_to_id[mask_type]
        input_token_ids = np.insert(input_token_ids, obj=0, values=mask_type_idx)
        tgt_token_ids = np.insert(tgt_token_ids, obj=0, values=-100)

        return input_token_ids, tgt_token_ids

    @timer
    def add_cls_token(self, input_token_ids: np.ndarray, tgt_token_ids: np.ndarray):
        cls_token = self.tokenizer.token_to_id["<cls>"]
        input_token_ids = np.insert(input_token_ids, obj=0, values=cls_token)
        tgt_token_ids = np.insert(tgt_token_ids, obj=0, values=-100)

        return input_token_ids, tgt_token_ids

    @timer
    def _quantize_and_encode(self, record: dict):
        record = self.quantizer.quantize_record(record)
        token_ids = self.tokenizer.encode(record)
        return record, token_ids

    @timer
    def _fetch_record(self, index: int) -> dict:
        record = self.dataset[index]
        filename = record["midi_filename"]
        return record, filename

    def _replace_nan(self, record: dict):
        if np.any(np.isnan(record["dstart"])):
            record["dstart"] = np.nan_to_num(record["dstart"], copy=False)
        return record

    def __getitem__(self, index: int) -> dict:
        record, filename = self._fetch_record(index)

        # sanity check, replace NaN with 0
        record = self._replace_nan(record)

        record = self.apply_augmentation(record)

        record, token_ids = self._quantize_and_encode(record)
        input_token_ids, tgt_token_ids = self.apply_masking(token_ids, record)
        input_token_ids, tgt_token_ids = self.add_cls_token(input_token_ids, tgt_token_ids)

        tokens = {
            "filename": filename,
            "source_token_ids": torch.tensor(token_ids, dtype=torch.long),
            "input_token_ids": torch.tensor(input_token_ids, dtype=torch.long),
            "tgt_token_ids": torch.tensor(tgt_token_ids, dtype=torch.long),
        }

        return tokens


if __name__ == "__main__":
    from omegaconf import DictConfig
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    from data_masked.tokenizer import QuantizedMidiEncoder

    quantization_cfg = DictConfig(
        {
            "dstart": 7,
            "duration": 7,
            "velocity": 7,
        }
    )

    ds = load_dataset("JasiekKaczmarczyk/maestro-v1-sustain-masked", split="train")

    quantizer = MidiQuantizer(7, 7, 7)
    tokenizer = QuantizedMidiEncoder(7, 7, 7)

    dataset = MidiDataset(ds, quantizer, tokenizer, pitch_shift_probability=0.1, time_stretch_probability=0.1)

    loader = DataLoader(dataset, batch_size=4)

    x = next(iter(loader))
    print(x["input_token_ids"].shape)
    print(x["input_token_ids"])
