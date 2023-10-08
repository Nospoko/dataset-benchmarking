import random

import torch
import numpy as np
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset as TorchDataset

from benchmarking.timer import timer
from data_midi_bert.encoder import MidiEncoder


class RobertDataset(TorchDataset):
    def __init__(self, dataset: HFDataset, tokenizer: PreTrainedTokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __rich_repr__(self):
        yield "RobertDataset"
        yield "size", len(self)
        yield "tokenizer", self.tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        record = self.dataset[idx]
        tokens = [record["mask_type"]] + record["tokens"]
        encoded_tokens = torch.tensor(self.tokenizer.encode(tokens))
        input_ids = encoded_tokens.clone()
        labels = input_ids.clone()

        # Mask inputs
        input_ids[1:][torch.BoolTensor(record["mask"])] = self.tokenizer.mask_token_id

        # Mask targets
        labels[1:][~torch.BoolTensor(record["mask"])] = -100
        labels[0] = -100

        # Everything is attentioned
        attention_mask = torch.ones_like(input_ids)
        data = {
            "labels": labels,
            "input_ids": input_ids,
            "tokens": encoded_tokens,
            "attention_mask": attention_mask,
        }
        return data

    @property
    def tokens_per_record(self) -> int:
        use_config_here_pls = 60
        return use_config_here_pls


class MyMaskedMidiDataset(TorchDataset):
    def __init__(
        self,
        dataset: HFDataset,
        encoder: MidiEncoder,
        sequence_len: int,
        mask_probability_min: float = 0.1,
        mask_probability_max: float = 0.5,
    ):
        self.mask_probability_min = mask_probability_min
        self.mask_probability_max = mask_probability_max
        self.probability_range = mask_probability_max - mask_probability_min
        # Mask randomness is implemented with numpy right now, see __getitem__
        self.dataset = dataset.with_format("numpy")
        self.encoder = encoder
        self.sequence_len = sequence_len
        # self.sequence_len = self.dataset.features["pitch"].length

    def _masking_probability(self) -> float:
        p = self.mask_probability_min + np.random.random() * self.probability_range
        return p

    @timer
    def _fetch_record(self, idx: int):
        record = self.dataset[idx]
        tokens = self.encoder.record_to_tokens(record)[: self.sequence_len]
        mask_name_token, mask = random.choice(list(record["masking_space"].items()))
        mask = mask[: self.sequence_len]
        return record, tokens, mask_name_token, mask

    @timer
    def _handle_masking(self, mask):
        masking_probability = self._masking_probability()
        n_masked = np.random.binomial(mask.shape[0], masking_probability)
        n_masked = min(n_masked, mask.sum())
        to_mask = np.where(mask)[0]
        to_mask = np.random.choice(to_mask, size=n_masked, replace=False)
        return to_mask, masking_probability

    @timer
    def _generate_masked_tokens_and_ids(self, tokens, to_mask, mask_name_token):
        masked = [token if it not in to_mask else self.encoder.mask_token for it, token in enumerate(tokens)]
        tokens = [mask_name_token] + tokens
        token_ids = self.encoder.tokens_to_token_ids(tokens)
        masked = [mask_name_token] + masked
        masked_ids = self.encoder.tokens_to_token_ids(masked)
        return token_ids, masked_ids

    @timer
    def _create_output(self, token_ids, masked_ids, to_mask):
        mask_tensor = torch.zeros(self.sequence_len, dtype=torch.bool)
        mask_tensor[to_mask] = True
        labels = torch.tensor(token_ids, dtype=torch.long)
        labels[1:][~mask_tensor] = -100
        labels[0] = -100
        out = {"labels": labels, "input_ids": torch.tensor(masked_ids, dtype=torch.long), "mask": mask_tensor}
        return out

    def __rich_repr__(self):
        yield "MyMaskedMidiDataset"
        yield "size", len(self)
        yield "encoder", self.encoder

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        record, tokens, mask_name_token, mask = self._fetch_record(idx)
        to_mask, masking_probability = self._handle_masking(mask)
        token_ids, masked_ids = self._generate_masked_tokens_and_ids(tokens, to_mask, mask_name_token)
        out = self._create_output(token_ids, masked_ids, to_mask)
        out["source"] = record["source"]
        out["masking_probability"] = masking_probability
        return out

    @property
    def tokens_per_record(self) -> int:
        return self.sequence_len
