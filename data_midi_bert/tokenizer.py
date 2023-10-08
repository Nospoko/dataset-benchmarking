import logging

import pandas as pd
from transformers.tokenization_utils import PreTrainedTokenizer

log = logging.getLogger(__name__)


class MidiTokenizer(PreTrainedTokenizer):
    def __init__(self, midi_vocab: list[str], mask_tokens: list[str], **kwargs):
        super().__init__(**kwargs)

        self.midi_vocab = midi_vocab
        # Types of mask
        self.mask_tokens = mask_tokens
        # Token actually doing the masking
        self.mask_token = "<MASK>"

        self.vocab = midi_vocab + mask_tokens + [self.mask_token]

        # Make a copy and add special tokens
        self.tokens_to_ids = {token: it for it, token in enumerate(self.vocab)}
        self.ids_to_tokens = {v: k for k, v in self.tokens_to_ids.items()}

        log.info(f"Number of tokens: {self.n_tokens}")

    @property
    def vocab_size(self) -> int:
        return self.n_tokens

    @property
    def n_tokens(self) -> int:
        return len(self.tokens_to_ids)

    @property
    def mask_token_id(self):
        return self.tokens_to_ids["<MASK>"]

    def encode_record(self, record: dict):
        tokens = [record["mask_type"]] + record["tokens"]
        return self.encode(tokens)

    def record_to_tokens(self, record: dict):
        keys = ["dstart_bin", "duration_bin", "velocity_bin", "pitch"]
        tokens = []
        n_samples = len(record[keys[0]])
        for idx in range(n_samples):
            token = "_".join([f"{record[key][idx]:0.0f}" for key in keys])
            tokens.append(token)

        return tokens

    def encode_frame(self, df: pd.DataFrame, **encode_kwargs):
        # TODO: Not sure where to put this stuff
        # it's entangled with the logic of vocab creation
        # SERIALIZATION?
        tokens = df.apply(lambda row: f"{row.dstart_bin}_{row.duration_bin}_{row.velocity_bin}_{row.pitch}", axis=1)

        encoded = self.encode(tokens.tolist(), **encode_kwargs)
        return encoded

    def untokenize(self, token_ids: list[int]) -> pd.DataFrame:
        rows = []
        for token_id in token_ids:
            dstart, duration, velocity, pitch = self.ids_to_tokens[token_id].split("_")
            row = dict(
                pitch=int(pitch),
                dstart_bin=int(dstart),
                duration_bin=int(duration),
                velocity_bin=int(velocity),
            )
            rows.append(row)
        out = pd.DataFrame(rows)
        return out

    def convert_tokens_to_ids(self, tokens):
        return [self.tokens_to_ids.get(token) for token in tokens]

    def convert_ids_to_tokens(self, token_ids, **kwargs):
        return [self.ids_to_tokens.get(token_id) for token_id in token_ids]

    def __rich_repr__(self):
        yield "MidiTokenizer"
        yield "n_tokens", self.n_tokens
