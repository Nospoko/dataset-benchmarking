import itertools

import pandas as pd

from data_midi_bert.quantizer import MidiQuantizer


class MidiEncoder:
    def __init__(self):
        self.token_to_id = None

    def record_to_tokens(self, record: dict) -> list[str]:
        raise NotImplementedError("Your encoder needs *record_to_tokens* implementation")

    def tokens_to_piece_frame(self, tokens: list[str]) -> pd.DataFrame:
        raise NotImplementedError("Your encoder needs *tokens_to_piece_frame* implementation")

    def token_ids_to_piece_frame(self, token_ids: list[int]) -> pd.DataFrame:
        tokens = [self.vocab[token_id] for token_id in token_ids]
        df = self.tokens_to_piece_frame(tokens)

        return df

    def record_to_token_ids(self, record: dict) -> list[int]:
        tokens = self.record_to_tokens(record)
        token_ids = [self.token_to_id[token] for token in tokens]
        return token_ids

    def tokens_to_token_ids(self, tokens: list[str]) -> list[int]:
        token_ids = [self.token_to_id[token] for token in tokens]
        return token_ids


class QuantizedMidiEncoder(MidiEncoder):
    def __init__(self, quantizer: MidiQuantizer, special_tokens: list[str]):
        super().__init__()
        self.quantizer = quantizer
        self.keys = ["pitch", "dstart_bin", "duration_bin", "velocity_bin"]

        default_special_tokens = ["<CLS>", "<MASK>"]
        self.special_tokens = default_special_tokens + special_tokens

        self.vocab = list(self.special_tokens)

        # add midi tokens to vocab
        self._build_midi_vocab()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}

    def __rich_repr__(self):
        yield "QuantizedMidiEncoder"
        yield "vocab_size", self.vocab_size

    @property
    def mask_id(self) -> int:
        return self.token_to_id[self.mask_token]

    @property
    def mask_token(self) -> str:
        return "<MASK>"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _build_midi_vocab(self):
        src_iterators_product = itertools.product(
            # Always include 88 pitches
            range(21, 109),
            range(self.quantizer.n_dstart_bins),
            range(self.quantizer.n_duration_bins),
            range(self.quantizer.n_velocity_bins),
        )

        # These are all bin ids, not true values (except pitch)
        for pitch, dstart, duration, velocity in src_iterators_product:
            key = f"{pitch}-{dstart}-{duration}-{velocity}"
            self.vocab.append(key)

    def record_to_tokens(self, record: dict) -> list[str]:
        quantized_record = self.quantizer.quantize_record(record)

        tokens = []
        n_samples = len(quantized_record[self.keys[0]])
        for idx in range(n_samples):
            token = "-".join([f"{quantized_record[key][idx]:0.0f}" for key in self.keys])
            tokens.append(token)

        return tokens

    def tokens_to_piece_frame(self, tokens: list[str]) -> pd.DataFrame:
        samples = []
        for token in tokens:
            if token in self.specials:
                continue

            values_txt = token.split("-")
            values = [eval(txt) for txt in values_txt]
            samples.append(values)

        df = pd.DataFrame(samples, columns=self.keys)

        return df
