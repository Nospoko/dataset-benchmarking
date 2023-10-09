import os
import argparse

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader

from data_masked.quantizer import MidiQuantizer
from data_masked.tokenizer import QuantizedMidiEncoder
from data_masked.modularized_dataset import MidiDataset
from benchmarking.timer import reset_timings, function_timings
from data_midi_bert.modularized_dataset import MyMaskedMidiDataset
from data_midi_bert.masking import AwesomeMasks as AwesomeMasksBert
from data_midi_bert.quantizer import MidiQuantizer as MidiQuantizerBert
from data_midi_bert.encoder import QuantizedMidiEncoder as QuantizedMidiEncoderBert


def prepare_dataset(dataset_name: str):
    token_access = ""
    if dataset_name == "midi-bert":
        token_access = "input_ids"
        quantizer = MidiQuantizerBert(
            n_dstart_bins=5,
            n_duration_bins=4,
            n_velocity_bins=4,
        )

        my_masks = AwesomeMasksBert()
        encoder = QuantizedMidiEncoderBert(
            quantizer=quantizer,
            special_tokens=my_masks.vocab,
        )
        token = os.environ["HUGGINGFACE_TOKEN"]
        maestro_test = load_dataset("SneakyInsect/masked-maestro", split="test", use_auth_token=token)

        test_set = MyMaskedMidiDataset(
            dataset=maestro_test,
            encoder=encoder,
            sequence_len=60,
            mask_probability_min=0.1,
            mask_probability_max=0.41,
        )

    elif dataset_name == "masked-midi":
        token_access = "input_token_ids"

        quantizer = MidiQuantizer(
            n_dstart_bins=5,
            n_duration_bins=4,
            n_velocity_bins=4,
        )
        tokenizer = QuantizedMidiEncoder(
            dstart_bin=5,
            duration_bin=4,
            velocity_bin=4,
        )

        token = os.environ["HUGGINGFACE_TOKEN"]
        maestro_test = load_dataset("SneakyInsect/masked-maestro-2", split="test", use_auth_token=token)

        test_set = MidiDataset(
            maestro_test,
            quantizer,
            tokenizer,
            pitch_shift_probability=0.0,
            time_stretch_probability=0.0,
            masking_probability=0.15,
        )

    return test_set, token_access


if __name__ == "__main__":
    # add argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument(
        "--dataset", type=str, choices=["masked-midi", "midi-bert"], default="masked-midi"
    )  # masked-midi or midi-bert, might want to change that
    args = parser.parse_args()

    test_set, token_access = prepare_dataset(args.dataset)
    reset_timings()
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    tokens = 0
    for _ in tqdm(range(args.num_batches)):
        batch = next(iter(loader))
        tokens += batch[token_access].nelement()

    print(f"Total tokens: {tokens}")

    for key, value in function_timings.items():
        print(f"mean time for {key}: {np.mean(value)}")
