import os

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

from data_midi_bert.masking import AwesomeMasks
from data_midi_bert.quantizer import MidiQuantizer
from data_midi_bert.encoder import QuantizedMidiEncoder
from benchmarking.timer import reset_timings, function_timings
from data_midi_bert.modularized_dataset import MyMaskedMidiDataset

quantizer = MidiQuantizer(
    n_dstart_bins=5,
    n_duration_bins=4,
    n_velocity_bins=4,
)

my_masks = AwesomeMasks()
encoder = QuantizedMidiEncoder(
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

reset_timings()
loader = DataLoader(test_set, batch_size=64, shuffle=True)

tokens = 0
for _ in range(200):
    batch = next(iter(loader))
    tokens += batch["input_ids"].nelement()

print(f"Total tokens: {tokens}")
fetch_time = function_timings["_fetch_record"]
handle_masking_time = function_timings["_handle_masking"]
generate_masked_tokens_and_ids_time = function_timings["_generate_masked_tokens_and_ids"]
create_output_time = function_timings["_create_output"]

print(f"average fetch time: {np.mean(fetch_time):.3f}")
print(f"average handle masking time: {np.mean(handle_masking_time):.3f}")
print(f"average generate masked tokens and ids time: {np.mean(generate_masked_tokens_and_ids_time):.3f}")
print(f"average create output time: {np.mean(create_output_time):.3f} ms")
