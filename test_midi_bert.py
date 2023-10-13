import os
import time

from tqdm import tqdm
from datasets import load_dataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from data_midi_bert.masking import AwesomeMasks
from data_midi_bert.quantizer import MidiQuantizer
from data_midi_bert.dataset import MyMaskedMidiDataset
from data_midi_bert.encoder import QuantizedMidiEncoder

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

BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256]

results = []

for batch_size in tqdm(BATCH_SIZES):
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Warm-up phase: loop but don't measure the time
    # This is to avoid including cache time in the measurement
    for _ in range(1):
        for _ in loader:
            pass

    # Actual measurement
    start_time = time.time()

    total_tokens = 0
    while total_tokens < 1_000_000:
        batch = next(iter(loader))
        total_tokens += batch["input_ids"].nelement()

    end_time = time.time()
    total_time = end_time - start_time

    time_per_kilotoken = total_time * 1_000 / (total_tokens / 1000)  # Convert to miliseconds for kilotoken
    results.append((batch_size, time_per_kilotoken))

# Visualize the results
batch_sizes, times = zip(*results)

plt.plot(batch_sizes, times, marker="o")
plt.xlabel("Batch Size")
plt.ylabel("Time per Kilo-token (ms)")
plt.suptitle("Time per Kilo-token vs. Batch Size", fontsize=16)
plt.title("MIDI BERT", fontsize=10)
plt.grid(True)
plt.show()
plt.savefig("midi_bert.png")
