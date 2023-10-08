import os
import time

from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader

from data_masked.dataset import MidiDataset
from data_masked.masking import AwesomeMasks
from data_masked.quantizer import MidiQuantizer
from data_masked.tokenizer import QuantizedMidiEncoder

# Initialization
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

masks = AwesomeMasks()

token = os.environ["HUGGINGFACE_TOKEN"]
maestro_test = load_dataset("SneakyInsect/masked-maestro-2", split="test", use_auth_token=token)

test_set = MidiDataset(
    maestro_test, quantizer, tokenizer, pitch_shift_probability=0.0, time_stretch_probability=0.0, masking_probability=0.15
)

# Set desired batch sizes to loop over
BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256]

results = []

for batch_size in tqdm(BATCH_SIZES):
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Warm-up phase: loop a few times but don't measure the time
    # This is to avoid including cache time in the measurement
    for _ in range(3):
        for _ in loader:
            pass

    # Actual measurement
    start_time = time.time()

    total_tokens = 0
    while total_tokens < 1_000_000:
        batch = next(iter(loader))
        total_tokens += batch["input_token_ids"].nelement()

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
plt.title("Masked MIDI Modeling", fontsize=10)
plt.grid(True)
plt.show()
plt.savefig("masked_modeling.png")
