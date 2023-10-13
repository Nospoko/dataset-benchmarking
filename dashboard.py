import streamlit as st
from datasets import load_dataset

from my_tokenizer.tokenizer import REMITokenizer
from utils.midi_tools import save_midi, plot_piano_roll, convert_to_dstart, to_fortepyan_midi, render_midi_to_mp3

st.set_page_config(layout="wide", page_title="Tokenizer effect", page_icon=":musical_keyboard")


def get_midi(record):
    fortepyan_midi = to_fortepyan_midi(
        pitch=record["pitch"],
        dstart=record["dstart"],
        duration=record["duration"],
        velocity=record["velocity"],
    )

    return fortepyan_midi


def compute_time_mse(original_record, decoded_record, attribute="dstart"):
    """Calculate Mean Squared Error for start times between original and decoded records."""
    assert len(original_record[attribute]) == len(decoded_record[attribute]), "Both records should have the same length"
    errors = [(o - r) ** 2 for o, r in zip(original_record[attribute], decoded_record[attribute])]
    print(errors)
    mse = sum(errors) / len(errors)
    return mse


def display_audio(fortepyan_midi, title="MIDI", filename="midi"):
    st.title(title)

    img_path = plot_piano_roll(
        fortepyan_midi,
        title=title,
    )
    original_mp3_path = render_midi_to_mp3(
        piece=fortepyan_midi,
        filename=f"{filename}.mp3",
    )
    # save midi file
    save_midi(fortepyan_midi, filename=f"{filename}.mid")

    st.image(img_path, width=800)
    st.audio(original_mp3_path, format="audio/mp3", start_time=0)


def main():
    data_train = load_dataset("roszcz/maestro-v1-sustain", split="train")

    with st.sidebar:
        # Show available checkpoints
        segments = st.selectbox(label="segments", options=[x for x in range(1, 10)])
        record_idx = st.text_input(label="record_idx", value="0")

        # Tokenizer settings
        st.subheader("Tokenizer Settings")
        bpm = st.slider("BPM", min_value=60, max_value=240, value=120)
        tpb = st.slider("TPB", min_value=120, max_value=960, value=480)
        resolution = st.slider("Resolution", min_value=60, max_value=960, value=480)
        fraction = st.slider("Fraction", min_value=1, max_value=32, value=16)
        velocity_bins = st.slider("Velocity Bins", min_value=1, max_value=128, value=32)
        duration_bins = st.slider("Duration Bins", min_value=1, max_value=128, value=64)

        record = data_train[int(record_idx)]["notes"]

    # Use the settings to create the tokenizer
    tokenizer = REMITokenizer(
        bpm=bpm, tpb=tpb, resolution=resolution, fraction=fraction, velocity_bins=velocity_bins, duration_bins=duration_bins
    )

    tokenized_record, num_of_tokens = tokenizer.encode(record, segments=segments)
    decoded_record = tokenizer.decode(tokenized_record)

    st.subheader("Tokenizer info")
    st.write(f"Vocabulary size: {tokenizer.vocabulary_size()}")
    st.write(f"{num_of_tokens} tokens used to encode {segments*15} notes")

    record = convert_to_dstart(record, split=segments * 15)
    decoded_record = convert_to_dstart(decoded_record)
    time_mse = compute_time_mse(record, decoded_record)

    st.write(f"MSE for dstart: {time_mse}")

    fortepyan_midi = get_midi(record)
    fortepyan_midi_decoded = get_midi(decoded_record)

    display_audio(fortepyan_midi, title="Original", filename="original")
    display_audio(fortepyan_midi_decoded, title="Decoded", filename="decoded")


if __name__ == "__main__":
    main()
