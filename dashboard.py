import pandas
import streamlit as st
from datasets import load_dataset
from fortepyan import MidiFile, MidiPiece

from my_tokenizer.tokenizer import REMITokenizer
from utils.midi_tools import save_midi, plot_piano_roll, convert_to_dstart, to_fortepyan_midi, render_midi_to_mp3

st.set_page_config(layout="wide", page_title="Tokenizer effect", page_icon=":musical_keyboard")


def piece_selector() -> pandas.DataFrame:
    # Use sidebar methods for the UI components
    # with st.sidebar:
    uploaded_file = st.file_uploader("Choose a MIDI file", type=["midi", "mid"])
    if uploaded_file is not None:
        midi_file = MidiFile(uploaded_file)
        piece = midi_file.piece
        piece.source["path"] = "file uploaded with streamlit"
    else:
        st.write("Or use a dataset")
        dataset_names = ["roszcz/maestro-v1-sustain"]
        dataset_name = st.selectbox(label="dataset", options=dataset_names)
        split = st.selectbox(label="split", options=["test", "validation", "train"])
        # Test/77 is Chopin "Etude Op. 10 No. 12"
        record_id = st.number_input(label="record id", value=77)

        hf_dataset = load_dataset(dataset_name, split=split)

        # Select one full piece
        record = hf_dataset[record_id]

        piece = MidiPiece.from_huggingface(record)

    return piece.df


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
    with st.sidebar:
        # Tokenizer settings
        st.subheader("Tokenizer Settings")
        segments = st.slider("Segments", min_value=1, max_value=10, value=1)
        bpm = st.slider("BPM", min_value=60, max_value=240, value=120)
        tpb = st.slider("TPB", min_value=120, max_value=960, value=480)
        resolution = st.slider("Resolution", min_value=60, max_value=960, value=480)
        fraction = st.slider("Fraction", min_value=1, max_value=32, value=16)
        velocity_bins = st.slider("Velocity Bins", min_value=1, max_value=128, value=32)
        duration_bins = st.slider("Duration Bins", min_value=1, max_value=128, value=64)

        record = piece_selector()
        st.subheader("Settings explanation")
        st.write(
            """
            - **segments**: number of 15-note segments to encode
            - **BPM**: Beats per minute
            - **TPB**: Ticks per beat
            - **Resolution**: Resolution of the grid in ticks
            - **Fraction**: Fraction of a beat, e.g. 16 means it's patitioned into 16th notes
            - **Velocity Bins**: Number of velocity bins for quantization
            - **Duration Bins**: Number of duration bins for quantization
            """
        )

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
