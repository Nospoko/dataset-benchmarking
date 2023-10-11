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
        record = data_train[int(record_idx)]["notes"]

    # Load selected checkpoint
    tokenizer = REMITokenizer(bpm=120, tpb=480, resolution=480, fraction=16, velocity_bins=32, duration_bins=64)

    tokenized_record = tokenizer.encode(record, segments=segments)
    decoded_record = tokenizer.decode(tokenized_record)

    record = convert_to_dstart(record, split=segments * 15)
    decoded_record = convert_to_dstart(decoded_record)

    fortepyan_midi = get_midi(record)
    fortepyan_midi_decoded = get_midi(decoded_record)

    display_audio(fortepyan_midi, title="Original", filename="original")
    display_audio(fortepyan_midi_decoded, title="Decoded", filename="decoded")


if __name__ == "__main__":
    main()
