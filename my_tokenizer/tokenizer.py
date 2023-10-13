import numpy as np
import pandas as pd


class Tokenizer:
    def encode(self, data):
        raise NotImplementedError

    def decode(self, tokenized_data):
        raise NotImplementedError


class REMITokenizer(Tokenizer):
    def __init__(self, bpm=120, tpb=480, resolution=480, fraction=16, velocity_bins=32, duration_bins=64):
        """ """
        self.bpm = bpm
        self.tpb = tpb
        self.resolution = resolution
        self.fraction = fraction
        self.velocity_bins = np.linspace(0, 128, velocity_bins + 1, dtype=int)
        # assume that the longest note is 20 seconds
        self.duration_bins = np.linspace(0, 4800, duration_bins + 1, dtype=int)

        self.ticks_per_second = self.tpb / (60 / self.bpm)

    def time_to_ticks(self, data_df):
        seconds_per_beat = 60 / self.bpm
        data_df["start_ticks"] = round(data_df["start"] / seconds_per_beat * self.tpb)
        data_df["end_ticks"] = round(data_df["end"] / seconds_per_beat * self.tpb)
        data_df["time"] = data_df["start_ticks"]

    def quantize_ticks(self, data_df):
        ticks = self.resolution
        grids = np.arange(0, data_df[["start_ticks", "end_ticks"]].max().max() + ticks, ticks, dtype=int)
        data_df["time"] = grids[np.abs(np.subtract.outer(grids, data_df["start_ticks"].to_numpy())).argmin(axis=0)]

    def group_data(self, tokenized_data):
        ticks_per_bar = self.resolution
        max_time = tokenized_data["end_ticks"].max()
        downbeats = np.arange(0, max_time + ticks_per_bar, ticks_per_bar)
        groups = []

        for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
            bar_notes = tokenized_data[(tokenized_data["start_ticks"] >= db1) & (tokenized_data["start_ticks"] < db2)]
            groups.append([db1, bar_notes, db2])

        return groups

    def vocabulary_size(self):
        # For "Bar" event
        bar_vocab = 1

        # For "Position" event
        position_vocab = self.fraction

        # For "Note Velocity" event
        velocity_vocab = len(self.velocity_bins) - 1  # Number of bins - 1

        # For "Note On" event (MIDI pitches range from 0 to 127)
        note_on_vocab = 128

        # For "Note Duration" event
        duration_vocab = len(self.duration_bins) - 1  # Number of bins - 1

        # Summing all the individual vocabularies
        total_vocab = bar_vocab + position_vocab + velocity_vocab + note_on_vocab + duration_vocab

        return total_vocab

    def groups_to_events(self, groups):
        events = []

        for group in groups:
            bar_st, bar_notes, bar_et = group

            # Add BAR event for each group
            events.append(
                {
                    "name": "Bar",
                    "time": bar_st,  # start of the bar
                    "value": None,
                    "text": "BAR",
                }
            )

            for _, note_data in bar_notes.iterrows():
                # Position
                flags = np.linspace(bar_st, bar_et, self.fraction, endpoint=False)
                index = np.argmin(abs(flags - note_data["start_ticks"]))
                events.append(
                    {
                        "name": "Position",
                        "time": bar_st,
                        "value": f"{index+1}/{self.fraction}",
                        "text": str(note_data["start_ticks"]),
                    }
                )

                # Velocity
                velocity_index = np.searchsorted(self.velocity_bins, note_data["velocity"], side="right") - 1
                events.append(
                    {
                        "name": "Note Velocity",
                        "time": bar_st,
                        "value": velocity_index,
                        "text": f'{note_data["velocity"]}/{self.velocity_bins[velocity_index]}',
                    }
                )

                # Pitch
                events.append(
                    {
                        "name": "Note On",
                        "time": bar_st,
                        "value": note_data["pitch"],
                        "text": str(note_data["pitch"]),
                    }
                )

                # Duration
                duration = note_data["end_ticks"] - note_data["start_ticks"]
                index = np.argmin(abs(self.duration_bins - duration))
                events.append(
                    {
                        "name": "Note Duration",
                        "time": bar_st,
                        "value": index,
                        "text": f"{duration}/{self.duration_bins[index]}",
                    }
                )

        return events

    def random_slice(self, data_df, segment_length):
        return data_df.iloc[:segment_length]

    def encode(self, data_df: pd.DataFrame, segments=1):
        """
        Encode the data dictionary into a list of events

        Parameters
        ----------
        data_dict : dict
            Dictionary containing the note data
        segments : int
            Number of segments; each segment corresponds to 15 notes

        Returns
        -------
        tokenized_data : list
            List of events (tokenized data)
        number_of_tokens : int
            Number of tokens used to encode the data
        """
        segment_length = segments * 15
        sliced_data = self.random_slice(data_df, segment_length)
        self.time_to_ticks(sliced_data)
        self.quantize_ticks(sliced_data)
        groups = self.group_data(sliced_data)
        tokenized_data = self.groups_to_events(groups)

        return tokenized_data, len(tokenized_data)

    def decode(self, tokenized_data):
        result = {"start": [], "end": [], "pitch": [], "velocity": [], "duration": []}

        # Temporary variables to store note data
        start_ticks, end_ticks, pitch, velocity = None, None, None, None

        for event in tokenized_data:
            event_name = event["name"]

            if event_name == "Bar":
                if start_ticks and end_ticks and pitch and velocity:
                    # Previous note was incomplete, handle it here if needed
                    # For now, we just reset it
                    start_ticks, end_ticks, pitch, velocity = None, None, None, None
                continue

            if event_name == "Position":
                # Decode the position (start_ticks) from the fraction
                fraction_val = int(event["value"].split("/")[0])
                bar_st = event["time"]
                start_ticks = bar_st + (self.resolution * (fraction_val - 1) / (self.fraction - 1))

            elif event_name == "Note Velocity":
                velocity_index = event["value"]
                velocity = (self.velocity_bins[velocity_index] + self.velocity_bins[velocity_index + 1]) / 2
                velocity = int(velocity)

            elif event_name == "Note On":
                pitch = event["value"]

            elif event_name == "Note Duration":
                index = event["value"]
                duration = self.duration_bins[index]
                end_ticks = start_ticks + duration

            if start_ticks is not None and pitch is not None and velocity is not None and end_ticks is not None:
                seconds_per_beat = 60 / self.bpm
                start_time = start_ticks / self.tpb * seconds_per_beat
                end_time = end_ticks / self.tpb * seconds_per_beat

                result["start"].append(start_time)
                result["end"].append(end_time)
                result["pitch"].append(pitch)
                result["velocity"].append(velocity)
                result["duration"].append(end_time - start_time)
                start_ticks, end_ticks, pitch, velocity = None, None, None, None  # Reset for the next note

        return result
