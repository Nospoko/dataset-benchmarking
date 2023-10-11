import numpy as np


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

    def time_to_ticks(self, data):
        seconds_per_beat = 60 / self.bpm
        data["start_ticks"] = round(data["start"] / seconds_per_beat * self.tpb)
        data["end_ticks"] = round(data["end"] / seconds_per_beat * self.tpb)
        data["time"] = data["start_ticks"]

    def quantize_ticks(self, data):
        ticks = self.resolution
        grids = np.arange(0, max(data["start_ticks"], data["end_ticks"]) + ticks, ticks, dtype=int)

        data["time"] = grids[np.abs(grids - data["start_ticks"]).argmin()]

    def group_data(self, data):
        ticks_per_bar = self.resolution
        max_time = data["end_ticks"]
        downbeats = np.arange(0, max_time + ticks_per_bar, ticks_per_bar)
        groups = []

        for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
            if db1 <= data["start_ticks"] < db2:
                groups.append([db1, data, db2])

        return groups

    def group_to_events(self, group):
        events = []
        note_data = group[1]
        bar_st, bar_et = group[0], group[-1]

        # Position
        flags = np.linspace(bar_st, bar_et, self.fraction, endpoint=False)
        index = np.argmin(abs(flags - note_data["start_ticks"]))
        events.append(
            {
                "name": "Position",
                "time": note_data["time"],
                "value": f"{index+1}/{self.fraction}",
                "text": str(note_data["start_ticks"]),
            }
        )

        # Velocity
        velocity_index = np.searchsorted(self.velocity_bins, note_data["velocity"], side="right") - 1
        events.append(
            {
                "name": "Note Velocity",
                "time": note_data["time"],
                "value": velocity_index,
                "text": f'{note_data["velocity"]}/{self.velocity_bins[velocity_index]}',
            }
        )

        # Pitch
        events.append(
            {
                "name": "Note On",
                "time": note_data["time"],
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
                "time": note_data["time"],
                "value": index,
                "text": f"{duration}/{self.duration_bins[index]}",
            }
        )

        return events

    def random_slice(self, data_dict, segment_length):
        """Randomly slice the data dictionary based on segment length."""
        if len(data_dict["duration"]) <= segment_length:
            return data_dict

        start_idx = 0  # fixed at 0 for now until I come up with a better way to handle time ticks.

        sliced_data = {}
        for key, values in data_dict.items():
            sliced_data[key] = values[start_idx : start_idx + segment_length]

        return sliced_data

    def encode(self, data_dict, segments=1):
        segment_length = segments * 15
        print("segment_length:", segment_length)
        sliced_data = self.random_slice(data_dict, segment_length)

        tokenized_data = []
        for i in range(segment_length):
            note_data = {key: val[i] for key, val in sliced_data.items()}
            self.time_to_ticks(note_data)
            self.quantize_ticks(note_data)
            groups = self.group_data(note_data)
            for group in groups:
                tokenized_data.extend(self.group_to_events(group))

        return tokenized_data

    def decode(self, tokenized_data):
        # Initialize the resulting data
        result = {"start": [], "end": [], "pitch": [], "velocity": [], "duration": []}

        # Temporary variables to store data until a full note has been processed
        start_ticks, end_ticks, pitch, velocity = None, None, None, None

        for event in tokenized_data:
            event_name = event["name"]

            if event_name == "Position":
                # Decode the position (start_ticks) from the fraction
                fraction_val = int(event["value"].split("/")[0])
                bar_st = event["time"] - (self.resolution / self.fraction) * (fraction_val - 1)
                bar_et = bar_st + self.resolution
                start_ticks = bar_st + (bar_et - bar_st) * (fraction_val - 1) / (self.fraction - 1)

            elif event_name == "Note Velocity":
                # Decoding velocity from the index and velocity_bins
                velocity_index = event["value"]
                velocity = (self.velocity_bins[velocity_index] + self.velocity_bins[velocity_index + 1]) / 2
                velocity = int(velocity)

            elif event_name == "Note On":
                # Directly get pitch value
                pitch = event["value"]

            elif event_name == "Note Duration":
                # Decode duration to compute end_ticks
                index = event["value"]
                duration = self.duration_bins[index]
                end_ticks = start_ticks + duration

            # Check if we have enough data for a note
            if start_ticks is not None and end_ticks is not None and pitch is not None and velocity is not None:
                # Convert start_ticks and end_ticks to seconds
                seconds_per_beat = 60 / self.bpm
                start_time = start_ticks / self.tpb * seconds_per_beat
                end_time = end_ticks / self.tpb * seconds_per_beat

                # Store in the result and reset temporary variables
                result["start"].append(start_time)
                result["end"].append(end_time)
                result["pitch"].append(pitch)
                result["velocity"].append(velocity)
                result["duration"].append(end_time - start_time)
                start_ticks, end_ticks, pitch, velocity = None, None, None, None

        return result
