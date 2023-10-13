import pandas as pd


def serialize_frame(df: pd.DataFrame) -> list[str]:
    def serialize_row(row: pd.Series) -> str:
        out = f"{row.dstart_bin}_{row.duration_bin}_{row.velocity_bin}_{row.pitch}"
        return out

    # Applying only to taret columns prevents pandas typing confusion
    cols = ["dstart_bin", "duration_bin", "velocity_bin", "pitch"]
    tokens = df[cols].apply(serialize_row, axis=1)
    return tokens.tolist()


def deserialize_tokens(tokens: list[str]) -> pd.DataFrame:
    rows = []
    for token in tokens:
        dstart_bin, duration_bin, velocity_bin, pitch = token.split("_")
        row = dict(
            pitch=int(pitch),
            dstart_bin=int(dstart_bin),
            duration_bin=int(duration_bin),
            velocity_bin=int(velocity_bin),
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
