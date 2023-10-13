import numpy as np
import pandas as pd


def change_speed(df: pd.DataFrame, factor: float = None) -> tuple[pd.DataFrame, float]:
    if not factor:
        slow = 0.8
        change_range = 0.4
        factor = slow + np.random.random() * change_range

    df.start /= factor
    df.end /= factor
    df.dstart /= factor
    df.duration = df.end - df.start
    return df, factor


def pitch_shift(df: pd.DataFrame, shift_threshold: int = 5) -> tuple[pd.DataFrame, int]:
    # No more than given number of steps
    PITCH_LOW = 21
    PITCH_HI = 108
    low_shift = -min(shift_threshold, df.pitch.min() - PITCH_LOW)
    high_shift = min(shift_threshold, PITCH_HI - df.pitch.max())

    if low_shift > high_shift:
        shift = 0
        print("Pitch shift edge case:", df.pitch.min(), df.pitch.max())
    else:
        shift = np.random.randint(low=low_shift, high=high_shift + 1)
    df.pitch += shift
    return df, shift
