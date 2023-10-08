import numpy as np
import pandas as pd


class Mask:
    token: str

    def __rich_repr__(self):
        yield self.token

    def mask(self, df: pd.DataFrame):
        raise NotImplementedError("Subclass must implement this method")

    def masking_space(self, df: pd.DataFrame):
        raise NotImplementedError("Subclass must implement this method")


class RandomMask(Mask):
    token: str = "<Random Mask>"

    def mask(self, df: pd.DataFrame, p: float) -> pd.DataFrame:
        noise = np.random.random(df.shape[0])
        df["masked"] = noise <= p
        return df

    def masking_space(self, df: pd.DataFrame) -> pd.Series:
        return df.pitch > 0


class LeftHandMask(Mask):
    token: str = "<LH Mask>"

    def mask(self, df: pd.DataFrame, p: float) -> pd.DataFrame:
        assert p < 0.5, "This strategy targets only a half on available notes, not possible to mask more than 0.5"
        n_masked = np.random.binomial(df.shape[0], p)
        n_masked = min(n_masked, df.shape[0] // 2)

        ids = self.masking_space(df)
        to_mask = np.random.choice(df[ids].index, size=n_masked, replace=False)
        df["masked"] = False
        df.loc[to_mask, "masked"] = True
        return df

    def masking_space(self, df: pd.DataFrame) -> pd.Series:
        middle_pitch = np.median(df.pitch)
        ids = df.pitch <= middle_pitch
        return ids


class RightHandMask(Mask):
    token: str = "<RH Mask>"

    def mask(self, df: pd.DataFrame, p: float) -> pd.DataFrame:
        assert p < 0.5, "This strategy targets only a half of available notes, not possible to mask more than 0.5"
        n_masked = np.random.binomial(df.shape[0], p)
        n_masked = min(n_masked, df.shape[0] // 2)

        ids = self.masking_space(df)
        to_mask = np.random.choice(df[ids].index, size=n_masked, replace=False)
        df["masked"] = False
        df.loc[to_mask, "masked"] = True
        return df

    def masking_space(self, df: pd.DataFrame) -> pd.Series:
        middle_pitch = np.median(df.pitch)
        ids = df.pitch >= middle_pitch
        return ids


class HarmonicRootMask(Mask):
    token: str = "<Harmonic Root Mask>"

    def mask(self, df: pd.DataFrame, p: float) -> pd.DataFrame:
        ids = self.masking_space(df)

        n_masked = min(np.random.binomial(df.shape[0], p), ids.sum())
        to_mask = np.random.choice(df[ids].index, size=n_masked, replace=False)
        df["masked"] = False
        df.loc[to_mask, "masked"] = True
        return df

    def masking_space(self, df: pd.DataFrame) -> pd.Series:
        df["absolute_pitch"] = df.pitch % 12

        top_k = 3
        top_pitches = df.absolute_pitch.value_counts().index[:top_k]
        ids = df.absolute_pitch.isin(top_pitches)
        return ids


class HarmonicOutliersMask(Mask):
    token: str = "<Harmonic Outliers Mask>"

    def mask(self, df: pd.DataFrame, p: float) -> pd.DataFrame:
        ids = self.masking_space(df)

        n_masked = min(np.random.binomial(df.shape[0], p), ids.sum())
        to_mask = np.random.choice(df[ids].index, size=n_masked, replace=False)
        df["masked"] = False
        df.loc[to_mask, "masked"] = True
        return df

    def masking_space(self, df: pd.DataFrame) -> pd.Series:
        df["absolute_pitch"] = df.pitch % 12
        top_k = 3
        top_pitches = df.absolute_pitch.value_counts().index[top_k:]
        ids = df.absolute_pitch.isin(top_pitches)
        return ids


class AwesomeMasks:
    def __init__(self):
        self.masks = [
            RandomMask(),
            LeftHandMask(),
            RightHandMask(),
            HarmonicRootMask(),
            HarmonicOutliersMask(),
        ]

    def __rich_repr__(self):
        yield self.masks

    def apply(self, df: pd.DataFrame, probability: float) -> tuple[pd.DataFrame, str]:
        mask = np.random.choice(self.masks)
        df = mask.mask(df, probability)
        return df, mask.token

    @property
    def vocab(self) -> list[str]:
        vocab = [mask.token for mask in self.masks]
        return vocab
