from omegaconf import DictConfig
from datasets import load_dataset
from transformers import PreTrainedTokenizer

from data_midi_bert.dataset import BERTDataset
from data_midi_bert.utils import hf_dataset_to_sentences


def make_dataset(
    split: str,
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizer,
) -> BERTDataset:
    # Training set ...
    sentences = hf_dataset_to_sentences(
        dataset=load_dataset("roszcz/maestro-v1-sustain", split=split),
        data_cfg=cfg.data,
        encoder=tokenizer.encoder,
    )
    dataset = BERTDataset(sentences, tokenizer, mlm_probability=cfg.train.mlm_probability)

    return dataset
