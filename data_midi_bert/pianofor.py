import os

from omegaconf import DictConfig
from datasets import load_dataset
from transformers import PreTrainedTokenizer

from data_midi_bert.dataset import BERTDataset
from data_midi_bert.utils import hf_dataset_to_sentences


def make_dataset(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizer,
) -> BERTDataset:
    token = os.getenv("HF_TOKEN")

    # Training set ...
    sentences = hf_dataset_to_sentences(
        dataset=load_dataset("roszcz/pianofor-ai-sustain", split="train", use_auth_token=token),
        data_cfg=cfg.data,
        encoder=tokenizer.encoder,
    )
    dataset = BERTDataset(sentences, tokenizer, mlm_probability=cfg.train.mlm_probability)

    return dataset


def make_combined_dataset(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizer,
) -> BERTDataset:
    token = os.getenv("HF_TOKEN")

    # Training set ...
    pfa_sentences = hf_dataset_to_sentences(
        dataset=load_dataset("roszcz/pianofor-ai-sustain", split="train", use_auth_token=token),
        data_cfg=cfg.data,
        encoder=tokenizer.encoder,
    )
    maestro_sentences = hf_dataset_to_sentences(
        dataset=load_dataset("roszcz/maestro-v1-sustain", split="train"),
        data_cfg=cfg.data,
        encoder=tokenizer.encoder,
    )
    sentences = pfa_sentences + maestro_sentences
    dataset = BERTDataset(sentences, tokenizer, mlm_probability=cfg.train.mlm_probability)

    return dataset
