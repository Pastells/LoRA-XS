import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import regex
import torch
from torch_audiomentations import (
    AddColoredNoise,
    BandPassFilter,
    Compose,
    PitchShift,
)
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def _str2bool(string):
    """Converts a string to a boolean value."""
    str2val = {"true": True, "false": False}
    if string and string.lower() in str2val:
        return str2val[string.lower()]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def initialize_whisper_objects(
    model_id, language="ca", task="transcribe", quantization_config=None
):
    global feature_extractor, tokenizer, processor, normalizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(model_id, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task)
    normalizer = BasicTextNormalizer()
    if quantization_config:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, quantization_config=quantization_config
        )
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    return feature_extractor, tokenizer, processor, model


def prepare_dataset(batch, normalize=True):
    audio = batch["audio"]

    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    input_str = batch["transcription"].strip()
    if normalize:
        input_str = normalizer(input_str)
    batch["labels"] = tokenizer(input_str).input_ids
    return batch


def prepare_dataset_mel(batch, normalize=True):
    """Prepare dataset for training with mel spectrograms."""
    batch["input_features"] = torch.load(batch["mel"])
    batch["input_length"] = batch["length"]
    input_str = batch["transcription"].strip()
    if normalize:
        input_str = normalizer(input_str)
    batch["labels"] = tokenizer(input_str).input_ids
    return batch


def prepare_dataset_aug(batch, compose_function=None, normalize=True):
    audio = batch["audio"]
    # Apply any specified audio augmentation
    if compose_function:
        # Shape T -> 1 x 1 x T
        audio.array = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        audio.array = compose_function(
            audio.array,
            sample_rate=audio["sampling_rate"],
        ).squeeze((0, 1))

    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    input_str = batch["transcription"].strip()
    if normalize:
        input_str = normalizer(input_str)
    batch["labels"] = tokenizer(input_str).input_ids
    return batch


def aug_function(
    # techniques: List[str] | str | None = None,
    techniques=None,
    sample_rate: int = 16000,
    p: float = 0.3,
    # Colored noise parameters
    min_snr_in_db: float = 15.0,
    max_snr_in_db: float = 30.0,
    # Band-pass filter parameters
    min_center_frequency: float = 1200.0,
    max_center_frequency: float = 1800.0,
    # Pitch shift parameters
    min_transpose_semitones: float = -0.5,
    max_transpose_semitones: float = 0.5,
):
    """
    Create a Compose object with the specified audio augmentation techniques.
    """
    if techniques is None:
        return None
    augmentations = []
    if "All" in techniques:
        techniques = [
            "AddColoredNoise",
            "BandPassFilter",
            "PitchShift",
        ]
    if "AddColoredNoise" in techniques:
        augmentations.append(
            AddColoredNoise(
                min_snr_in_db=min_snr_in_db,
                max_snr_in_db=max_snr_in_db,
                p=p,
            )
        )
    if "BandPassFilter" in techniques:
        augmentations.append(
            BandPassFilter(
                min_center_frequency=min_center_frequency,
                max_center_frequency=max_center_frequency,
                p=p,
            )
        )
    if "PitchShift" in techniques:
        augmentations.append(
            PitchShift(
                sample_rate=sample_rate,
                min_transpose_semitones=min_transpose_semitones,
                max_transpose_semitones=max_transpose_semitones,
                p=p,
            )
        )
    if not augmentations:
        raise ValueError(
            "No valid augmentation techniques provided. "
            "Available techniques: 'AddColoredNoise', 'BandPassFilter', 'PitchShift'. "
            "Or use 'All' to apply all techniques."
        )
    return Compose(augmentations)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        # print("before pop", batch)
        # batch.pop("input_ids", None)
        # print(batch)

        return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s)
    )


class BasicTextNormalizer:
    def __init__(self, split_letters: bool = False):
        self.clean = remove_symbols
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        return s


def get_specific_layer_names(model):
    """Gets the Lora trainable layers from the model."""
    layer_names = []

    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            layer_names.append(name)

    return layer_names
