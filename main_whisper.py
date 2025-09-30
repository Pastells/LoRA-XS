# LORA-XS

# Hay warnings sin sentido, gradient_checkpointing=False, use_cache=False ..
# ruff: noqa: E402
import argparse

# --------------------------------------
# Arg Parser
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, help="GPU to use", default=0)
parser.add_argument(
    "--model_id",
    type=str,
    help="ID of the model to fine-tune",
    default="openai/whisper-large-v3-turbo",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="Directory to save the checkpoints",
    default="checkpoints",
)
parser.add_argument("--wandb_project", type=str, help="Wandb project name", default="")
parser.add_argument("--batch_size", type=int, help="Batch size", default=64)
parser.add_argument("--accumulation_batch", type=int, help="Accumulation batch", default=64)
parser.add_argument("--lora_r", type=int, help="Lora r", default=32)
parser.add_argument("--lora_alpha", type=int, help="Lora alpha, usually 2x lora r", default=64)
parser.add_argument("--loraplus_lr_ratio", type=float, help="Lora+ lr ratio", default=16)
parser.add_argument(
    "--target_modules",
    nargs="+",
    type=str,
    help="Target modules",
    default=["q_proj", "v_proj"],
)
parser.add_argument(
    "--optimizer", type=str, help="Whether to use lora plus optimizer", default="True"
)
parser.add_argument(
    "--scheduler",
    type=str,
    help="type of scheduler to use. EX: constant, lixnear,cosine",
    default="constant",
)
parser.add_argument("--learning_rate", type=float, help="Learning rate", default=1e-5)
parser.add_argument("--n_epochs", type=int, help="Number of epochs", default=1)
# Augmentation
parser.add_argument(
    "--augmentation_factor",
    type=int,
    help="Factor of augmentation to apply. 1 means no dataset size increase",
    default=1,
)
parser.add_argument("--mask_time_prob", type=float, help="Mask time prob Spec Augment", default=0.0)
parser.add_argument(
    "--mask_time_length",
    type=int,
    help="Mask time length Spec Augment",
    default=10,
)
parser.add_argument(
    "--mask_time_min_masks",
    type=int,
    help="Mask time min masks Spec Augment",
    default=1,
)
parser.add_argument(
    "--mask_feature_prob", type=float, help="Mask feature prob Spec Augment", default=0.0
)
parser.add_argument(
    "--mask_feature_length",
    type=int,
    help="Mask feature length Spec Augment",
    default=32,
)
parser.add_argument(
    "--mask_feature_min_masks",
    type=int,
    help="Mask feature min masks Spec Augment",
    default=0,
)

parser.add_argument(
    "--dataset",
    type=str,
    help="Dataset to use. It will default to dataset. Options: All, Down, Paralysis",
    default="All",
)
parser.add_argument(
    "--run_name",
    type=str,
    help="Run name to identify the run. If empty, it will be model_id + timestamp",
    default=None,
)
parser.add_argument(
    "--lora_init",
    type=str,
    help="Lora initialization method. Options: 'lora', 'pissa'. Default 'lora'",
    default="lora",
)
parser.add_argument(
    "--lora_dropout",
    type=float,
    help="Dropout for the lora layers. Default 0.05",
    default=0.05,
)


args = parser.parse_args()

# Print args
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

# --------------------------------------
import os

# CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["WORLD_SIZE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# --------------------------------------
import json
from datetime import datetime
from typing import List

import pandas as pd
import yaml
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from utils.initialization_utils import find_and_initialize
from utils.utils import (
    DataCollatorSpeechSeq2SeqWithPadding,
    _str2bool,
    compute_metrics,
    get_specific_layer_names,
    initialize_whisper_objects,
    prepare_dataset,
)

# Keys:
secret = pd.read_csv("secret.config", header=None)
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_API_KEY"] = secret[1][0]
HF_TOKEN = secret[1][1]
del secret
login(HF_TOKEN)

# WANDB PROJECT
if args.wandb_project == "":
    # if empty model + timestamp
    timestamp = datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
    os.environ["WANDB_PROJECT"] = args.model_id.split("/")[1] + timestamp
else:
    os.environ["WANDB_PROJECT"] = args.wandb_project

# VARIABLES
MAX_INPUT_LENGTH = 30.0  # 30s per audio
MAX_LABEL_LENGTH = 440  # It is actually 448 for whisper
MODEL_ID = args.model_id
BATCH_SIZE = args.batch_size
ACCUMULATION_BATCH = args.accumulation_batch


# Load the model and tokenizer to get the layers after (quantization for lora)
feature_extractor, tokenizer, processor, model = initialize_whisper_objects(
    MODEL_ID  # , quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# VARIABLES Lora
LORA_R = args.lora_r
LORA_ALPHA = args.lora_alpha  # usually 2x LORA_R
LORAPLUS_LR_RATIO = args.loraplus_lr_ratio  # this is for the lora+ optimizer
if args.lora_init not in ["pissa", "lora"]:
    raise ValueError("lora_init must be 'pissa' or 'lora'")
LORA_INIT = "pissa" if args.lora_init == "pissa" else True
LORA_DROPOUT = args.lora_dropout
LAYERS = args.target_modules
if args.target_modules == ["all"]:
    TARGET_MODULES = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
elif args.target_modules == ["decoder"]:
    TARGET_MODULES = []
    trainable_layers = get_specific_layer_names(model)
    for name in trainable_layers:
        if ("model.decoder.layers" in name) and (("proj" in name) or ("fc" in name)):
            TARGET_MODULES.append(name)
elif args.target_modules == ["encoder"]:
    raise ValueError("Encoder modules are not supported for finetuning with LoRA.")
else:
    TARGET_MODULES = args.target_modules

# VARIABLES Training
OPTIMIZER = _str2bool(args.optimizer)
SCHEDULER = args.scheduler
LEARNING_RATE = args.learning_rate
N_EPOCHS = args.n_epochs
if args.run_name is None:
    RUN_NAME = f"Rank:{LORA_R}_Alpha:{LORA_ALPHA}_Optim:{OPTIMIZER}_Sched:{SCHEDULER}_Layers:{','.join(LAYERS)}_Batch:{BATCH_SIZE}"
else:
    RUN_NAME = args.run_name
MODEL_OUT_NAME = os.path.join(args.checkpoint_dir, os.environ["WANDB_PROJECT"], RUN_NAME)

model.config.apply_spec_augment = False
if args.mask_time_prob > 0 or args.mask_feature_prob > 0:
    model.config.apply_spec_augment = True
    # Time Spec Augmentation
    model.config.mask_time_prob = args.mask_time_prob
    model.config.mask_time_length = args.mask_time_length
    model.config.mask_time_min_masks = args.mask_time_min_masks
    # Feature Spec Augmentation
    model.config.mask_feature_prob = args.mask_feature_prob
    model.config.mask_feature_length = args.mask_feature_length
    model.config.mask_feature_min_masks = args.mask_feature_min_masks


# FUNCTIONS
def is_audio_in_length_range(length):
    return length < MAX_INPUT_LENGTH


def is_label_in_length_range(labels):
    """Filter label sequences longer than max length"""
    return len(labels) < MAX_LABEL_LENGTH


# ------------------------------------------------
print("Loading Dataset")
assert args.augmentation_factor > 0, "Augmentation factor must be greater than 0"

if args.dataset == "All":
    train_path = "data/train/preeliminary_no_dupl.csv"
    test_path = "data/test/preeliminary_no_dupl.csv"
elif args.dataset == "Down":
    train_path = "data/train/preeliminary_down_data.csv"
    test_path = "data/test/preeliminary_down_data.csv"
elif args.dataset == "Paralysis":
    train_path = "data/train/preeliminary_paralisis_data.csv"
    test_path = "data/test/preeliminary_paralisis_data.csv"
else:
    raise ValueError("Dataset not recognized. Options: All, Down, Paralisi")

df = DatasetDict()
train_datasets: List = [
    load_dataset(
        "csv",
        data_files=train_path,
        split="train",
        nrows=10,
    )
    for _ in range(args.augmentation_factor)
]
df["train"] = concatenate_datasets(train_datasets)
df["test"] = load_dataset(
    "csv",
    data_files=test_path,
    split="train",
    nrows=10,
)

# Compute the number of steps
LEN_DATASET = df["train"].num_rows
STEPS = LEN_DATASET // max(BATCH_SIZE, ACCUMULATION_BATCH)
MAX_STEPS = STEPS * N_EPOCHS
EVAL_STEPS = STEPS // 5

# PREPARE DATASET
raw_datasets_features_train = list(df["train"].features.keys())
raw_datasets_features_eval = list(df["test"].features.keys())
df = df.cast_column("audio", Audio(sampling_rate=16000))
df["train"] = df["train"].map(prepare_dataset, remove_columns=raw_datasets_features_train)
df["test"] = df["test"].map(prepare_dataset, remove_columns=raw_datasets_features_eval)

df["train"] = df["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)
df["train"] = df["train"].filter(is_label_in_length_range, input_columns=["labels"])
print(df)
print(df["train"])

# LORA
model = prepare_model_for_kbit_training(model)
# whisper is not a TaskType defined in peft_types, instead the target_modules argument is used
config = LoraConfig(
    # task_type="CAUSAL_LM",
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    init_lora_weights=LORA_INIT,
)

adapter_name = "default"
peft_config_dict = {adapter_name: config}

model = get_peft_model(model, config)

# Load reconstruction config (YAML) and inject LoRA-XS initialization
with open("config/reconstruct_config.yaml") as f:
    reconstr_config = yaml.safe_load(f)

reconstr_type = reconstr_config["reconstruction_type"]
reconstr_config[reconstr_type]["rank"] = config.r

# Save for reproducibility
os.makedirs(MODEL_OUT_NAME, exist_ok=True)
with open(os.path.join(MODEL_OUT_NAME, "reconstr_config.json"), "w") as fp:
    json.dump(reconstr_config, fp, indent=2)

# Initialize LoRA-XS weights
find_and_initialize(
    model,
    peft_config_dict,
    adapter_name=adapter_name,
    reconstr_type=reconstr_type,
    reconstruct_config=reconstr_config,
)

model.print_trainable_parameters()

for name, module in model.named_modules():
    if name in TARGET_MODULES:
        for param in module.parameters():
            param.requires_grad = True

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./" + MODEL_OUT_NAME,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    # WARNING on gradients being set to none: Grandient checkpointing seems to be enabled by specifying requires_grad_(True) for the first parameter of the model. However, LoRA training does not train that parameter, so this warning seems to appear.
    gradient_accumulation_steps=int(
        ACCUMULATION_BATCH / BATCH_SIZE
    ),  # increase by 2x for every 2x decrease in batch size
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=SCHEDULER,
    warmup_ratio=0,  # no warmup
    num_train_epochs=N_EPOCHS,  # gets overridden by max_steps
    gradient_checkpointing=False,  # Incompatible
    fp16=True,
    # eval_strategy="steps", # newer
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=EVAL_STEPS,
    save_strategy="steps",
    # save_strategy="best",  # we make a checkpoint when we get a new best validation loss
    save_total_limit=1,  # keep just the best model
    max_steps=MAX_STEPS,  # "train_dataset does not implement __len__, max_steps has to be specified"
    predict_with_generate=True,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    group_by_length=False,  # not available for streaming datasets
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
    run_name=RUN_NAME,
    # Spec Augment
)


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            str(args.output_dir), f"{str(PREFIX_CHECKPOINT_DIR)}-{str(state.global_step)}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

        # return control # newer version
        return


# Optimizer and Scheduler
if OPTIMIZER:
    # AdamW optimizer (like in GLUE LoRA-XS)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

else:
    optimizer = None


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=df["train"],
    eval_dataset=df["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # processing_class=processor.feature_extractor, # tokenizer name change on transformers v5
    tokenizer=tokenizer,
    callbacks=[SavePeftModelCallback],
    optimizers=(optimizer, None),
)

trainer.train()
