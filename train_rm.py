# adapted from https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py
import warnings
from dotenv import load_dotenv
import os
import torch.functional as F
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from functools import partial

load_dotenv()

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    EvalPrediction,
    Trainer
)

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

from utils import find_latest_checkpoint, H4ArgumentParser
from data import get_prompt_and_responses_from_trl_sample


@dataclass
class RewardScriptArguments(ScriptArguments):
    add_margin_loss: Optional[bool] = field(
        default=False, metadata={"help": "Add margin loss to the reward model."}
    )
    add_length_bias_loss: Optional[bool] = field(
        default=False, metadata={"help": "Add length bias loss to the reward model."}
    )


@dataclass
class CustomRewardConfig(RewardConfig):
    length_bias_coefficient: Optional[float] = field(
        default=1.0, metadata={"help": "Coefficient for the length bias loss."}
    )


class CustomRewardTrainer(RewardTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        )["logits"]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        )["logits"]
        # calculate loss, optionally modulate with margin

        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(
                rewards_chosen - rewards_rejected - inputs["margin"]
            )
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)

        if "length_bias" in inputs:
            loss += self.args.length_bias_coefficient * inputs["length_bias"]

        loss = loss.mean()

        if self.args.center_rewards_coefficient is not None:
            loss += self.args.center_rewards_coefficient * torch.mean(
                (rewards_chosen + rewards_rejected) ** 2
            )

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss

    def evaluate(self, *args, **kwargs):
        num_print_samples = kwargs.pop("num_print_samples", 4)
        # self.visualize_samples(num_print_samples)
        return Trainer.evaluate(self, *args, **kwargs)

def add_margin(row):
    # Assume you have a score_chosen and score_rejected columns that you want to use to compute the margin
    return {"margin": row["score_chosen"] - row["score_rejected"]}


def add_length_bias(row, max_length_bias):
    _, chosen_response, rejected_response = get_prompt_and_responses_from_trl_sample(
        row
    )
    return {
        "length_bias": max(
            0, (len(chosen_response.split()) - len(rejected_response.split()))
        )
        / max_length_bias
    }


def compute_accuracy(eval_pred: EvalPrediction) -> dict[str, float]:
    predictions, labels = eval_pred

    if predictions.ndim == 3:
        # Token classification task. Shapes are (batch_size, seq_len, num_labels) and (batch_size, seq_len)
        # Used to compute the accuracy in the prm_trainer.
        predictions = np.argmax(predictions, axis=2)

        # Flatten the predictions and labels to remove the ignored tokens.
        predictions = np.array(
            [
                p
                for prediction, label in zip(predictions, labels)
                for (p, lbl) in zip(prediction, label)
                if lbl != -100
            ]
        )
        labels = np.array([lbl for label in labels for lbl in label if lbl != -100])

    else:
        # Here, predictions is rewards_chosen and rewards_rejected. Shapes are (batch_size, 2) and (batch_size,)
        # We want to see how much of the time rewards_chosen > rewards_rejected.
        equal_mask = predictions[:, 0] == predictions[:, 1]
        equal_predictions_count = int(equal_mask.sum())

        if equal_predictions_count > 0:
            warnings.warn(
                f"There are {equal_predictions_count} out of {len(predictions[:, 0])} instances where the predictions "
                "for both options are equal. These instances are ignored in the accuracy computation.",
                UserWarning,
            )

        # Filter out equal predictions
        predictions = predictions[~equal_mask]
        labels = labels[~equal_mask]

        mean_rewards_chosen = predictions[:, 0].mean()
        mean_rewards_rejected = predictions[:, 1].mean()

        # Use the remaining predictions for accuracy calculation
        predictions = np.argmax(predictions, axis=1)

    accuracy = np.array(predictions == labels, dtype=float).mean().item()
    metrics = {
        "accuracy": accuracy,
        "equal_predictions_count": equal_predictions_count,
        "mean_rewards_chosen": mean_rewards_chosen,
        "mean_rewards_rejected": mean_rewards_rejected,
    }

    return metrics


if __name__ == "__main__":
    parser = H4ArgumentParser((RewardScriptArguments, CustomRewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # Find the latest checkpoint if available
    checkpoint_dir = training_args.output_dir
    if training_args.resume_from_checkpoint:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            training_args.resume_from_checkpoint = latest_checkpoint
            print(f"Resuming from checkpoint: {latest_checkpoint}")
        else:
            print(f"No checkpoint found in {checkpoint_dir}. Starting from scratch.")
            training_args.resume_from_checkpoint = None

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
        attn_implementation="eager" if "gemma" in model_args.model_name_or_path.lower() else model_args.attn_implementation
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    # Align padding tokens between tokenizer and model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_args.use_peft and model_args.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.",
            UserWarning,
        )

    ##############
    # Load dataset
    ##############
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    if script_args.add_margin_loss:
        dataset = dataset.map(add_margin)

    if script_args.add_length_bias_loss:
        length_biases = []
        for sample in dataset[script_args.dataset_train_split]:
            prompt, chosen_response, rejected_response = (
                get_prompt_and_responses_from_trl_sample(sample)
            )
            length_biases.append(
                len(chosen_response.split()) - len(rejected_response.split())
            )
        max_length_bias = max(length_biases)
        dataset = dataset.map(partial(add_length_bias, max_length_bias=max_length_bias))

    ##########
    # Training
    ##########
    trainer = CustomRewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split].shuffle(
            seed=training_args.seed
        ),
        eval_dataset=(
            dataset[script_args.dataset_test_split].shuffle(seed=training_args.seed)
            if training_args.eval_strategy != "no"
            else None
        ),
        peft_config=get_peft_config(model_args),
        compute_metrics=compute_accuracy,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Push to hub
    if training_args.push_to_hub:
        trainer.push_to_hub()
