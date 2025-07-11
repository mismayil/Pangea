# adapted from https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py
import warnings
from dotenv import load_dotenv
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any, Union
from accelerate import PartialState
import torch.nn as nn

load_dotenv()

import torch
from datasets import load_dataset
from transformers import (
    EvalPrediction,
    ProcessorMixin,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config
)

from utils import find_latest_checkpoint, H4ArgumentParser
from trl.data_utils import maybe_apply_chat_template
from modeling import load_pretrained_cls_model

@dataclass
class RewardScriptArguments(ScriptArguments):
    add_margin_loss: Optional[bool] = field(
        default=False, metadata={"help": "Add margin loss to the reward model."}
    )


def add_margin(row):
    # Assume you have a score_chosen and score_rejected columns that you want to use to compute the margin
    return {"margin": row["score_chosen"] - row["score_rejected"]}

def _process(batch: dict[str, list[Any]], tokenizer: "ProcessorMixin") -> dict[str, list[Any]]:
    """Tokenize a batch from a reward modelling dataset."""
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
        "pixel_values": [],
    }
    for chosen, rejected, image in zip(batch["chosen"], batch["rejected"], batch["image"]):
        tokenized_chosen = tokenizer(chosen, [image])
        tokenized_rejected = tokenizer(rejected, [image])
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        new_examples["pixel_values"].append(tokenized_chosen["pixel_values"])  # Assuming pixel_values are the same for both chosen and rejected

    return new_examples

def process_dataset(dataset, processor, max_length=1024, dataset_num_proc=1):
    with PartialState().main_process_first():
        fn_kwargs = {"tokenizer": processor}
        dataset = dataset.map(maybe_apply_chat_template, fn_kwargs={"tokenizer": processor})
        dataset = dataset.map(
            _process,
            batched=True,
            fn_kwargs=fn_kwargs,
            num_proc=dataset_num_proc,
        )
        # This filter is important because otherwise you get samples that exceed the model's context length and
        # get truncated => noisy signal the chosen/rejected label gets lost. The downside is that the
        # user might get surprised if N samples are missing from training.
        dataset = dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= max_length and len(x["input_ids_rejected"]) <= max_length,
            num_proc=dataset_num_proc,
        )
    return dataset


@dataclass
class MultimodalRewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        pad_to_multiple_of (`int` or `None`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features_chosen = []
        features_rejected = []
        margin = []
        # check if we have a margin. If we do, we need to batch it as well
        has_margin = "margin" in features[0]
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                    "pixel_values": feature.get("pixel_values", None),  # Assuming pixel_values is optional
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                    "pixel_values": feature.get("pixel_values", None),  # Assuming pixel_values is optional
                }
            )
            if has_margin:
                margin.append(feature["margin"])
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "pixel_values": batch_chosen["pixel_values"],  # Assuming pixel_values are the same for both chosen and rejected
            "return_loss": True,
            "modalities": ["image"] * len(batch_chosen["input_ids"]),
        }
        if has_margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch["margin"] = margin
        return batch

class MultimodalRewardTrainer(RewardTrainer):
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            pixel_values=inputs.get("pixel_values", None),  # Assuming pixel_values is optional
            modalities=inputs.get("modalities", None),  # Assuming modalities is optional
            return_dict=True,
        )["logits"]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            pixel_values=inputs.get("pixel_values", None),  # Assuming pixel_values is optional
            modalities=inputs.get("modalities", None),  # Assuming modalities is optional
            return_dict=True,
        )["logits"]
        # calculate loss, optionally modulate with margin
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if self.args.center_rewards_coefficient is not None:
            loss += self.args.center_rewards_coefficient * torch.mean((rewards_chosen + rewards_rejected) ** 2)

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss

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
    parser = H4ArgumentParser((RewardScriptArguments, RewardConfig, ModelConfig))
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

    ## Load model and processor
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
        attn_implementation="eager" if "gemma" in model_args.model_name_or_path.lower() else model_args.attn_implementation,
        num_labels=1,  # For reward models, we typically have a single output for the score
    )
    model, processor = load_pretrained_cls_model(model_args.model_name_or_path, multimodal=True, **model_kwargs)

    if model_args.use_peft and model_args.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.",
            UserWarning,
        )

    ## Load dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    if script_args.add_margin_loss:
        dataset = dataset.map(add_margin)

    train_dataset = process_dataset(
        dataset[script_args.dataset_train_split],
        processor=processor,
        max_length=training_args.max_length,
        dataset_num_proc=training_args.dataset_num_proc,
    )
    eval_dataset = (
        process_dataset(
            dataset[script_args.dataset_test_split],
            processor=processor,
            max_length=training_args.max_length,
            dataset_num_proc=training_args.dataset_num_proc,
        )
        if training_args.eval_strategy != "no"
        else None
    )

    ## Training
    trainer = MultimodalRewardTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset.shuffle(
            seed=training_args.seed
        ),
        eval_dataset=(
            eval_dataset.shuffle(seed=training_args.seed)
            if eval_dataset is not None
            else None
        ),
        peft_config=get_peft_config(model_args),
        compute_metrics=compute_accuracy,
        data_collator=MultimodalRewardDataCollatorWithPadding(
            tokenizer=processor.tokenizer,
        ),
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_model(training_args.output_dir)

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Push to hub
    if training_args.push_to_hub:
        trainer.push_to_hub()
