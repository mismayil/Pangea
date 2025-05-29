from typing import List, Optional, Dict
import torch
import torch.nn as nn
import re
from PIL import Image

from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedTokenizer
from transformers import Qwen2Config, Qwen2Model, Qwen2ForSequenceClassification, ProcessorMixin, BatchEncoding
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.cache_utils import Cache

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

CHAT_TEMPLATE = """
{%- set nl = '\\n' -%}
{%- set im_start = '<|im_start|>' -%}
{%- set im_end = '<|im_end|>' -%}
{{- im_start -}}system{{- nl -}}{{- system_message -}}{{- im_end -}}{{- nl -}}
{%- for message in messages -%}
    {%- if message.role == 'user' -%}
        {{- im_start -}}user{{- nl -}}
        {%- if message.content is not none -%}
            {%- set parts = message.content.split('<image>') -%}
            {%- for part in parts -%}
                {{- part -}}
                {%- if not loop.last -%}<image>{{- nl -}}{%- endif -%}
            {%- endfor -%}
        {%- endif -%}
        {{- im_end -}}{{- nl -}}
    {%- elif message.role == 'assistant' -%}
        {{- im_start -}}assistant{{- nl -}}
        {{- message.content if message.content is not none else '' -}}
        {{- im_end -}}{{- nl -}}
    {%- endif -%}
{%- endfor -%}
"""

class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)

class LlavaQwenForSequenceClassification(Qwen2ForSequenceClassification, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        Qwen2ForSequenceClassification.__init__(self, config)
        self.num_labels = config.num_labels
        self.model = LlavaQwenModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
    ) -> SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, pixel_values, modalities, image_sizes)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

class LlavaQwenProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    chat_template = CHAT_TEMPLATE
    # image_processor_class = "LayoutLMv3ImageProcessor"
    # tokenizer_class = ("LayoutLMv3Tokenizer", "LayoutLMv3TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        super().__init__(image_processor, tokenizer)

    def apply_chat_template(self, conversation, chat_template=None, **kwargs):
        return self.tokenizer.apply_chat_template(
            conversation,
            chat_template=chat_template or self.chat_template,
            tokenize=False,
            add_generation_prompt=False,
            **kwargs
        )

    def __call__(
        self,
        text,
        images,
        **kwargs,
    ) -> BatchEncoding:
        image_tensors = []

        for image in images:
            image = Image.open(image)
            image_tensors.append(image)

        pixel_values = self.image_processor.preprocess(image_tensors, return_tensors='pt')['pixel_values']

        encoded_inputs = self.tokenizer(text)
        image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        encoded_inputs["input_ids"][encoded_inputs["input_ids"].index(image_token_id)] = IMAGE_TOKEN_INDEX
        encoded_inputs["pixel_values"] = pixel_values
        return encoded_inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "pixel_values"]

    
AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForSequenceClassification.register(LlavaQwenConfig, LlavaQwenForSequenceClassification)