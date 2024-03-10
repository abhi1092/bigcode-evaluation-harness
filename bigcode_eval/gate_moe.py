import logging
import os.path
from torch.nn import CrossEntropyLoss
import torch
from typing import Optional, List, Union, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
logger = logging.getLogger(__name__)
from transformers.modeling_utils import GenerationMixin


class TwoModels(GenerationMixin, torch.nn.Module):
    def __init__(self, model_a, model_b,
             ):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def prepare_inputs_for_generation(
            self, *args, **kwargs
    ):
        self.model_a.prepare_inputs_for_generation(*args, **kwargs)
        return self.model_b.prepare_inputs_for_generation(*args, **kwargs)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        logits_a = self.model_a(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, None, use_cache,
                            output_attentions, output_hidden_states, return_dict, cache_position)
        input_ids_b, attention_mask_b = input_ids.detach().clone().to(self.device), attention_mask.detach().clone().to(self.device)

        logits_b = self.model_b(input_ids_b, attention_mask_b, position_ids, past_key_values, inputs_embeds, None,
                                use_cache, output_attentions, output_hidden_states, return_dict, cache_position)
        logits = (logits_a.logits + logits_b.logits)/ 2

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            loss = None
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=logits_a.past_key_values,
            hidden_states=logits_a.hidden_states,
            attentions=logits_a.attentions,
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.model_a.gradient_checkpointing_enable(**kwargs)
        self.model_b.gradient_checkpointing_enable(**kwargs)


    @property
    def config(self):
        return self.model_b.config

    @property
    def device(self):
        return self.model_a.device

    @property
    def generation_config(self):
        return self.model_b.generation_config

    @property
    def main_input_name(self):
        return self.model_b.main_input_name

    def save_pretrained(self, save_path):
        model_a_path = os.path.join(save_path, "model_a")
        model_b_path = os.path.join(save_path, "model_b")
        self.model_a.save_pretrained(model_a_path)
        self.model_b.save_pretrained(model_b_path)

    def can_generate(self):
        return True

class TwoModelGate(TwoModels):
    def __init__(self, model_a, model_b):
        super().__init__(model_a, model_b)
        self.gate = torch.nn.Linear(self.model_a.config.hidden_size, 2, dtype=torch.bfloat16,  device=model_a.device, bias=False)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        logits_a = self.model_a(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, None,
                                use_cache,
                                output_attentions, output_hidden_states, return_dict, cache_position)
        input_ids_b, attention_mask_b = input_ids.detach().clone().to(self.device), attention_mask.detach().clone().to(
            self.device)

        logits_b = self.model_b(input_ids_b, attention_mask_b, position_ids, past_key_values, inputs_embeds, None,
                                use_cache, output_attentions, output_hidden_states, return_dict, cache_position)

        # router_weights: [1, 432, 2]
        # logits_a/b: [1, 432, 32016]
        router_weights = self.gate(logits_a.inputs_embeds)
        if labels is not None:
            # router_weights = torch.clamp(router_weights, min=0, max=1)
            # router_weights = router_weights + 1e-6
            #
            # # Normalize
            # router_weights = router_weights / router_weights.sum(dim=2, keepdim=True)
            #
            # # Expand router weights for broadcasting
            # router_weights = router_weights[:, :, :, None].expand(-1, -1, -1, 32016)
            #
            # # Route logits
            # routed_logits_a = router_weights[:, :, 0] * logits_a.logits
            # routed_logits_b = router_weights[:, :, 1] * logits_b.logits
            #
            # logits = routed_logits_a + routed_logits_b
            # max_index = torch.argmax(router_weights, dim=2)
            max_val, max_index = torch.max(router_weights, axis=2)

            # Copy the value in vocab dimension
            max_index_expanded = max_index.unsqueeze(-1).expand(-1, -1, logits_a.logits.shape[-1])

            max_val = max_val.unsqueeze(-1).expand(-1, -1, logits_a.logits.shape[-1])
            logits_ = torch.where(max_index_expanded == 0, logits_a.logits, logits_b.logits)
            # max_index_expanded_normalized = torch.where(max_index_expanded != 0, max_index_expanded / max_index_expanded, max_index_expanded)
            # import IPython; IPython.embed()
            # exit()
            logits = logits_ * max_val
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            router_weights = self.gate(logits_a.inputs_embeds)
            max_index = torch.argmax(router_weights, dim=2)
            max_index_expanded = max_index.unsqueeze(-1).expand(-1, -1, logits_a.logits.shape[-1])
            logits = torch.where(max_index_expanded == 0, logits_a.logits, logits_b.logits)
            loss = None
        # print(self.gate.weight)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=logits_a.past_key_values,
            hidden_states=logits_a.hidden_states,
            attentions=logits_a.attentions,
        )

    def save_pretrained(self, save_path):
        model_a_path = os.path.join(save_path, "model_a")
        model_b_path = os.path.join(save_path, "model_b")
        torch.save(self.gate.state_dict(), os.path.join(save_path, "gate.pt"))
        self.model_a.save_pretrained(model_a_path)
        # self.tokenizer_a.save_pretrained(save_path)
        self.model_b.save_pretrained(model_b_path)

    def load_gate_from_pt(self, load_path):
        print("Loading gate weights")
        self.gate.load_state_dict(torch.load(os.path.join(load_path, 'gate.pt')))

    def save_gate_to_pt(self, save_path):
        torch.save(self.gate.state_dict(), os.path.join(save_path, "gate.pt"))
