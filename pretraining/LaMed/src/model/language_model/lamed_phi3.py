from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, Phi3Config, Phi3Model, Phi3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..lamed_arch import LamedMetaModel, LamedMetaForCausalLM

from torch.distributed import is_initialized, get_rank


class LamedPhi3Config(Phi3Config):
    model_type = "lamed_phi3"


class LamedPhi3Model(LamedMetaModel, Phi3Model):
    config_class = LamedPhi3Config
    def __init__(self, config: Phi3Config):
        super(LamedPhi3Model, self).__init__(config)


class LamedPhi3ForCausalLM(LamedMetaForCausalLM, Phi3ForCausalLM):
    config_class = LamedPhi3Config

    def __init__(self, config):
        super(LamedPhi3ForCausalLM, self).__init__(config)
        self.model = LamedPhi3Model(config)
        self.score_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.radgraph_enable = False
        self.output_score = False
        self.alpha = 10
        self.count = 0

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            images: Optional[torch.FloatTensor] = None,
            input_ids: torch.LongTensor = None,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            segs: Optional[torch.FloatTensor] = None,
            answer: Optional[list] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        input_ids_pre = input_ids

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )
        if torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any():
            print("Input contains NaN/Inf!")
        try:
            seg_ids = torch.nonzero(torch.sum(segs, dim=(1, 2, 3, 4))).flatten().tolist()
        except:
            seg_ids = []
        
        # segmentation
        if self.get_model().seg_enable and seg_ids:
            outputs = super().forward(
                                    input_ids=input_ids,
                                    inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                    output_hidden_states=True,
                                    position_ids=position_ids,
                                    past_key_values=past_key_values,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    return_dict=return_dict
                                )

            output_hidden_states = outputs.hidden_states

            last_hidden_state = output_hidden_states[-1]

            seg_token_mask = input_ids_pre[:, 1:] == self.config.seg_token_id
            seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1), dtype=seg_token_mask.dtype).cuda(),
                ],
                dim=1,
            )

            seg_prompts = []
            for i in seg_ids:
                if torch.sum(seg_token_mask[i]) == 1:
                    seg_token = last_hidden_state[i][seg_token_mask[i]]
                    seg_prompt = self.get_model().seg_projector(seg_token)
                elif torch.sum(seg_token_mask[i]) > 1:
                    seg_tokens = last_hidden_state[i][seg_token_mask[i]]
                    seg_token = torch.mean(seg_tokens, dim=0, keepdim=True)
                    seg_prompt = self.get_model().seg_projector(seg_token)
                else:
                    seg_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype,
                                             device=last_hidden_state.device)
                seg_prompts.append(seg_prompt)

            seg_prompts = torch.cat(seg_prompts, dim=0)
            logits = self.get_model().seg_module(images[seg_ids], text_emb=seg_prompts)
            loss_dice = self.get_model().dice_loss(logits, segs[seg_ids])
            loss_bce = self.get_model().bce_loss(logits, segs[seg_ids])
            seg_loss = loss_dice + loss_bce
            outputs.loss = outputs.loss + seg_loss
            return outputs
        # radgraph loss
        elif self.radgraph_enable:
            outputs = super().forward(
                                input_ids=input_ids,
                                inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                labels=labels,
                                output_hidden_states=output_hidden_states,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                use_cache=use_cache,
                                output_attentions=output_attentions,
                                return_dict=return_dict
                            )
            logits = outputs.logits
            predicted_token_ids = logits.argmax(dim=-1)
            # print(predicted_token_ids)
            # torch.set_printoptions(profile="full")
            # for label in labels:
            #     print(label)
            # exit()
            # TODO: check if the predicted_token_ids are correct
            labels_cp = labels.clone()
            labels_cp[labels == -100] = self.tokenizer.pad_token_id
            refs = self.tokenizer.batch_decode(labels_cp, skip_special_tokens=True)
            predicted_token_ids_cp = predicted_token_ids.clone()
            predicted_token_ids_cp[labels == -100] = self.tokenizer.pad_token_id
            hyps = self.tokenizer.batch_decode(predicted_token_ids_cp, skip_special_tokens=True)
            # print(type(hyps[0]))
            # print(type(refs[0]))
            # exit()
            # print(hyps[0].device)
            # print(refs[0].device)
            # exit()
            mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = self.f1radgraph(hyps=hyps, refs=refs)
            # if get_rank() == 0:
            #     print(f"mean_reward: {mean_reward}")
            #     print(f"reward_list: {reward_list}")
            #     print(f"hypothesis_annotation_lists: {hypothesis_annotation_lists}")
            #     print(f"outputs.loss: {outputs.loss}")
            # exit()
            
            radgraph_loss = 1.0 - np.mean(mean_reward)
            radgraph_loss = self.alpha * radgraph_loss
            
            if self.count % 160 == 0 and get_rank() == 0:
                print(f"text loss: {outputs.loss}, radgraph_loss: {radgraph_loss}")
                print(hyps)
            self.count += 1
            outputs.loss += radgraph_loss
            
            # print(type(outputs.logits))
            # exit()
            return outputs
        
        # output score
        elif self.output_score:
            outputs = super().forward(
                                input_ids=input_ids,
                                inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                labels=labels,
                                output_hidden_states=output_hidden_states,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                use_cache=use_cache,
                                output_attentions=output_attentions,
                                return_dict=return_dict
                            )
            hidden_state = outputs.hidden_states[-1]
            score = self.score_layer(hidden_state)
            label = nn.functional.one_hot(labels, num_classes=11)
            cross_entropy = nn.functional.cross_entropy(score, label)
            outputs.loss = cross_entropy
            return outputs
        else:
            # print(f"output_attentions: {output_attentions}")
            # print(input_ids.dtype)
            # print(inputs_embeds.dtype)
            # exit()
            # if (attention_mask.sum(dim=-1) == 0).any():
            #     print("Warning: Some rows in the attention mask are entirely masked.")
            #     exit()
            # print(f"input max: {inputs_embeds.max()}, input min: {inputs_embeds.min()}")
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

    @torch.no_grad()
    def add_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        seg_enable: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor, Any]:
        # print(111)
        # exit()
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        is_eval = kwargs.pop("is_eval", False)
        patch_id = kwargs.pop("patch_id", None)
        # tokenizer = kwargs.pop("tokenizer", None)
        # patch_id = tokenizer.convert_tokens_to_ids('<im_patch>')
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                is_eval,
                patch_id
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        # print(inputs_embeds)
        # exit()
        

        if seg_enable:
            outputs = super().generate(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs
            )

            output_hidden_states = outputs.hidden_states
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.config.seg_token_id

            last_tensors = [tuple[-1] for tuple in output_hidden_states]
            last_hidden_state = torch.cat(last_tensors[1:], dim=1)

            seg_prompts = []
            noseg_ids = []
            for i in range(len(seg_token_mask)):
                if torch.sum(seg_token_mask[i]) == 1:
                    seg_token = last_hidden_state[i][seg_token_mask[i]]
                    seg_prompt = self.get_model().seg_projector(seg_token)
                elif torch.sum(seg_token_mask[i]) > 1:
                    seg_tokens = last_hidden_state[i][seg_token_mask[i]]
                    seg_token = torch.mean(seg_tokens, dim=0, keepdim=True)
                    seg_prompt = self.get_model().seg_projector(seg_token)
                else:
                    noseg_ids.append(i)
                    seg_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype,
                                             device=last_hidden_state.device)
                seg_prompts.append(seg_prompt)

            seg_prompts = torch.cat(seg_prompts, dim=0)
            logits = self.get_model().seg_module(images, seg_prompts)
            logits[noseg_ids] = -torch.inf

            return output_ids, logits
        else:
            output_ids = super().generate(
                inputs_embeds=inputs_embeds,
                **kwargs
            )
            return output_ids


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        return inputs


AutoConfig.register("lamed_phi3", LamedPhi3Config)
AutoModelForCausalLM.register(LamedPhi3Config, LamedPhi3ForCausalLM)