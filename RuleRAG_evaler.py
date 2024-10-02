import re

from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import numpy as np
import copy
import inspect
import warnings
from dataclasses import dataclass

import torch
from torch import nn
import torch.distributed as dist
# from vllm import LLM, SamplingParams
# from vllm.lora.request import LoRARequest
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import logging
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import (
    GreedySearchEncoderDecoderOutput,
    GreedySearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchDecoderOnlyOutput,
)
from transformers.generation.logits_process import LogitsProcessorList
from tqdm import tqdm
import time
from data_utils.basic import read_txt_as_list, read_json
from RuleRAG_eval_utils import read_results, read_num_and_li_results
from collections import defaultdict, Counter
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from neox import init_neox, text_generation
from basic import blockPrinting
import os


Role = Literal["system", "user", "assistant"]

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)
m_F1=[]

class Evaler:
    def __init__(self, topk, tests, test_ans,
                 eval_txt_path, args,
                 model=None, tokenizer=None, patterns=None,
                 early_stop_chars=None, obligations=[]):

        self.llama = 1
        self.model = model
        self.tokenizer = tokenizer
        self.patterns = patterns
        self.tests = tests
        self.test_ans = test_ans
        self.eval_txt_path = eval_txt_path
        self.topk = topk

        self.args = args

        self.obligations = obligations
        self.constraints = []
        self.zone_zero = early_stop_chars

        self.first_check = 0
        self.top = 1
        f_entity2id = open(
            '/home/',
            'r')
        entity_json = f_entity2id.read()
        self.entity_set = json.loads(entity_json)

    def restrict_list_hard(self, tokens, prev_pos, min_prompt_len, input_text_mask, eos_reached, m=0):
        logits = self.model.forward(tokens[:, prev_pos:min_prompt_len], prev_pos)
        logits_last = logits[:, -1]
        top_10_indices = torch.topk(logits_last, k=logits.shape[-1], dim=-1).indices
        values_to_extract = [29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929]
        top_10_indices_np = top_10_indices.cpu().numpy()
        mask = np.isin(top_10_indices_np, values_to_extract)
        extracted_elements = top_10_indices_np[mask][:10]
        # Convert back to Tensor type
        top_10_indices = torch.tensor(extracted_elements)

        next_token = top_10_indices[m]
        next_token = next_token.reshape(-1)

        next_token = torch.where(
            input_text_mask[:, min_prompt_len], tokens[:, min_prompt_len], next_token
        )

        tokens[:, min_prompt_len] = next_token
        eos_reached |= (~input_text_mask[:, min_prompt_len]) & (
                next_token == self.tokenizer.eos_id
        )

        self.first_check = 1  # skip first_check
        return next_token, eos_reached

    def first_checking(self, next_tokens, next_tokens_scores):
        this_peer_finished = False
        if self.first_check == 0:  # first check
            if self.obligations and (next_tokens not in self.obligations):
                this_peer_finished = True
                self.first_check = -1  # not begin with nums

            if self.constraints and (next_tokens in self.constraints):
                self.top += 1
                next_tokens = torch.argsort(next_tokens_scores, dim=-1, descending=True)[:, self.top - 1]
                self.constraints.append(next_tokens)
                self.first_check = -1  # breach of obligs
            else:
                self.constraints.append(next_tokens)
                self.first_check = 1  # check sign passed
        return this_peer_finished, next_tokens

    def gen_set_ans(self, tests='', dir_full_test='', dir_time2id=''):

        if tests == '':
            tests = self.tests
        dict_qu_ans = {}
        if dir_full_test == '':
            full_test_ans = self.test_ans
            for i in tqdm(range(0, len(tests) - 1)):
                try:
                    query = tests[i].split('Question')[3]
                except:
                    query = tests[i].split('Question')[2]
                if query == '':
                    break
                if dict_qu_ans.get(query) == None:
                    dict_qu_ans[query] = set()
                dict_qu_ans[query].add(full_test_ans[i])  # add answers to the set
                # time.sleep(0.001)
        else:
            dict_t2id = {}
            if dir_time2id != '':
                dict_t2id = read_json(dir_time2id)
            else:
                print("Attention: icews18 needs its ts2id file to convert time into time_id")
            fulltest = read_txt_as_list(dir_full_test)  # only load essentially
            li_queries = [test.split('\n')[-1] for test in tests]
            # build sets
            for i in range(0, len(li_queries) - 1):
                query = li_queries[i]
                if query == '':
                    break
                if dict_qu_ans.get(query) is None:
                    dict_qu_ans[query] = set()
            end_time = li_queries[-3].split(':')[0]
            for line in fulltest:
                quadruple = line.strip().split('\t')
                time_quadruple = dict_t2id[quadruple[3]] if dir_time2id != '' else quadruple[3]
                if int(time_quadruple) > int(end_time):
                    break
                built_query = f"{time_quadruple}: [{quadruple[0]}, {quadruple[1]},"
                if dict_qu_ans.get(built_query) is not None:
                    dict_qu_ans[built_query].add(quadruple[2])  # add answers to the set
            print("duplicate answers checked")
        return dict_qu_ans

    def generate_extra_answers(self, m_inloop, k_inloop):
        if self.args.ft == 1:
            raw_answers, answer_regs = self.model_calling(m_inloop)  # call for more generated ans
        elif self.llama == 1:  # icl llama2
            answer_regs = self.text_completion(m_inloop,
                                               str(self.args.PROMPT),
                                               max_gen_len=self.args.max_gen_len,
                                               temperature=self.args.TEMPERATURE,
                                               # top_p=top_p,
                                               )
            answer_regs = [answer_reg['generation'] for answer_reg in answer_regs]
            raw_answers = answer_regs
        else:  # icl gpt neox
            raw_answers = text_generation(m_inloop, k_inloop, self.model, self.tokenizer,
                                          str(self.args.PROMPT),
                                          # icews14 28, icews18 34, ecola 18, GDELT 16, YAGO 25.
                                          max_seq_len=34,
                                          verbose=False)
            pattern = re.compile(r'\s*(\d+)\.(.*?)\]')
            answer_regs = re.match(pattern, raw_answers).group(2).strip() \
                if re.match(pattern, raw_answers) else raw_answers
            answer_regs = [answer_regs]
        return raw_answers, answer_regs

    def my_generate_top10(self, model_instance, m, gen_length, **kwargs):
        # base_model = model_instance.base_model
        base_model = model_instance

        # original prepare_inputs_for_generation and generation_config
        original_prepare_inputs_for_generation = base_model.prepare_inputs_for_generation
        original_generation_config = getattr(base_model, "generation_config", None)

        # prepare_inputs_for_generation and generation_config
        base_model.prepare_inputs_for_generation = model_instance.prepare_inputs_for_generation
        if hasattr(base_model, "model"):
            base_model.model.generation_config = model_instance.generation_config
        else:
            base_model.generation_config = model_instance.generation_config

        try:
            # base_model generate_top10
            outputs = self.my_utils_generate_top10(base_model, m, gen_length, **kwargs)
        except Exception as e:
            # prepare_inputs_for_generation
            base_model.prepare_inputs_for_generation = original_prepare_inputs_for_generation
            if original_generation_config is not None:
                base_model.generation_config = original_generation_config
            raise e
        else:
            base_model.prepare_inputs_for_generation = original_prepare_inputs_for_generation
            # recover generation_config
            if original_generation_config is not None:
                base_model.generation_config = original_generation_config
            return outputs

    @torch.no_grad()
    def my_utils_generate_top10(self,
                                model_instance, m,
                                gen_length,
                                inputs: Optional[torch.Tensor] = None,
                                generation_config: Optional[GenerationConfig] = None,
                                logits_processor: Optional[LogitsProcessorList] = None,
                                stopping_criteria: Optional[StoppingCriteriaList] = None,
                                # max_length=max_length,
                                prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
                                synced_gpus: Optional[bool] = None,
                                assistant_model: Optional["PreTrainedModel"] = None,
                                streamer: Optional["BaseStreamer"] = None,
                                **kwargs,
                                ):

        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        if generation_config is None:
            if model_instance.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(model_instance.config)
                if new_generation_config != model_instance.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use a generation configuration file (see"
                        " https://huggingface.co/docs/transformers/main_classes/text_generation)"
                    )
                    model_instance.generation_config = new_generation_config
            generation_config = model_instance.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        model_instance._validate_model_kwargs(model_kwargs.copy())
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        inputs_tensor, model_input_name, model_kwargs = model_instance._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(model_instance.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = model_instance._prepare_attention_mask_for_generation(
                inputs_tensor, inputs_tensor, inputs_tensor
            )

        if not model_instance.config.is_encoder_decoder:
            if (
                    generation_config.pad_token_id is not None
                    and len(inputs_tensor.shape) == 2
                    and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if model_instance.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            model_kwargs = model_instance._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        if model_instance.config.is_encoder_decoder:
            input_ids, model_kwargs = model_instance._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if model_instance.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        is_constraint_gen_mode = (
                generation_config.constraints is not None or generation_config.force_words_ids is not None
        )

        is_contrastive_search_gen_mode = (
                (generation_config.num_beams == 1)
                and generation_config.top_k is not None
                and generation_config.top_k > 1
                and generation_config.do_sample is False
                and generation_config.penalty_alpha is not None
                and generation_config.penalty_alpha > 0
        )

        is_greedy_gen_mode = (
                (generation_config.num_beams == 1)
                and (generation_config.num_beam_groups == 1)
                and generation_config.do_sample is False
                and not is_constraint_gen_mode
                and not is_contrastive_search_gen_mode
        )
        is_sample_gen_mode = (
                (generation_config.num_beams == 1)
                and (generation_config.num_beam_groups == 1)
                and generation_config.do_sample is True
                and not is_constraint_gen_mode
                and not is_contrastive_search_gen_mode
        )
        is_beam_gen_mode = (
                (generation_config.num_beams > 1)
                and (generation_config.num_beam_groups == 1)
                and generation_config.do_sample is False
                and not is_constraint_gen_mode
                and not is_contrastive_search_gen_mode
        )
        is_beam_sample_gen_mode = (
                (generation_config.num_beams > 1)
                and (generation_config.num_beam_groups == 1)
                and generation_config.do_sample is True
                and not is_constraint_gen_mode
                and not is_contrastive_search_gen_mode
        )
        is_group_beam_gen_mode = (
                (generation_config.num_beams > 1)
                and (generation_config.num_beam_groups > 1)
                and not is_constraint_gen_mode
                and not is_contrastive_search_gen_mode
        )
        is_assisted_gen_mode = False
        if assistant_model is not None:
            if not (is_greedy_gen_mode or is_sample_gen_mode):
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )
            is_assisted_gen_mode = True

        if generation_config.num_beam_groups > generation_config.num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and generation_config.do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if model_instance.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {model_instance.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{model_instance.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        logits_processor = model_instance._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = model_instance._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        if is_assisted_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")

            if assistant_model.config.is_encoder_decoder:
                assistant_model_kwargs = copy.deepcopy(model_kwargs)
                inputs_tensor, model_input_name, assistant_model_kwargs = assistant_model._prepare_model_inputs(
                    inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_model_kwargs
                )
                assistant_model_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, assistant_model_kwargs, model_input_name
                )
                model_kwargs["assistant_encoder_outputs"] = assistant_model_kwargs["encoder_outputs"]

            return model_instance.assisted_decoding(
                input_ids,
                assistant_model=assistant_model,
                do_sample=generation_config.do_sample,
                logits_processor=logits_processor,
                logits_warper=model_instance._get_logits_warper(
                    generation_config) if generation_config.do_sample else None,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing greedy search, "
                    f"but is {generation_config.num_return_sequences}."
                )
            return self.my_utils_greedy_search_top10(model_instance,
                                                     m,
                                                     gen_length,
                                                     input_ids,
                                                     logits_processor=logits_processor,
                                                     stopping_criteria=stopping_criteria,
                                                     pad_token_id=generation_config.pad_token_id,
                                                     eos_token_id=generation_config.eos_token_id,
                                                     output_scores=generation_config.output_scores,
                                                     return_dict_in_generate=generation_config.return_dict_in_generate,
                                                     synced_gpus=synced_gpus,
                                                     streamer=streamer,
                                                     **model_kwargs,
                                                     )

        elif is_contrastive_search_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing contrastive search, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")

            return model_instance.contrastive_search(
                input_ids,
                top_k=generation_config.top_k,
                penalty_alpha=generation_config.penalty_alpha,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            logits_warper = model_instance._get_logits_warper(generation_config)

            input_ids, model_kwargs = model_instance._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=model_instance.config.is_encoder_decoder,
                **model_kwargs,
            )

            return model_instance.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            input_ids, model_kwargs = model_instance._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=model_instance.config.is_encoder_decoder,
                **model_kwargs,
            )
            return model_instance.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            logits_warper = model_instance._get_logits_warper(generation_config)

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * generation_config.num_return_sequences,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                max_length=generation_config.max_length,
            )

            input_ids, model_kwargs = model_instance._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams * generation_config.num_return_sequences,
                is_encoder_decoder=model_instance.config.is_encoder_decoder,
                **model_kwargs,
            )

            return model_instance.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if generation_config.num_beams % generation_config.num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            has_default_typical_p = kwargs.get("typical_p") is None and generation_config.typical_p == 1.0
            if not has_default_typical_p:
                raise ValueError("Decoder argument `typical_p` is not supported with beam groups.")

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            input_ids, model_kwargs = model_instance._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=model_instance.config.is_encoder_decoder,
                **model_kwargs,
            )
            return model_instance.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_constraint_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            if generation_config.num_beams <= 1:
                raise ValueError("`num_beams` needs to be greater than 1 for constrained generation.")

            if generation_config.do_sample:
                raise ValueError("`do_sample` needs to be false for constrained generation.")

            if generation_config.num_beam_groups is not None and generation_config.num_beam_groups > 1:
                raise ValueError("`num_beam_groups` not supported yet for constrained generation.")

            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                        not isinstance(generation_config.force_words_ids, list)
                        or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                                any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                                for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            input_ids, model_kwargs = model_instance._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=model_instance.config.is_encoder_decoder,
                **model_kwargs,
            )
            return model_instance.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    def my_utils_greedy_search_top10(self,
                                     model_instance,
                                     gen_length,
                                     input_ids: torch.LongTensor,
                                     logits_processor: Optional[LogitsProcessorList] = None,
                                     stopping_criteria: Optional[StoppingCriteriaList] = None,
                                     max_length: Optional[int] = None,
                                     pad_token_id: Optional[int] = None,
                                     eos_token_id: Optional[Union[int, List[int]]] = None,
                                     output_attentions: Optional[bool] = None,
                                     output_hidden_states: Optional[bool] = None,
                                     output_scores: Optional[bool] = None,
                                     return_dict_in_generate: Optional[bool] = None,
                                     synced_gpus: bool = False,
                                     streamer: Optional["BaseStreamer"] = None,
                                     **model_kwargs,
                                     ):

        # init values
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=gen_length + input_ids.shape[1])])
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else model_instance.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else model_instance.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else model_instance.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else model_instance.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else model_instance.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else model_instance.generation_config.return_dict_in_generate
        )

        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and model_instance.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only


        while True:
            if synced_gpus:
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)

                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)

                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = model_instance.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = model_instance(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :]

            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if model_instance.config.is_encoder_decoder else (
                        outputs.attentions,)
                    )
                    if model_instance.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if model_instance.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            top_sign = self.top - 1 if self.first_check == 0 else 0  # first check, or to generate the rest
            next_tokens = torch.argsort(next_tokens_scores, dim=-1, descending=True)[:, top_sign]

            this_peer_finished, next_tokens = self.first_checking(next_tokens, next_tokens_scores)

            if next_tokens in self.zone_zero:
                this_peer_finished = True

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = model_instance._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=model_instance.config.is_encoder_decoder
            )

            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if model_instance.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    def require_first_to_be(self, next_tokens_scores, values_to_extract=[29871]):
        top_k_indices = torch.topk(next_tokens_scores, k=next_tokens_scores.shape[-1], dim=-1).indices
        top_k_indices_np = top_k_indices.cpu().numpy()
        mask = np.isin(top_k_indices_np, values_to_extract)
        top_k_indices = top_k_indices_np[mask][0]
        top_k_indices = torch.tensor(top_k_indices)

        next_tokens = top_k_indices.item()
        next_tokens = torch.tensor(next_tokens).reshape(-1)
        current_device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        return next_tokens.to(current_device), top_k_indices.to(current_device)


    def remove_str(self , entity):
        entity = entity.replace('_', ' ').replace('-', ' ')
        entity = entity.replace('(', ' ').replace(')', ' ')
        entity = entity.replace('\\', ' ').replace('.', ' ')
        entity = entity.replace('\"', ' ').replace('/', ' ')
        entity = entity.replace('\'', ' ').replace('&', ' ')
        entity = entity.replace('  ', ' ').replace('   ', ' ').replace('   ', ' ').replace('   ', ' ').replace('   ', ' ')
        entity = entity.lower()
        return entity




    def tell_entity_name(self, answer_regs, query, m_inloop, filter_m_count):
        entity_ans = []
        answer_regs = self.remove_str(answer_regs)
        for entity in self.entity_set:
            entity = self.remove_str(entity)
            if entity in answer_regs:
                entity_ans.append(entity)
        if len(entity_ans) == 1 and self.remove_str(entity_ans[0]) not in self.remove_str(query):
            return entity_ans[0], m_inloop, filter_m_count
        elif len(entity_ans) == 0:
            return "final", m_inloop, filter_m_count
        elif len(entity_ans) > 1:
            id_answ = 1000000000000000000
            for id in range(len(entity_ans)-1, -1, -1):
                current_position = answer_regs.find(entity_ans[id])
                if current_position < id_answ and self.remove_str(entity_ans[id]) not in self.remove_str(query):
                    id_answ = current_position
                    id_right = id
            try:
                return entity_ans[id_right], m_inloop, filter_m_count
            except:
                return entity_ans[0], m_inloop, filter_m_count
        else:
            return "final", m_inloop , filter_m_count




    def model_calling(self, m_inloop, query, filter_m_count):
        ids = self.tokenizer.encode(self.args.PROMPT)
        input_ids = torch.LongTensor([ids]).to('cuda')
        self.first_check = 0
        out = self.my_generate_top10(model_instance=self.model, m=m_inloop,
                                     input_ids=input_ids,
                                     max_length=self.args.CONTEXT_LEN,
                                     gen_length=36,
                                     do_sample=False,
                                     )
        out_text = self.tokenizer.decode(out[0])
        answer = out_text.replace(self.args.PROMPT, "").replace("\nEND", "").strip()

        answer = answer.replace("\n", "")
        answer_regs, m_inloop, filter_m_count = self.tell_entity_name(answer, query, m_inloop, filter_m_count)
        answer___ = []
        answer___.append(answer_regs)
        return answer, answer___, m_inloop, filter_m_count

    def eval(self, c, cnt=0, path_results=None, filter_yes=True):
        def clculate_f1(tok_pred, tok_gold):
            if len(tok_gold) == 0:  # do not generate anything
                if len(tok_pred) == 0:
                    f1 = 1
                    m_F1.append(f1)
                else:
                    f1 = 0
                    m_F1.append(f1)
            else:
                tok_gold_dict = Counter(tok_gold)
                tok_pred_dict = Counter(tok_pred)
                tokens = set([*tok_gold_dict] + [*tok_pred_dict])
                hit = 0
                for token in tokens:
                    hit += min(tok_gold_dict.get(token, 0), tok_pred_dict.get(token, 0))
                p = hit / (sum(tok_pred_dict.values()) + 1e-10)
                r = hit / (sum(tok_gold_dict.values()) + 1e-10)
                F1 = 2 * p * r / (p + r + 1e-10)
                m_F1.append(F1)

        query = ""
        c1 = c["c1"]

        if path_results is not None:
            test_results = read_results(path_results)
        dict_qu_ans = self.gen_set_ans(dir_full_test=self.args.fulltest, dir_time2id=self.args.time2id)
        set_checked_qu = set()
        num_infer = len(self.tests)  #
        for i in tqdm(range(cnt, num_infer)):
            his_query = self.tests[i]
            try:
                query = his_query.split('Question')[3]
            except:
                query = his_query.split('Question')[2]
            val_trunc = -1
            if len(his_query) - 1 > val_trunc and val_trunc != -1:
                li_his_trunc = his_query.split('\n')[-val_trunc - 1:-1]  # backward
                li_his_trunc.append(query)
                his_query = "\n".join(li_his_trunc)

            
            
            delete = False
            if delete == True:
                his_query = re.sub(r'\d+:\s', '', his_query)


            ins = '''You must be able to correctly predict the next {object} from a given text consisting of multiple sentnences in the form of "At time {time} {subject} {relation} {object}." and the query in the form of "At time {time} what does {subject} {relation} ?" in the end. You must directly generate the missing {object}.\n'''
            self.args.PROMPT = ins + his_query


            if query not in set_checked_qu:
                set_checked_qu.add(query)
                hello = "For"

            else:
                hello = "Duplicate query:"
            print(hello, query)
            if query == '':
                continue
            print("Given answers", dict_qu_ans[query], "with", self.test_ans[i], "as the gt")

            content_to_write = []
            content_to_write2 = []
            m_inloop = -1
            filter_m_count = -1
            k_inloop = 5
            self.constraints = []
            self.top = 1
            exist_num = 0
            if path_results is not None:
                num_Test, li_results = read_num_and_li_results(test_results[i])
                exist_num = len(li_results)
                if int(num_Test) != i:
                    print(num_Test, i)
                    raise ValueError("Test id and i do not match.")
            while m_inloop < k_inloop - 1 and m_inloop <= 5:
                m_inloop += 1
                filter_m_count += 1
                with torch.no_grad():
                    if path_results is None:
                        raw_ans, answer_regs, m_inloop, filter_m_count = self.model_calling(m_inloop, query, filter_m_count)
                        print(str(m_inloop) + "-th time, I would say, ", answer_regs)
                    else:

                        if m_inloop >= exist_num:
                            if not filter_yes:
                                break
                            else:
                                print("call of duty")
                                raw_ans, answer_regs = self.generate_extra_answers(m_inloop, k_inloop)
                                print(str(m_inloop) + "-th time, I would say, ", answer_regs)
                        else:
                            raw_ans = answer_regs = [li_results[m_inloop]]
                            pattern = re.compile(r'.*?[\d:@][._](.*)\]')
                            answer_regs = [re.match(pattern, answer_regs[0]).group(2).strip()] \
                                if re.match(pattern, answer_regs[0]) else answer_regs
                            print(str(m_inloop) + " read ", answer_regs)
                            self.top += 1

                    content_to_write.append('\n' + str(answer_regs))
                    content_to_write2.append('\n' + str(raw_ans))

                    bingo = False
                    dict_qu_ans_lower = [self.remove_str(ans).lower() for ans in dict_qu_ans[query]]
                    for answer in answer_regs:
                        answerlow = answer.lower()

                        gtlow = self.test_ans[i].lower()
                        clculate_f1(answerlow, gtlow)
                        if answer == '':
                            content_to_write.append("(none string; removed)")
                            k_inloop += 1
                            filter_m_count -= 1
                            print("increased k: " + str(k_inloop))
                            break
                        if (
                                self.remove_str(answerlow) != self.remove_str(gtlow) and answerlow in dict_qu_ans_lower) and filter_yes:  # first_check = -1 if to check breach of obligation
                            print("Got another answer: " + answer)
                            bingo = True
                            if filter_m_count == 0:
                                c1 += 1
                            print("Bingo! Line: ", i, "count after filtering: ", filter_m_count + 1, "all count: ", \
                                  m_inloop + 1, "answer: ", answer, "gt: ", self.test_ans[i])
                            break
                        elif self.remove_str(answerlow) == self.remove_str(gtlow):
                            bingo = True
                            if filter_m_count == 0:
                                c1 += 1
                            print("Bingo! Line: ", i, "count after filtering: ", filter_m_count + 1, "all count: ", \
                                  m_inloop + 1, "answer: ", answer, "gt: ", self.test_ans[i])
                            break

                    if bingo:
                        break
            hits_1 = c1 / (i + 1)

            with open(self.eval_txt_path, "a", encoding="utf-8") as fout:
                if self.args.ft == 1:
                    fout.write('current model: ' + self.args.LORA_CHECKPOINT_DIR + ', \n')
                else:
                    fout.write('current model: ' + self.args.MODEL_NAME + ', \n')
                fout.write(self.args.output_file + ' currently finished: ' + str(i + 1) + '; results: \n')
                fout.write("Hits@1: " + str(round(hits_1, 3)) + "\n")
                fout.write(str(c1) + "\n")
                fout.write("F1" + "\n")
                F111 = sum(m_F1)/len(m_F1)
                fout.write( str(F111) + "\n")

            with open(self.args.output_file, 'a', encoding='utf-8') as f:
                f.write('{"Test' + str(i) + '": ["' + ', '.join(content_to_write) + '"]}, \n\n')
            with open(self.args.output_file.replace(".txt", "_raw.txt"), 'a', encoding='utf-8') as f:
                f.write('{"Test' + str(i) + '": ["' + ', '.join(content_to_write2) + '"]}, \n\n')

            print('processing: ' + self.args.output_file, i + 1)
            time.sleep(0.001)