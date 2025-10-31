import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding


def prepare_batch(
    tokenizer: AutoTokenizer,
    input_texts: List[str],
    output_texts: List[str] = None,
    bos_token: str = "",
    eos_token: str = "",
    add_bos_token: bool = False,
    add_eos_token: bool = False,
    prompt_keep_tokens: List[int] = None,
    max_length: int = 1024,
    filter_strings: List[str] = ["<bos>"],
) -> BatchEncoding:
    bos_token = bos_token or tokenizer.bos_token
    bos_token = bos_token or ""
    eos_token = eos_token or tokenizer.eos_token
    eos_token = eos_token or ""

    # remove bos
    for s in filter_strings:
        input_texts = [i.replace(s, "") for i in input_texts]
        if output_texts is not None:
            output_texts = [o.replace(s, "") for o in output_texts]

    add_bos_token = False  # TODO: remove
    if add_bos_token:
        inputs = [bos_token + i for i in input_texts]
    else:
        inputs = input_texts

    if output_texts is not None:
        assert len(input_texts) == len(
            output_texts
        ), "Inputs and outputs must match in length."

        for inp, out in zip(input_texts, output_texts):
            if add_eos_token:
                outputs = [o + eos_token for o in output_texts]
            else:
                outputs = output_texts

        # Combine inputs and outputs with proper BOS/EOS tokens
        full_texts = [
            # tokenizer.bos_token + inp.strip() + " " + out.strip() + tokenizer.eos_token
            inp + out
            for inp, out in zip(inputs, outputs)
        ]
    else:
        full_texts = inputs

    # Tokenize full sequence (input + completion)
    tokenized_full = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        padding_side="right",
    )

    # get full text lengths
    full_text_lens = tokenizer(full_texts, return_length=True).length
    # get input lengths
    input_lens = tokenizer(inputs, return_length=True).length
    if output_texts is not None:
        output_lens = tokenizer(outputs, return_length=True).length
    else:
        output_lens = None

    input_ids = tokenized_full.input_ids
    attention_mask = tokenized_full.attention_mask

    # check if double bos token, if so remove
    while (
        input_ids[0][0] == tokenizer.bos_token_id
        and input_ids[0][1] == tokenizer.bos_token_id
    ):
        print("double bos token, removing")
        input_ids = input_ids[:, 1:]
        attention_mask = attention_mask[:, 1:]
        full_text_lens = [l - 1 for l in full_text_lens]
        input_lens = [l - 1 for l in input_lens]

    # Create labels tensor and mask input tokens with -100
    labels = input_ids.clone()
    for idx in range(labels.size(0)):
        # input_len = tokenized_inputs.input_ids[idx].size(0)
        # input_lens.append(input_len)
        input_len = input_lens[idx]
        if prompt_keep_tokens is not None:
            # make -100 everywhere except up to input_len except for prompt_keep_tokens
            labels[idx, :input_len] = -100
            # get where input_ids is in prompt_keep_tokens
            prompt_keep_mask = torch.tensor(
                [i in prompt_keep_tokens for i in input_ids[idx]]
            )
            # set those to the actual token
            labels[idx, prompt_keep_mask] = input_ids[idx, prompt_keep_mask]
        else:
            labels[idx, :input_len] = -100  # Mask prompt tokens
        # mask after full text length
        labels[idx, full_text_lens[idx] :] = -100

    # Return as BatchEncoding object
    batch_encoding = BatchEncoding(
        {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    )

    batch_encoding.total_lens = np.array(full_text_lens)
    batch_encoding.input_lens = np.array(input_lens)
    if output_lens is not None:
        batch_encoding.output_lens = np.array(output_lens)
    else:
        batch_encoding.output_lens = (
            batch_encoding.total_lens - batch_encoding.input_lens
        )

    # check if any input_lens is greater than max_length
    if (batch_encoding.input_lens > max_length).any():
        print(f"Input length is greater than max_length: {batch_encoding.input_lens}")
        breakpoint()
        raise ValueError(
            f"Input length is greater than max_length: {batch_encoding.input_lens}"
        )

    return batch_encoding


def score_batch(
    batch: BatchEncoding,
    model: AutoModelForCausalLM,
) -> Dict[str, Any]:
    # move batch to model device
    batch = batch.to(model.device)
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
        loss = outputs.loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss_per_token = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    loss_per_token = loss_per_token.view(shift_labels.size())

    loss_mask = batch["labels"] != -100
    # get total tokens
    total_tokens = loss_mask.sum(axis=1)
    loss_per_seq = loss_per_token.sum(axis=1)
    perplexity = loss_per_seq / total_tokens
    return {
        "total_loss": loss.item(),
        "nll": loss_per_seq.tolist(),
        "perplexity": perplexity.tolist(),
        "total_tokens": total_tokens.tolist(),
    }


def score_all(
    input_texts: List[str],
    output_texts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
) -> pd.DataFrame:
    results = []
    for i in tqdm(range(0, len(input_texts), batch_size)):
        input_texts_batch = input_texts[i : i + batch_size]
        output_texts_batch = output_texts[i : i + batch_size]
        batch = prepare_batch(tokenizer, input_texts_batch, output_texts_batch)
        scored = score_batch(batch, model)
        for j in range(len(input_texts_batch)):
            results.append(
                {
                    "nll": scored["nll"][j],
                    "perplexity": scored["perplexity"][j],
                    "total_tokens": scored["total_tokens"][j],
                }
            )
    return pd.DataFrame(results)


def get_logprobs(
    batch: BatchEncoding,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer = None,
    restrict_tokens: List[int] = None,
    n_logprobs: int = 10,
    min_coverage: float = 0.1,
) -> List[float]:
    batch = batch.to(model.device)
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
    # # logits to cpu
    # logits = logits.cpu()
    # batch.input_lens
    # get logits at the input_len position
    logits = logits[np.arange(logits.shape[0]), batch.input_lens - 1, :]
    # if restrict_tokens is not None, restrict the logits to the tokens in restrict_tokens
    # if restrict_tokens is not None:
    #     logits = logits[:, restrict_tokens]
    # do torch softmax
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    # if restrict_tokens is not None, restrict the logprobs to the tokens in restrict_tokens
    if restrict_tokens is not None:
        logprobs = logprobs[:, restrict_tokens]
    # shift_logits = logits[:, :-1, :].contiguous()
    # shift_labels = batch["labels"][:, 1:].contiguous()
    # get the logprobs at theh last position
    # logprobs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    # logprobs = logprobs.gather(dim=-1, index=shift_labels.unsqueeze(-1))

    # logprobs = get_logprobs(batch, model)
    # get the top logprobs
    top_tokens = logprobs.argsort(dim=-1, descending=True)[:, :n_logprobs]
    # get the logprobs of the top logprobs
    top_logprobs = logprobs.gather(dim=-1, index=top_tokens)

    ret_dict = {
        "all_logprobs": logprobs,
        # 'top_logprobs': top_logprobs,
        # 'top_tokens': top_tokens,
    }
    if restrict_tokens is None:
        dicts = []
        for i in range(logprobs.shape[0]):
            logprob_dict = {}
            # for j in range(n_logprobs):
            for j in range(min(n_logprobs, top_logprobs.shape[1])):
                # logprob_dict[tokenizer.decode(top_tokens[i, j])] = top_logprobs[i, j].item()
                if tokenizer is not None:
                    logprob_dict[tokenizer.decode(top_tokens[i, j])] = float(
                        np.exp(top_logprobs[i, j].item())
                    )
                else:
                    logprob_dict[top_tokens[i, j].item()] = float(
                        np.exp(top_logprobs[i, j].item())
                    )
            dicts.append(logprob_dict)
        ret_dict["logprobs_dicts"] = dicts

    # coverage is the total weight on restrict_tokens, if exists
    if restrict_tokens is not None:
        coverage = logprobs.exp().sum(dim=-1)
        # normalize by coverage
        logprobs = logprobs - torch.log(coverage).reshape(-1, 1)
        ret_dict["coverage"] = coverage
        ret_dict["logprobs"] = logprobs
        # if coverage < min_coverage, raise warning
        if coverage.min() < min_coverage:
            text = f"Coverage is {coverage.min()} (<{min_coverage}), logprobs may be unreliable"
            print(text)
            breakpoint()
            raise ValueError(text)
            print(text)
    return ret_dict


def prepare_gen_batch(
    tokenizer: AutoTokenizer,
    gen_texts: List[str],
    bos_token: str = None,
    add_bos_token: bool = False,
) -> BatchEncoding:
    bos_token = bos_token or tokenizer.bos_token
    bos_token = bos_token or ""
    if add_bos_token:
        gen_texts = [bos_token + i for i in gen_texts]
    return tokenizer(gen_texts, return_tensors="pt", padding=True, padding_side="left")


def batch_generate(
    batch: BatchEncoding,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 10,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
    **kwargs,
) -> List[str]:
    """Generate text completions for a batch of inputs."""
    # send to device
    batch = batch.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs,
        )

        # Decode the generated outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove the input tokens to get only the generated part
            input_length = batch["input_ids"].shape[1]
            generated_tokens = output[input_length:]
            generated_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            generated_texts.append(generated_text)

        return generated_texts


def all_generate(
    gen_texts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    n_gens_per_input: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """Generate text completions for a batch of inputs."""
    results = []
    # if gen_texts[0] is a list of dicts, apply chat template first
    if isinstance(gen_texts[0], list):
        if isinstance(gen_texts[0][0], dict):
            # apply chat template
            gen_texts = tokenizer.apply_chat_template(
                gen_texts, tokenize=False, add_generation_prompt=True
            )
    # repeat gen_texts n_gens_per_input
    gen_texts = np.repeat(gen_texts, n_gens_per_input)
    # turn back to list
    gen_texts = gen_texts.tolist()
    # save text_inds
    text_inds = np.repeat(
        np.arange(len(gen_texts) // n_gens_per_input), n_gens_per_input
    )
    # and gen_inds
    gen_inds = np.tile(np.arange(n_gens_per_input), len(gen_texts) // n_gens_per_input)

    for i in tqdm(range(0, len(gen_texts), batch_size)):
        gen_texts_batch = gen_texts[i : i + batch_size]
        batch = prepare_gen_batch(tokenizer, gen_texts_batch)
        generated_texts = batch_generate(batch, model, tokenizer, **kwargs)
        for input_text, generated_text, text_ind, gen_ind in zip(
            gen_texts_batch,
            generated_texts,
            text_inds[i : i + batch_size],
            gen_inds[i : i + batch_size],
        ):
            results.append(
                {
                    "text_ind": text_ind,
                    "input": input_text,
                    "output": generated_text,
                    "gen_ind": gen_ind,
                }
            )
    return pd.DataFrame(results)


def get_restrict_tokens(
    input_texts: List[str],
    tokenizer: AutoTokenizer,
) -> List[int]:
    # tokenize input_texts
    # tokenized_texts = tokenizer(input_texts, return_tensors='pt', padding=True, padding_side='left')
    tokenized_texts = tokenizer(
        input_texts, return_tensors="pt", padding=True, padding_side="right"
    )
    # if starts with bos token, remove
    if tokenized_texts["input_ids"][0][0] == tokenizer.bos_token_id:
        tokenized_texts["input_ids"] = tokenized_texts["input_ids"][:, 1:]
        tokenized_texts["attention_mask"] = tokenized_texts["attention_mask"][:, 1:]
    # get the first_token of each input
    restrict_tokens = tokenized_texts["input_ids"][:, 0]
    # if not unique, raise error
    if len(restrict_tokens) != len(set(restrict_tokens)):
        raise ValueError("restrict_tokens must be unique")
    return restrict_tokens


def all_logprobs(
    input_texts: List[str],
    # output_texts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    restrict_tokens: List[int] = None,
    max_length: int = 1024,
    coverage_min: float = 0.1,
) -> pd.DataFrame:
    if restrict_tokens is None:
        raise ValueError("restrict_tokens must be provided")
    # move restrict_tokens to device
    restrict_tokens = torch.tensor(restrict_tokens, device=model.device)
    results = []
    for i in tqdm(range(0, len(input_texts), batch_size)):
        input_texts_batch = input_texts[i : i + batch_size]
        batch = prepare_batch(tokenizer, input_texts_batch, max_length=max_length)
        logprobs = get_logprobs(
            batch,
            model,
            tokenizer,
            restrict_tokens=restrict_tokens,
            min_coverage=coverage_min,
        )
        for input_text, logprobs, coverage in zip(
            input_texts_batch, logprobs["all_logprobs"], logprobs["coverage"]
        ):
            results.append(
                {
                    "input": input_text,
                    "logprobs": logprobs.detach().cpu().numpy(),
                    "probs": logprobs.exp().detach().cpu().numpy(),
                    "coverage": coverage.detach().cpu().numpy().item(),
                    "pred": logprobs.argmax(dim=-1).detach().cpu().numpy(),
                }
            )
    return pd.DataFrame(results)


import torch
from transformers import AutoModelForCausalLM


def create_model_soup(model_names, weights=None):
    """
    Memory‑efficient model soup that forms a weighted average of several models
    **in place** on the first model, so we never hold an extra full copy of the
    weights in GPU memory.

    Args:
        model_names (List[str]): Hugging Face hub ids or local paths.  The first
            model in the list acts as the running accumulator and will be
            returned.
        weights (List[float], optional): Non‑negative weights that must sum to
            1.  If None, the models are averaged equally.

    Returns
        AutoModelForCausalLM: The averaged model resident on the same devices as
            the first model.
    """
    if weights is None:
        weights = [1.0 / len(model_names)] * len(model_names)

    assert len(weights) == len(model_names), "Provide a weight for every model."
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1."

    # --------------------------------------------------------------------- #
    # Load the first model and scale its parameters by its mixing weight.   #
    # --------------------------------------------------------------------- #
    base_model = AutoModelForCausalLM.from_pretrained(
        model_names[0],
        device_map="auto",
        attn_implementation="eager" if "gemma" in model_names[0] else None,
    )
    base_state = base_model.state_dict()

    with torch.no_grad():
        for p in base_state.values():
            p.mul_(weights[0])

    torch.cuda.empty_cache()

    # --------------------------------------------------------------------- #
    # Iteratively load each remaining model, add its weighted parameters,   #
    # and immediately free its memory.                                      #
    # --------------------------------------------------------------------- #
    for idx, model_name in enumerate(model_names[1:], 1):
        next_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )
        next_state = next_model.state_dict()

        if set(next_state.keys()) != set(base_state.keys()):
            raise ValueError(
                f"Model '{model_name}' is not architecture‑compatible with "
                f"'{model_names[0]}'."
            )

        with torch.no_grad():
            for key, next_param in next_state.items():
                base_state[key].add_(
                    weights[idx] * next_param.to(base_state[key].device)
                )

        # Free GPU/CPU RAM held by the current model before loading the next
        del next_model, next_state
        torch.cuda.empty_cache()

    return base_model
