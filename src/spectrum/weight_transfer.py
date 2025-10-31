"""
uv run src/spectrum/weight_transfer.py

Note: With the memory requirements to hold all models, may need to have more than a single 80GB A100 to avoid OOM errors.
"""

import json
import logging
import os
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def copy_token_embeddings(
    base_model_name: str,
    instruct_model_name: str,
    tokens: List[str],
    device_map: str = "auto",
    save_path: Optional[str] = None,
) -> AutoModelForCausalLM:
    """
    Copy embedding and unembedding weights for specific tokens from instruct model to base model.

    Args:
        base_model_name: Name or path of the base model (e.g., "Qwen/Qwen3-0.6B-Base")
        instruct_model_name: Name or path of the instruct model (e.g., "Qwen/Qwen3-0.6B")
        tokens: List of tokens to copy weights for
        device_map: Device mapping strategy (default: "auto")
        save_path: Optional path to save the modified model

    Returns:
        Modified base model with copied token embeddings

    Raises:
        ValueError: If tokens are not found in the instruct model tokenizer
        RuntimeError: If model architectures don't match
    """
    logger.info(f"Loading base model: {base_model_name}")

    # Load base model and tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map=device_map
    )

    logger.info(f"Loading instruct model: {instruct_model_name}")

    # Load instruct model and tokenizer
    instruct_tokenizer = AutoTokenizer.from_pretrained(
        instruct_model_name, use_fast=True
    )
    if instruct_tokenizer.pad_token is None:
        instruct_tokenizer.pad_token = instruct_tokenizer.eos_token

    instruct_model = AutoModelForCausalLM.from_pretrained(
        instruct_model_name, device_map=device_map
    )

    # Validate model architectures match
    if type(base_model) != type(instruct_model):
        raise RuntimeError(
            f"Model architectures don't match: {type(base_model)} vs {type(instruct_model)}"
        )

    # Get embedding and unembedding layers
    base_embeddings = base_model.get_input_embeddings()
    base_lm_head = base_model.get_output_embeddings()

    instruct_embeddings = instruct_model.get_input_embeddings()
    instruct_lm_head = instruct_model.get_output_embeddings()

    # Validate embedding dimensions match
    if base_embeddings.weight.shape != instruct_embeddings.weight.shape:
        raise RuntimeError(
            f"Embedding dimensions don't match: {base_embeddings.weight.shape} vs {instruct_embeddings.weight.shape}"
        )

    if base_lm_head.weight.shape != instruct_lm_head.weight.shape:
        raise RuntimeError(
            f"LM head dimensions don't match: {base_lm_head.weight.shape} vs {instruct_lm_head.weight.shape}"
        )

    # Get token IDs for the specified tokens
    token_ids = []
    missing_tokens = []

    for token in tokens:
        # Try to encode the token
        token_id = instruct_tokenizer.encode(token, add_special_tokens=False)

        if len(token_id) != 1:
            missing_tokens.append(token)
            continue

        token_id = token_id[0]

        # Check if token exists in base tokenizer
        base_token_id = base_tokenizer.encode(token, add_special_tokens=False)
        if len(base_token_id) != 1 or base_token_id[0] != token_id:
            logger.warning(
                f"Token '{token}' has different ID in base model: {base_token_id} vs {token_id}"
            )

        token_ids.append(token_id)

    if missing_tokens:
        raise ValueError(
            f"Tokens not found or ambiguous in instruct tokenizer: {missing_tokens}"
        )

    logger.info(f"Copying embeddings for {len(token_ids)} tokens: {tokens}")

    # Copy embedding weights
    with torch.no_grad():
        for token_id in token_ids:
            # Copy input embedding
            base_embeddings.weight[token_id].copy_(instruct_embeddings.weight[token_id])

            # Copy output embedding (lm_head)
            base_lm_head.weight[token_id].copy_(instruct_lm_head.weight[token_id])

    logger.info("Weight copying completed successfully")

    # copy over config from instruct model to base model
    base_model.config = instruct_model.config
    # # copy over preprocessor config
    # print(instruct_model.preprocessor_config)
    # print(base_model.preprocessor_config)
    # base_model.preprocessor_config = instruct_model.preprocessor_config

    # Save model if path provided
    if save_path:
        logger.info(f"Saving modified model to: {save_path}")
        base_model.save_pretrained(save_path)
        base_tokenizer.save_pretrained(save_path)
        # save preprocessor config
        if "gemma" in base_model_name:
            with open(os.path.join(save_path, "preprocessor_config.json"), "w") as f:
                json.dump(gemma_preprocessor_config, f)

    del base_model, base_tokenizer, instruct_model, instruct_tokenizer


if __name__ == "__main__":
    import os

    device_map = "auto"
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    save_dir = "base_modified"

    model_args = {
        "meta-llama/Llama-3.1-8B": {
            "base_model_name": "meta-llama/Llama-3.1-8B",
            "instruct_model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "tokens": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        },
        "Qwen/Qwen3-14B-Base": {
            "base_model_name": "Qwen/Qwen3-14B-Base",
            "instruct_model_name": "Qwen/Qwen3-14B",
            "tokens": ["<|im_start|>", "<|im_end|>"],
        },
        "google/gemma-3-12b-pt": {
            "base_model_name": "google/gemma-3-12b-pt",
            "instruct_model_name": "google/gemma-3-12b-it",
            "tokens": ["<start_of_turn>", "<end_of_turn>"],
        },
    }

    for model_name, args in model_args.items():
        save_path = os.path.join(save_dir, model_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        copy_token_embeddings(
            args["base_model_name"],
            args["instruct_model_name"],
            args["tokens"],
            device_map=device_map,
            save_path=save_path,
        )
        # clear all memory
        torch.cuda.empty_cache()
