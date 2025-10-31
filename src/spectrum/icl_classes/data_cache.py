import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer

from spectrum.format_utils import messages_to_loss_texts
from spectrum.icl_classes.dataloaders import load_default_data
from spectrum.icl_classes.utils import format_loss_texts
from spectrum.sample_utils import (
    MyDataCollatorForLanguageModeling,
    tokenize_example,
    tokenize_loss_texts,
)

DEFAULT_CACHE_DIR = "tokenized_datasets"

logger = logging.getLogger(__name__)


def get_safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def get_model_family(model_name: str) -> str:
    safe_model_name = get_safe_model_name(model_name)
    if "Qwen3" in safe_model_name:
        return "Qwen3"
    if "gemma3" in safe_model_name or "gemma-3" in safe_model_name:
        return "gemma3"
    if "Llama-3.1" in safe_model_name:
        return "Llama3.1"
    if "Llama-3.2" in safe_model_name:
        return "Llama3.2"
    else:
        raise ValueError(f"Model family not found for {safe_model_name} for cache dir")


def get_cache_dir(dataset_name: str, model_name: str, cache_parent_dir=None) -> str:
    if cache_parent_dir is None:
        cache_parent_dir = DEFAULT_CACHE_DIR
    model_family = get_model_family(model_name)
    cache_dir = os.path.join(cache_parent_dir, dataset_name, model_family)
    return cache_dir


def get_processed_cache_dir(
    dataset_name: str,
    model_name: str,
    max_length: int = None,
    format: str = "spectrum",
    cache_parent_dir: str = None,
    fixed_example_number: int = None,
) -> str:
    """Get cache directory path for processed datasets (truncated & filtered)."""
    cache_dir = get_cache_dir(dataset_name, model_name, cache_parent_dir)

    # Use format directly as subdirectory name
    cache_dir = os.path.join(cache_dir, format)

    if max_length is not None and fixed_example_number is not None:
        print(
            f"max_length and fixed_example_number are provided together for {dataset_name}"
        )
        breakpoint()
        raise ValueError(
            "max_length and fixed_example_number cannot be provided together"
        )

    if max_length is not None:
        processed_dir = os.path.join(cache_dir, f"processed_maxlen_{max_length}")
    elif fixed_example_number is not None:
        processed_dir = os.path.join(
            cache_dir, f"processed_fixed_{fixed_example_number}"
        )
    else:
        breakpoint()
        raise ValueError(
            "Either max_length or fixed_example_number must be provided for processed cache dir"
        )
    return processed_dir


def get_raw_cache_dir(
    dataset_name: str,
    model_name: str,
    format: str = "spectrum",
    cache_parent_dir: str = None,
) -> str:
    """Get cache directory path for raw datasets (unprocessed)."""
    cache_dir = get_cache_dir(dataset_name, model_name, cache_parent_dir)

    # Use format directly as subdirectory name
    cache_dir = os.path.join(cache_dir, format)

    raw_dir = os.path.join(cache_dir, "raw")
    return raw_dir


def migrate_to_raw_structure(cache_dir: str) -> bool:
    """Migrate existing cache structure to include raw/ subdirectory.

    Args:
        cache_dir: Directory like tokenized_datasets/dataset/model_family/variant/

    Returns:
        True if migration was performed, False if not needed
    """
    cache_path = Path(cache_dir)
    raw_dir = cache_path / "raw"

    # Check if migration is needed
    if raw_dir.exists():
        return False  # Already migrated

    # Check if there are dataset files to migrate
    dataset_files = list(cache_path.glob("*.arrow")) + list(cache_path.glob("*.json"))
    if not dataset_files:
        return False  # No files to migrate

    logger.info(f"Migrating cache structure to raw/ for {cache_dir}")

    try:
        # Create raw directory
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Move all dataset files to raw/
        for file_path in cache_path.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                dest_path = raw_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))
                logger.debug(f"Moved {file_path.name} to raw/")

        logger.info(
            f"Successfully migrated {len(dataset_files)} files to raw/ structure"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to migrate cache structure: {e}")
        # If migration fails, try to clean up
        if raw_dir.exists():
            shutil.rmtree(str(raw_dir), ignore_errors=True)
        raise


def processed_cache_exists(
    dataset_names: List[str],
    model_name: str,
    max_length: int = None,
    format: str = "spectrum",
    cache_parent_dir: str = None,
    fixed_example_number: int = None,
) -> bool:
    """Check if processed cache exists for all datasets."""
    for dataset_name in dataset_names:
        processed_dir = get_processed_cache_dir(
            dataset_name,
            model_name,
            max_length,
            format,
            cache_parent_dir,
            fixed_example_number,
        )
        if not os.path.isdir(processed_dir):
            return False
    return True


def truncate_to_max_length(dataset: Dataset, max_length: int) -> Dataset:
    """Apply truncating, n_labels calculation, and filtering to dataset."""
    logger.info(
        f"Applying processing: truncating to {max_length}, calculating n_labels, filtering"
    )

    # 1. Truncate to max_length
    dataset = dataset.map(
        lambda x: {
            "input_ids": x["input_ids"][:max_length],
            "attention_mask": x["attention_mask"][:max_length],
            "labels": x["labels"][:max_length],
            "example_inds": x["example_inds"][:max_length],
            "data_ids": x["data_ids"][:max_length],
        }
    )

    # 2. Calculate n_labels (count non -100 labels)
    dataset = dataset.map(
        lambda x: {"n_labels": len([a for a in x["labels"] if a != -100])}
    )

    # 3. Filter out examples with no labels
    original_len = len(dataset)
    # Save original indices before filtering
    dataset = dataset.map(lambda x, idx: {"original_index": idx}, with_indices=True)
    dataset = dataset.filter(lambda x: x["n_labels"] > 0)
    filtered_len = len(dataset)

    logger.info(
        f"Filtered {original_len - filtered_len} examples with no labels (kept {filtered_len})"
    )

    return dataset


def truncate_to_fixed_examples(dataset: Dataset, fixed_example_number: int) -> Dataset:
    """Truncate to fixed number of examples."""

    def truncate_to_fixed_examples(example):
        inds = np.array(example["example_inds"]) == fixed_example_number
        if not np.any(inds):
            print(f"No inds found for fixed_example_number {fixed_example_number}")
            breakpoint()
            raise ValueError(
                f"No inds found for fixed_example_number {fixed_example_number}"
            )
        last_ind = np.argwhere(inds).max() + 1
        # truncate to last_ind
        return {
            "input_ids": example["input_ids"][:last_ind],
            "attention_mask": example["attention_mask"][:last_ind],
            "labels": example["labels"][:last_ind],
            "example_inds": example["example_inds"][:last_ind],
            "data_ids": example["data_ids"][:last_ind],
        }

    return dataset.map(truncate_to_fixed_examples)


def cache_processed_dataset(
    dataset: Dataset,
    dataset_name: str,
    model_name: str,
    max_length: int = None,
    format: str = "spectrum",
    cache_parent_dir: str = None,
    fixed_example_number: int = None,
) -> None:
    """Cache processed dataset to disk."""
    processed_dir = get_processed_cache_dir(
        dataset_name,
        model_name,
        max_length,
        format,
        cache_parent_dir,
        fixed_example_number,
    )

    logger.info(f"Caching processed dataset to {processed_dir}")
    os.makedirs(processed_dir, exist_ok=True)
    dataset.save_to_disk(processed_dir)


def load_processed_dataset(
    dataset_name: str,
    model_name: str,
    max_length: int = None,
    format: str = "spectrum",
    cache_parent_dir: str = None,
    fixed_example_number: int = None,
) -> Dataset:
    """Load processed dataset from cache."""
    processed_dir = get_processed_cache_dir(
        dataset_name,
        model_name,
        max_length,
        format,
        cache_parent_dir,
        fixed_example_number,
    )

    logger.info(f"Loading processed dataset from {processed_dir}")
    dataset = load_from_disk(processed_dir)
    return dataset


def add_example_inds(dataset):
    # TAYLOR - some formats (eg chat) may have example_ind that are not properly set up - for now, it's hacky but let's infer it by just counting the contiguous segments of labels that are not -100
    def get_example_ind(example):
        example_inds = []
        curr_ind = -1
        last_has_content = False
        for label in example["labels"]:
            if label != -100:
                if not last_has_content:
                    curr_ind += 1
                last_has_content = True
                example_inds.append(curr_ind)
            else:
                last_has_content = False
                example_inds.append(-1)
        if "example_inds" in example.keys():
            if np.any(np.array(example["example_inds"]) != -1):
                # assert that they should be the same
                old_example_inds = np.array(example["example_inds"])
                # where labels == -100, set example_inds to -1
                old_example_inds[np.array(example["labels"]) == -100] = -1
                if not np.all(np.array(example_inds) == old_example_inds):
                    print("Example_inds are not the same - check implementation")
                    breakpoint()
                    raise ValueError(
                        "Example_inds are not the same - check implementation"
                    )
                return {"example_inds": example["example_inds"]}
        return {"example_inds": example_inds}

    return dataset.map(get_example_ind)


def load_tokenized_dataset(
    dataset_name: str,
    force_reload: bool = False,
    model_name: str = None,
    save=True,
    cache_parent_dir=None,
    max_length=None,
    drop_additional_columns=True,
    keep_inds=None,
    fixed_example_number=None,
    **kwargs,
) -> Dataset:
    print(f"Loading tokenized dataset {dataset_name}")

    # Check template format - prefer format argument, fall back to individual flags for backward compatibility
    format = kwargs.get("format", None)
    # TAYLOR - now, let's have ALL use spectrum data structure for simplicity

    # If max_length is provided, try to load processed cache first
    if max_length and not force_reload:
        processed_dir = get_processed_cache_dir(
            dataset_name,
            model_name,
            max_length,
            format,
            cache_parent_dir,
            fixed_example_number,
        )
        if os.path.isdir(processed_dir):
            logger.info(f"Loading processed dataset from cache: {processed_dir}")
            dataset = load_from_disk(processed_dir)
            return dataset
    elif fixed_example_number and not force_reload:
        processed_dir = get_processed_cache_dir(
            dataset_name,
            model_name,
            max_length,
            format,
            cache_parent_dir,
            fixed_example_number,
        )
        if os.path.isdir(processed_dir):
            logger.info(f"Loading processed dataset from cache: {processed_dir}")
            dataset = load_from_disk(processed_dir)
            return dataset

    # Get raw cache directory
    raw_cache_dir = get_raw_cache_dir(
        dataset_name, model_name, format, cache_parent_dir
    )

    # Check for legacy cache and migrate if needed
    legacy_cache_dir = get_cache_dir(dataset_name, model_name, cache_parent_dir)
    # Legacy formats used "regular" for what is now "spectrum"
    legacy_format = "regular" if format == "spectrum" else format
    legacy_cache_dir = os.path.join(legacy_cache_dir, legacy_format)

    # Migrate legacy cache structure if needed
    if os.path.isdir(legacy_cache_dir):
        migrate_to_raw_structure(legacy_cache_dir)

    # Try to load raw dataset from cache
    dataset = None
    if not force_reload and os.path.isdir(raw_cache_dir):
        logger.info(f"Loading raw dataset from cache: {raw_cache_dir}")
        dataset = load_from_disk(raw_cache_dir)

    # If no cached dataset, create it
    if dataset is None:
        logger.info(f"Creating new tokenized dataset for {dataset_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # add pad token id if not present
        if tokenizer.pad_token is None:
            print("No pad token found, setting to eos token")
            tokenizer.pad_token = tokenizer.eos_token

        df = load_default_data(dataset_name, **kwargs)
        # convert from messages to text loss
        df["loss_texts"] = df["messages"].apply(
            lambda x: messages_to_loss_texts(x, tokenizer, format)
        )

        # tokenize

        if drop_additional_columns:
            df = df[["loss_texts", "name"]]
            df["loss_texts"] = df["loss_texts"].apply(
                lambda x: [
                    {"text": a["text"], "compute_loss": a["compute_loss"]} for a in x
                ]
            )

        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(
            tokenize_example,
            fn_kwargs={
                "processing_class": tokenizer,
                "loss_on_eos": False,
                "max_length": None,  # Don't truncate during tokenization - we'll do it in processing
            },
        )

        # TAYLOR - add in example_inds if not present
        dataset = add_example_inds(dataset)

        # Save raw dataset
        if save:
            os.makedirs(raw_cache_dir, exist_ok=True)
            dataset.save_to_disk(raw_cache_dir)
            logger.info(f"Saved raw dataset to cache: {raw_cache_dir}")

    # TAYLOR - here, probably needs to be where we add in the fixed_example_number examples thing - should probably either take in example_inds and max_k and truncate accordingly, or take in max_k and truncate accordingly.

    if keep_inds is not None and not fixed_example_number:
        # TAYLOR - may want to enable this later, but removing for now to help w/ error catching
        breakpoint()
        raise ValueError("keep_inds cannot be provided without fixed_example_number")

    if keep_inds is not None:
        dataset = dataset.select(keep_inds)

    if max_length and fixed_example_number:
        raise ValueError("max_length and max_k cannot be provided together")

    # If max_length is provided, apply processing and cache result
    if max_length:
        logger.info(f"Applying processing for max_length={max_length}")
        processed_dataset = truncate_to_max_length(dataset, max_length)

        # Validate that all examples have labels
        min_labels = min(processed_dataset["n_labels"])
        if min_labels < 1:
            raise ValueError(
                f"Dataset {dataset_name} has examples with no labels after processing"
            )

        logger.info(f"Minimum n_labels in processed dataset: {min_labels}")

        # Cache processed dataset
        if save:
            cache_processed_dataset(
                dataset=processed_dataset,
                dataset_name=dataset_name,
                model_name=model_name,
                max_length=max_length,
                format=format,
                cache_parent_dir=cache_parent_dir,
            )

        return processed_dataset

    if fixed_example_number:
        # We have to assume here that the max_k isn't ridiculously long for length, since hopefully it has been decided taking length into account
        # REMOVE EXTRANEOUS INPUT INDS AND LABELS
        processed_dataset = truncate_to_fixed_examples(dataset, fixed_example_number)

        if save:
            cache_processed_dataset(
                dataset=processed_dataset,
                dataset_name=dataset_name,
                model_name=model_name,
                fixed_example_number=fixed_example_number,
                format=format,
                cache_parent_dir=cache_parent_dir,
            )

        print(f"First input_ids: {processed_dataset[0]['input_ids'][:50]}")
        return processed_dataset
        # find the last ind of the kth example ind that is not -1, drop everything after that

    print(f"First input_ids: {dataset[0]['input_ids'][:50]}")
    return dataset


def load_tokenized_datasets(
    dataset_names: List[str],
    force_reload: bool = False,
    model_name: str = None,
    save=True,
    max_length: Optional[int] = None,
    ignore_missing_datasets=False,
    **kwargs,
) -> Dataset:
    """Load tokenized datasets and return combined dataset.

    Args:
        dataset_names: List of dataset names to load
        force_reload: Force reload from source data
        model_name: Model name for tokenizer and cache naming
        save: Whether to save to cache
        max_length: If provided, apply truncating and filtering
        **kwargs: Additional arguments (format flags, is_chat, etc.)

    Returns:
        Combined dataset with all requested datasets
    """
    datasets = []
    for dataset_name in dataset_names:
        try:
            dataset = load_tokenized_dataset(
                dataset_name,
                force_reload,
                model_name,
                save,
                max_length=max_length,
                **kwargs,
            )
            datasets.append(dataset)
        except Exception as e:
            if ignore_missing_datasets:
                logger.warning(f"Error loading dataset {dataset_name}: {e}")
                continue
            else:
                raise e

    return concatenate_datasets(datasets)
