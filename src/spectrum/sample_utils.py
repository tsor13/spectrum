import difflib
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from trl.trainer.utils import pad


def tokenize_loss_texts(loss_texts, processing_class, loss_on_eos=False):
    # TODO - this is currently functional but a little slow. A couple of ideas to try could be
    # to either split the loss_texts into a batch and then concat everythnig after, or do a whole batch
    # of "loss_texts" inputs at a time. For now, leaving as is and caching though
    # should accept either text or
    # if a string
    if loss_on_eos:
        raise ValueError("Loss on EOS is not currently supported.")
    if isinstance(loss_texts, str):
        processed = processing_class(text=loss_texts)
    else:
        # should be list of dicts with keys "text" (str) and "compute_loss" (bool)
        all_processed = []
        all_loss_texts = ""
        example_inds = []
        dataset_inds = []
        example_ind = 0
        for i, item in enumerate(loss_texts):
            processed = processing_class(text=item["text"])
            # make sure BOS token is only on first text segment
            if i != 0 and processing_class.bos_token_id == processed["input_ids"][0]:
                # truncate to remove first
                processed["input_ids"] = processed["input_ids"][1:]
                processed["attention_mask"] = processed["attention_mask"][1:]
            # for now, manually remove eos if ends with it
            if processed["input_ids"][
                -1
            ] == processing_class.eos_token_id and not processing_class.eos_token in [
                "<|eot_id|>",
                "<|im_end|>",
                "<end_of_turn>",
            ]:
                processed["input_ids"] = processed["input_ids"][:-1]
                processed["attention_mask"] = processed["attention_mask"][:-1]
            if item["compute_loss"]:
                processed["labels"] = processed["input_ids"]
            else:
                processed["labels"] = [-100] * len(
                    processed["input_ids"]
                )  # -100 mask value
            # if all_processed is not empty, check if starts with bos and if so, remove
            if all_processed:
                if processed["input_ids"][0] == processing_class.bos_token_id:
                    processed["input_ids"] = processed["input_ids"][1:]
                    processed["attention_mask"] = processed["attention_mask"][1:]
                    processed["labels"] = processed["labels"][1:]
            all_processed.append(processed)
            all_loss_texts += item["text"]
            this_num = -1
            # TAYLOR - TODO - maybe loosen this restriction for data_id to be present?
            if "example_ind" in item.keys():
                if item["example_ind"] is not None:
                    this_num = item["example_ind"]
            example_inds.extend([this_num] * len(processed["input_ids"]))
            dataset_ind = -1
            if "data_id" in item.keys():
                if item["data_id"] is not None:
                    dataset_ind = item["data_id"]
            dataset_inds.extend([dataset_ind] * len(processed["input_ids"]))
        try:
            processed = all_processed[0].copy()
        except:
            breakpoint()
            pass
        # concat everythnig together
        processed["input_ids"] = [
            item
            for sublist in [p["input_ids"] for p in all_processed]
            for item in sublist
        ]
        processed["attention_mask"] = [
            item
            for sublist in [p["attention_mask"] for p in all_processed]
            for item in sublist
        ]
        processed["labels"] = [
            item for sublist in [p["labels"] for p in all_processed] for item in sublist
        ]
        processed["example_inds"] = example_inds
        processed["data_ids"] = dataset_inds

        # also tokenize all_loss_texts and make sure same length
        processed_all = processing_class(text=all_loss_texts)
        if len(processed_all["input_ids"]) != len(processed["input_ids"]):
            # raise a warning
            warnings.warn(
                f"All loss_texts are not the same length as the first text. Please check your dataset. {len(processed_all['input_ids'])} != {len(processed['input_ids'])}"
            )
            # get the diff between the two
            all_text = processing_class.decode(
                processed_all["input_ids"], skip_special_tokens=False
            )
            processed_text = processing_class.decode(
                processed["input_ids"], skip_special_tokens=False
            )
            # get the diff between the two
            # github style diff
            diff = difflib.unified_diff(
                all_text.splitlines(), processed_text.splitlines()
            )
            # to string
            diff_str = "\n".join(diff)
            print("Diff between loss_texts:")
            print(diff_str)

            # diff between tokenized loss_texts, with \n
            all_tokens_str = "\n".join([str(s) for s in processed_all["input_ids"]])
            processed_tokens_str = "\n".join([str(s) for s in processed["input_ids"]])
            # get the diff between the two
            token_diff = difflib.unified_diff(
                all_tokens_str.splitlines(), processed_tokens_str.splitlines()
            )
            # to string
            token_diff_str = "\n".join(token_diff)
            print("Diff between tokenized loss_texts:")
            print(token_diff_str)
            print(
                f"All loss_texts are not the same length as the first text. Please check your dataset. {len(processed_all['input_ids'])} != {len(processed['input_ids'])}"
            )
            print(f"Diff: {all_text} != {processed_text}")
            print("Text difference:")
            # breakpoint() # COULD ADD BACK IN FOR DEBUGGING

    if (
        processing_class.eos_token_id is not None
        and processed["input_ids"][-1] != processing_class.eos_token_id
    ):
        processed["input_ids"] = processed["input_ids"] + [
            processing_class.eos_token_id
        ]
        processed["example_inds"] = processed["example_inds"] + [-1]
        processed["attention_mask"] = processed["attention_mask"] + [1]
        if processed["labels"] is not None:
            if loss_on_eos:
                processed["labels"] = processed["labels"] + [
                    processing_class.eos_token_id
                ]
            else:
                processed["labels"] = processed["labels"] + [-100]
        if "data_ids" in processed:
            processed["data_ids"] = processed["data_ids"] + [-1]
    return processed


def tokenize_example(example, processing_class, loss_on_eos=False, max_length=None):
    if max_length is None:
        max_length = 8192
    loss_texts = example["loss_texts"]
    processed = tokenize_loss_texts(loss_texts, processing_class, loss_on_eos)
    # add back in columns
    for key in example.keys():
        if key not in processed.keys():
            processed[key] = example[key]
    for key in processed.keys():
        # if list, truncate
        if isinstance(processed[key], list):
            processed[key] = processed[key][:max_length]
    return processed


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


@dataclass
class MyDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling data. Inputs are dynamically padded to the maximum length of a batch if
    they are not all of the same length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        completion_only_loss (`bool`, *optional*, defaults to `True`):
            When the input contains a completion mask (`completion_mask`), the labels are set to -100 for the tokens
            that are no in the completion.
        pad_to_multiple_of (`int` or `None`, *optional*, defaults to `None`):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import DataCollatorForLanguageModeling
    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0)
    >>> examples = [
    ...     {"input_ids": [1, 2, 3]},
    ...     {"input_ids": [4, 5]}
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'labels': tensor([[   1,    2,    3],
                       [   4,    5, -100]])}
    >>> # With completion mask
    >>> examples = [
    ...     {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
    ...     {"input_ids": [4, 5], "completion_mask": [0, 1]}
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'labels': tensor([[-100,    2,    3],
                       [-100,    5, -100]])}
    ```
    """

    pad_token_id: int
    completion_only_loss: bool = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(
        self, examples: list[Union[list[int], Any, dict[str, Any]]]
    ) -> dict[str, Any]:
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        attention_mask = [torch.ones_like(input_ids) for input_ids in input_ids]
        if self.completion_only_loss:
            labels = [torch.tensor(example["labels"]) for example in examples]
        else:
            labels = [torch.tensor(example["input_ids"]) for example in examples]
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = [
                torch.tensor(example["completion_mask"]) for example in examples
            ]
        if "data_ids" in examples[0]:
            data_ids = [torch.tensor(example["data_ids"]) for example in examples]
        if "example_inds" in examples[0]:
            example_inds = [
                torch.tensor(example["example_inds"]) for example in examples
            ]

        # Pad
        output = {}
        output["input_ids"] = pad(
            input_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
        )
        output["attention_mask"] = pad(
            attention_mask,
            padding_value=0,
            padding_side="right",
        )
        output["labels"] = pad(
            labels,
            padding_value=-100,
            padding_side="right",
        )
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = pad(
                completion_mask,
                padding_value=0,
                padding_side="right",
            )
            output["labels"][
                completion_mask == 0
            ] = -100  # mask everything that is not in the completion
        if "data_ids" in examples[0]:
            output["data_ids"] = pad(
                data_ids,
                padding_value=-1,
                padding_side="right",
            )
        if "example_inds" in examples[0]:
            output["example_inds"] = pad(
                example_inds,
                padding_value=-1,
                padding_side="right",
            )
        if "dataset" in examples[0]:
            output["dataset"] = [example["dataset"] for example in examples]

        return output
