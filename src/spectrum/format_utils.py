"""
uv run format_utils.py --tokenizer_name Qwen/Qwen3-14B --format chat_simple
uv run format_utils.py --tokenizer_name Qwen/Qwen3-14B --format spectrum
uv run format_utils.py --tokenizer_name meta-llama/Llama-3.1-8B-Instruct --format spectrum
uv run format_utils.py --tokenizer_name google/gemma-3-1b-it --format colon
"""

import re
from typing import Any, Callable, Dict, List

from transformers import AutoTokenizer

model_params = {
    "gemma-3": {
        "spectrum": {
            "description_map": lambda x: [
                {
                    "text": "<start_of_turn>description\n" + x + "<end_of_turn>\n",
                    "compute_loss": False,
                }
            ],
            "input_map": lambda x: [
                {
                    "text": "<start_of_turn>input\n" + x + "<end_of_turn>\n",
                    "compute_loss": False,
                }
            ],
            "output_map": lambda x: [
                {
                    "text": "<start_of_turn>output\n",
                    "compute_loss": False,
                },
                {
                    "text": x + "<end_of_turn>",
                    "compute_loss": True,
                },
                {
                    "text": "\n",
                    "compute_loss": False,
                },
            ],
            "start_generation": "<start_of_turn>output\n",
            "stop_generation": "<end_of_turn>",
        },
        "chat": {
            "model_start_text": "<start_of_turn>model\n",
            "end_string": "<end_of_turn>",
            "extra_mappings": [
                {
                    "key": "<bos>",
                    "value": "",
                },  # FOR GEMMA TO MAKE SURE THAT IT DOESN'T ADD A BOS TOKEN
            ],
        },
    },
    # Design decisions: Qwen instruct models put 100% weight on <think> token if not included. For now, we add it in for each instance for chat, but omit it for spectrum. This could be subject to change if the loss is super high for the model on output.
    "Qwen3": {
        "spectrum": {
            "description_map": lambda x: [
                {
                    "text": "<|im_start|>description\n" + x + "<|im_end|>\n",
                    "compute_loss": False,
                }
            ],
            "input_map": lambda x: [
                {
                    "text": "<|im_start|>input\n" + x + "<|im_end|>\n",
                    "compute_loss": False,
                }
            ],
            "output_map": lambda x: [
                {
                    "text": "<|im_start|>output\n",
                    "compute_loss": False,
                },
                {
                    "text": x + "<|im_end|>",
                    "compute_loss": True,
                },
                {
                    "text": "\n",
                    "compute_loss": False,
                },
            ],
            "start_generation": "<|im_start|>output\n",
            "stop_generation": "<|im_end|>",
        },
        "chat": {
            "model_start_text": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "end_string": "<|im_end|>",
            "extra_mappings": [
                # remove think
                {
                    "key": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    "value": "<|im_start|>assistant\n",
                },
                # add think back in for each instance
                {
                    "key": "<|im_start|>assistant\n",
                    "value": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                },
            ],
        },
    },
    # Design decisions: Currently for chat, the date is included in the system prompt. To maintain parity, including for now.
    "Llama-3.1": {
        "spectrum": {
            "description_map": lambda x: [
                {
                    "text": "<|start_header_id|>description<|end_header_id|>\n\n"
                    + x
                    + "<|eot_id|>",
                    "compute_loss": False,
                }
            ],
            "input_map": lambda x: [
                {
                    "text": "<|start_header_id|>input<|end_header_id|>\n\n"
                    + x
                    + "<|eot_id|>",
                    "compute_loss": False,
                }
            ],
            "output_map": lambda x: [
                {
                    "text": "<|start_header_id|>output<|end_header_id|>\n\n",
                    "compute_loss": False,
                },
                {
                    "text": x + "<|eot_id|>",
                    "compute_loss": True,
                },
            ],
            "start_generation": "<|start_header_id|>output<|end_header_id|>\n\n",
            "stop_generation": "<|eot_id|>",
        },
        "chat": {
            "model_start_text": "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "end_string": "<|eot_id|>",
            "extra_mappings": [],
        },
    },
}


def messages_to_spectrum_loss_texts(
    messages: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    start_generation: bool = False,
) -> List[Dict[str, Any]]:
    model_name = tokenizer.name_or_path
    spectrum_params = None
    # iterate through keys in model_params, check if key is in model_name
    for key in model_params:
        if key in model_name:
            spectrum_params = model_params[key]["spectrum"]
            break
    if spectrum_params is None:
        raise ValueError(
            f"Model {model_name} not found in model_params. Supported models: {list(model_params.keys())}"
        )

    texts = []
    has_description = False
    first_output = True

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "description":
            has_description = True
            texts.extend(spectrum_params["description_map"](content))
        elif role == "input":
            texts.extend(spectrum_params["input_map"](content))
        elif role == "output":
            out_texts = spectrum_params["output_map"](content)
            if first_output and not has_description:
                # set compute_loss to False for all
                for text in out_texts:
                    text["compute_loss"] = False
            texts.extend(out_texts)
            first_output = False
        else:
            raise ValueError(
                f"Unknown role: {role}. Must be description, input, or output."
            )

    # Add generation prompt if start_generation is True
    if start_generation:
        texts.extend(
            [{"text": spectrum_params["start_generation"], "compute_loss": False}]
        )

    return texts


def to_raw_text(texts: List[Dict[str, Any]]) -> str:
    return "".join([text["text"] for text in texts])


def show_loss_texts(texts: List[Dict[str, Any]], sep: str = "____") -> str:
    # Output the raw texts appended together, with ____ at the beginning and end of each calculate loss
    all_text = ""
    currently_loss = False
    for text in texts:
        # if no loss and not currently loss, just add
        if not text["compute_loss"] and not currently_loss:
            all_text += text["text"]
        # if loss and not currently loss, add sep and text
        elif text["compute_loss"] and not currently_loss:
            all_text += sep + text["text"]
            currently_loss = True
        # if no loss and currently loss, add sep and text
        elif not text["compute_loss"] and currently_loss:
            all_text += sep + text["text"]
            currently_loss = False
        # if loss and currently loss, add text
        elif text["compute_loss"] and currently_loss:
            all_text += text["text"]
            currently_loss = False
    if currently_loss:
        all_text += sep
    return all_text


def messages_to_spectrum_raw_text(
    messages: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    start_generation: bool = False,
) -> str:
    """Convert spectrum messages to raw text by flattening the result of messages_to_spectrum_loss_texts."""
    texts = messages_to_spectrum_loss_texts(messages, tokenizer, start_generation)
    return to_raw_text(texts)


default_chat_prompt = """You are tasked with generating outputs from a particular, potentially stochastic, generative process. You will be given some combination of:
- Description: A natural description of the generative process / data distribution
- Input: An input on which to condition the generative process.
- Example outputs: Example outputs from the process, either in a user message or as prior generations from a chat message. You may assume that any given outputs are exchangeable with one another (order-invariant) and generated from the same process (roughly i.i.d.). If the output data pertains to a single object, it just contains the output. If it contains multiple objects, use json formatting with keys for the name of the output variable.
You will be provided at least either a description or an example output.

Given these components, your job is to generate JUST the output in your response, roughly approximating the underlying generative process, maintaining any underlying stochasticity (if any is present). If you are asked to generate again, you will either be given an additional input to condition on, or will just be told to "Generate".
"""


def messages_to_chat_loss_texts(
    messages: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    start_generation: bool = False,
    chat_prompt: str | None = None,
    default_user_message: str | None = None,
    delim=None,
    description_lambda: Callable[[str], str] | None = None,
    example_input_lambda: Callable[[str], str] | None = None,
    example_output_lambda: Callable[[str], str] | None = None,
    all_examples_in_prompt: bool = False,
) -> List[Dict[str, Any]]:
    texts = []
    chat_messages = []

    if chat_prompt is None:
        chat_prompt = default_chat_prompt
    if default_user_message is None:
        default_user_message = "Generate."
    if delim is None:
        delim = "\n\n"
    if description_lambda is None:
        description_lambda = lambda x: "Description: " + x
    if example_input_lambda is None:
        example_input_lambda = lambda x: "Example Input: " + x
    if example_output_lambda is None:
        example_output_lambda = lambda x: "Example Output: " + x

    # SYSTEM MESSAGE
    system_message = chat_prompt
    has_description_or_output = False
    has_input = False
    if not all_examples_in_prompt:
        for message in messages:
            if message["role"] == "description":
                system_message += delim + description_lambda(message["content"])
                if system_message.strip() != "":
                    chat_messages.append(
                        {"role": "system", "content": system_message.strip()}
                    )
                has_description_or_output = True
            elif message["role"] == "input":
                has_input = True
                if not has_description_or_output:
                    system_message += delim + example_input_lambda(message["content"])
                else:
                    chat_messages.append(
                        {"role": "user", "content": message["content"]}
                    )
            elif message["role"] == "output":
                if not has_description_or_output:
                    system_message += delim + example_output_lambda(message["content"])
                    if system_message.strip() != "":
                        chat_messages.append(
                            {"role": "system", "content": system_message.strip()}
                        )
                    has_description_or_output = True
                else:
                    if not has_input:
                        chat_messages.append({"role": "user", "content": "Generate"})
                    chat_messages.append(
                        {"role": "assistant", "content": message["content"]}
                    )
    else:
        has_input = False
        has_examples = False
        has_description = False
        for message in messages:
            if message["role"] == "description":
                system_message += delim + description_lambda(message["content"])
                has_description = True
            elif message["role"] == "input":
                system_message += delim + example_input_lambda(message["content"])
                has_examples = True
            elif message["role"] == "output":
                has_examples = True
                system_message += delim + example_output_lambda(message["content"])
        chat_messages.append({"role": "system", "content": system_message.strip()})
        message_to_add = None
        if has_description and has_examples:
            message_to_add = (
                "Now generate an output based on the description and examples."
            )
        elif has_description:
            message_to_add = "Generate something that fits this description."
        elif has_examples:
            message_to_add = "Generate an output like the provided examples."
        else:
            raise ValueError("No description or examples provided.")
        chat_messages.append({"role": "user", "content": message_to_add})
    if len(chat_messages) == 0:
        # add system message
        chat_messages.append({"role": "system", "content": system_message.strip()})
        chat_messages.append({"role": "user", "content": ""})
    if len(chat_messages) == 1:
        chat_messages.append({"role": "user", "content": ""})

    # if last message is output and start_generation is true, add a default user message
    if start_generation and chat_messages[-1]["role"] == "assistant":
        chat_messages.append({"role": "user", "content": default_user_message})

    model_name = tokenizer.name_or_path
    chat_params = None
    # iterate through keys in model_params, check if key is in model_name
    for key in model_params:
        if key in model_name:
            chat_params = model_params[key]["chat"]
            break
    if chat_params is None:
        raise ValueError(
            f"Model {model_name} not found in model_params. Supported models: {list(model_params.keys())}"
        )

    # Apply chat template
    full_text = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=start_generation
    )

    # if extra_mappings is in chat_params, apply it to full_text
    # FOR QWEN TO MAKE SURE THAT IT ALWAYS GENERATES THINK BEFORE RESPONSE
    if "extra_mappings" in chat_params:
        for mapping in chat_params["extra_mappings"]:
            key, value = mapping["key"], mapping["value"]
            full_text = full_text.replace(key, value)

    text_to_split = full_text
    # now, find all places starting with <start_of_turn>model\n
    model_start_text = chat_params["model_start_text"]
    end_string = chat_params["end_string"]
    first = True
    while model_start_text in text_to_split:
        # get location of model_start_text
        model_start_loc = text_to_split.find(model_start_text)
        split_ind = model_start_loc + len(model_start_text)
        text_to_add, text_to_split = (
            text_to_split[:split_ind],
            text_to_split[split_ind:],
        )
        # add to texts
        texts.append({"text": text_to_add, "compute_loss": False})
        # get location of end_string
        end_string_loc = text_to_split.find(end_string)
        end_ind = end_string_loc + len(end_string)
        text_to_add, text_to_split = text_to_split[:end_ind], text_to_split[end_ind:]
        # set compute_loss to True for text up to and including end_string
        if len(text_to_add) > 0:
            texts.append({"text": text_to_add, "compute_loss": True})
        first = False
    if len(text_to_split) > 0:
        texts.append({"text": text_to_split, "compute_loss": False})
    if len(texts) == 0:
        breakpoint()

    return texts


def messages_to_chat_raw_text(
    messages: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    start_generation: bool = False,
    chat_prompt: str | None = None,
    default_user_message: str | None = None,
    delim=None,
    description_lambda: Callable[[str], str] | None = None,
    example_input_lambda: Callable[[str], str] | None = None,
    example_output_lambda: Callable[[str], str] | None = None,
    all_examples_in_prompt: bool = False,
) -> str:
    """Convert chat messages to raw text by flattening the result of messages_to_chat_loss_texts."""
    texts = messages_to_chat_loss_texts(
        messages=messages,
        tokenizer=tokenizer,
        start_generation=start_generation,
        chat_prompt=chat_prompt,
        default_user_message=default_user_message,
        delim=delim,
        description_lambda=description_lambda,
        example_input_lambda=example_input_lambda,
        example_output_lambda=example_output_lambda,
        all_examples_in_prompt=all_examples_in_prompt,
    )
    return to_raw_text(texts)


simple_chat_prompt = ""


# chat simple - same as chat, but with simpler formatting and whatnot
def messages_to_chat_simple_loss_texts(
    messages: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    start_generation: bool = False,
    chat_prompt: str | None = None,
    default_user_message: str | None = None,
    delim=None,
    description_lambda: Callable[[str], str] | None = None,
    example_input_lambda: Callable[[str], str] | None = None,
    example_output_lambda: Callable[[str], str] | None = None,
) -> List[Dict[str, Any]]:
    """Convert chat messages to simple-formatted text with loss computation flags."""
    texts = messages_to_chat_loss_texts(
        messages=messages,
        tokenizer=tokenizer,
        start_generation=start_generation,
        chat_prompt=chat_prompt,
        default_user_message=default_user_message,
        delim=delim,
        description_lambda=description_lambda,
        example_input_lambda=example_input_lambda,
        example_output_lambda=example_output_lambda,
    )
    return texts


def messages_to_colon_loss_texts(
    messages: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    start_generation: bool = False,
    delim="\n\n",
) -> List[Dict[str, Any]]:
    """Convert messages to colon-formatted text with loss computation flags.

    Format: Description: <desc>\nInput: <input>\nOutput: <output>
    Loss is calculated on 'Output: <content>\n' for output messages.
    """
    texts = []
    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "description":
            text = f"Description: {content}" + delim
            texts.append({"text": text, "compute_loss": False, **message})
        elif role == "input":
            text = f"Input: {content}" + delim
            texts.append({"text": text, "compute_loss": False, **message})
        elif role == "output" or role == "assistant":
            # For output, include the "Output: " prefix and ending newline in loss calculation
            # text = f"Output: {content}\n"
            # texts.append({"text": text, "compute_loss": True, **message})
            texts.append({"text": "Output:", "compute_loss": False, **message})
            text = content + delim
            texts.append(
                {"text": " " + text.strip() + delim, "compute_loss": True, **message}
            )
        else:
            # Handle other roles by treating them as descriptions
            text = f"{role.capitalize()}: {content}" + delim
            texts.append({"text": text, "compute_loss": False, **message})

    # Add generation prompt if start_generation is True
    if start_generation:
        gen_prompt = f"Output:"
        texts.append({"text": gen_prompt, "compute_loss": False})

    return texts


def messages_to_colon_raw_text(
    messages: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    start_generation: bool = False,
) -> str:
    """Convert messages to colon-formatted raw text by flattening the result of messages_to_colon_loss_texts."""
    texts = messages_to_colon_loss_texts(messages, tokenizer, start_generation)
    return "".join([text["text"] for text in texts])


simple_chat_args = {
    "chat_prompt": "",
    "description_lambda": lambda x: x,
}


def messages_to_chat_simple_loss_texts(**kwargs):
    # check if any keys overlap - if so, raise an error
    if set(kwargs.keys()) & set(simple_chat_args.keys()):
        raise ValueError(
            f"Keys overlap between kwargs and simple_chat_args: {set(kwargs.keys()) & set(simple_chat_args.keys())}"
        )
    kwargs.update(simple_chat_args)
    return messages_to_chat_loss_texts(**kwargs)


def messages_to_chat_simple_raw_text(**kwargs):
    if set(kwargs.keys()) & set(simple_chat_args.keys()):
        raise ValueError(
            f"Keys overlap between kwargs and simple_chat_args: {set(kwargs.keys()) & set(simple_chat_args.keys())}"
        )
    kwargs.update(simple_chat_args)
    return messages_to_chat_raw_text(**kwargs)


TEMPLATE_MAPS = {
    "spectrum": {
        "messages_to_loss_texts": messages_to_spectrum_loss_texts,
        "messages_to_text": messages_to_spectrum_raw_text,
        "end_string": None,  # depends on tokenizer
    },
    "chat": {
        "messages_to_loss_texts": messages_to_chat_loss_texts,
        "messages_to_text": messages_to_chat_raw_text,
        "end_string": None,  # depends on tokenizer
    },
    "chat_simple": {
        "messages_to_loss_texts": messages_to_chat_simple_loss_texts,
        "messages_to_text": messages_to_chat_simple_raw_text,
        "end_string": None,  # depends on tokenizer
    },
    "colon": {
        "messages_to_loss_texts": messages_to_colon_loss_texts,
        "messages_to_text": messages_to_colon_raw_text,
        "end_string": "\n",
    },
}


def messages_to_loss_texts(
    messages: List[Dict[str, Any]], tokenizer: AutoTokenizer, format: str
) -> List[Dict[str, Any]]:
    if format not in TEMPLATE_MAPS:
        raise ValueError(
            f"Format {format} not found in TEMPLATE_MAPS. Supported formats: {list(TEMPLATE_MAPS.keys())}"
        )
    return TEMPLATE_MAPS[format]["messages_to_loss_texts"](
        messages=messages, tokenizer=tokenizer
    )


if __name__ == "__main__":
    # Simple test
    # read in tokenizer_name from command line, format from command line
    import argparse

    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--format", type=str, required=True)
    args = parser.parse_args()
    tokenizer_name = args.tokenizer_name
    format = args.format

    # check if format is in TEMPLATE_MAPS
    if format not in TEMPLATE_MAPS:
        raise ValueError(
            f"Format {format} not found in TEMPLATE_MAPS. Supported formats: {list(TEMPLATE_MAPS.keys())}"
        )

    # Try using a chat-compatible tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_messages0 = [
        {"role": "description", "content": "DESCRIPTION TEXT"},
        {"role": "input", "content": "INPUT 1 TEXT"},
        {"role": "output", "content": "OUTPUT 1 TEXT"},
        {"role": "input", "content": "INPUT 2 TEXT"},
        {"role": "output", "content": "OUTPUT 2 TEXT"},
        {"role": "input", "content": "INPUT 3 TEXT"},
        {"role": "output", "content": "OUTPUT 3 TEXT"},
    ]

    texts = TEMPLATE_MAPS[format]["messages_to_loss_texts"](
        messages=test_messages0, tokenizer=tokenizer
    )
    print(show_loss_texts(texts))
    print()
    breakpoint()

    test_messages0 = [
        {"role": "description", "content": "DESCRIPTION TEXT"},
        {"role": "output", "content": "OUTPUT 1 TEXT"},
        {"role": "output", "content": "OUTPUT 2 TEXT"},
        {"role": "output", "content": "OUTPUT 3 TEXT"},
    ]
    texts = TEMPLATE_MAPS[format]["messages_to_loss_texts"](
        messages=test_messages0, tokenizer=tokenizer, start_generation=True
    )
    print(show_loss_texts(texts))
    breakpoint()
