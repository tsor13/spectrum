import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spectrum.diverse_valid.llm_judge import LLMJudge
from spectrum.lm_tools import (
    all_logprobs,
    get_logprobs,
    get_restrict_tokens,
    prepare_batch,
)


class GenerationTask(ABC):
    def __init__(
        self,
        name: str,
        examples: Optional[List[str] | List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int | None = None,
    ):
        self.name = name
        self.examples = examples
        self.description = description
        self._provided_messages = messages
        # make sure that either examples or description is provided
        if examples is None and description is None and messages is None:
            raise ValueError(
                "Either examples, description, or messages must be provided."
            )
        if messages is not None:
            if examples is not None or description is not None:
                raise ValueError(
                    "If messages are provided, examples and description must be None."
                )
            # store a default copy so existing code continues to work
            self.messages = [message.copy() for message in messages]
        else:
            self.messages = self.description_examples_to_messages(description, examples)
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else 128

    def description_examples_to_messages(
        self, description: str | None, examples: List[str] | List[Dict[str, Any]] | None
    ) -> List[Dict[str, Any]]:
        messages = []
        if description is not None:
            messages.append({"role": "description", "content": description})
        if examples is not None:
            for example in examples:
                if isinstance(example, str):
                    messages.append({"role": "output", "content": example})
                elif isinstance(example, Dict):
                    # make sure that the example has input and output
                    if "input" not in example:
                        raise ValueError(
                            "Each example must have input if provided as a dictionary."
                        )
                    messages.append({"role": "input", "content": example["input"]})
                    if "output" in example:
                        messages.append(
                            {"role": "output", "content": example["output"]}
                        )
        return messages

    def get_messages(
        self,
        include_description: bool = True,
        include_examples: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return prompt messages with optional components."""
        if self._provided_messages is not None:
            return [message.copy() for message in self._provided_messages]

        description = self.description if include_description else None
        examples = self.examples if include_examples else None
        return self.description_examples_to_messages(description, examples)

    def validate(self, text: str) -> bool:
        """
        Determine if a text is valid for this task.
        Wraps _validate_impl with error handling.
        """
        try:
            return self._validate_impl(text)
        except Exception as e:
            print(
                f"Warning: Validation error for task '{self.name}' with text '{text[:50]}...': {e}"
            )
            return False

    @abstractmethod
    def _validate_impl(self, text: str) -> bool:
        """
        Implementation of validation logic. Should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def validate_texts(self, texts: List[str]) -> List[bool]:
        """
        Batch validate a list of texts. If more efficient batching is possible, this should be overridden.
        """
        return [self.validate(text) for text in texts]

    def test_validation(
        self, valid_cases: List[str], invalid_cases: List[str], verbose: bool = True
    ) -> bool:
        """
        Test the validation function with provided test cases.

        Args:
            valid_cases: List of strings that should validate as True
            invalid_cases: List of strings that should validate as False
            verbose: Whether to print detailed results

        Returns:
            True if all tests pass, False otherwise
        """
        all_passed = True

        # Test valid cases
        for case in valid_cases:
            try:
                result = self.validate(case)
                if result:
                    if verbose:
                        print(f"  ✅ Valid: '{case}' -> {result}")
                else:
                    if verbose:
                        print(f"  ❌ Valid: '{case}' -> {result} (expected True)")
                    all_passed = False
            except Exception as e:
                if verbose:
                    print(f"  💥 Valid: '{case}' -> Exception: {e}")
                all_passed = False

        # Test invalid cases
        for case in invalid_cases:
            try:
                result = self.validate(case)
                if not result:
                    if verbose:
                        print(f"  ✅ Invalid: '{case}' -> {result}")
                else:
                    if verbose:
                        print(f"  ❌ Invalid: '{case}' -> {result} (expected False)")
                    all_passed = False
            except Exception as e:
                if verbose:
                    print(f"  💥 Invalid: '{case}' -> Exception: {e}")
                all_passed = False

        # Summary
        total_tests = len(valid_cases) + len(invalid_cases)
        if verbose:
            if all_passed:
                print(f"✅ All {total_tests} validation tests passed!")
            else:
                print(f"❌ Some validation tests failed ({total_tests} total tests)")

        return all_passed


class InclusionTask(GenerationTask):
    def __init__(
        self,
        name: str = "inclusion_task",
        examples: Optional[List[str] | List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int | None = None,
        valid_strings: Optional[List[str]] = None,
    ):
        super().__init__(name, examples, description, messages, max_new_tokens)
        if valid_strings is None:
            raise ValueError("valid_strings must be provided for InclusionTask.")
        self.valid_strings = valid_strings

    def _validate_impl(self, text: str) -> bool:
        return text in self.valid_strings


class FunctionTask(GenerationTask):
    def __init__(
        self,
        name: str,
        examples: Optional[List[str] | List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int | None = None,
        validation_fn: Callable[[str], bool] = None,
    ):
        super().__init__(name, examples, description, messages, max_new_tokens)
        if validation_fn is None:
            raise ValueError("validation_fn must be provided for LambdaTask.")
        self.validation_fn = validation_fn

    def _validate_impl(self, text: str) -> bool:
        return self.validation_fn(text)


class LLMJudgeTask(GenerationTask):
    """
    Generic LLM-as-judge task that takes an LLMJudge instance as a parameter.
    This provides maximum flexibility by allowing direct control over the judge.
    """

    def __init__(
        self,
        name: str,
        llm_judge: LLMJudge,
        examples: Optional[List[str] | List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int | None = None,
    ):
        """
        Initialize the LLMJudgeTask.

        Args:
            name: Task name
            llm_judge: LLMJudge instance to use for validation
            examples: Examples of valid outputs
            description: Description of the task
            messages: Pre-formatted messages
            max_new_tokens: Maximum tokens for generation
        """
        super().__init__(name, examples, description, messages, max_new_tokens)
        self.llm_judge = llm_judge

    def _validate_impl(self, text: str) -> bool:
        """
        Single text validation using the judge model.
        Note: This is inefficient for batch validation - use validate_texts instead.
        """
        return self.validate_texts([text])[0]

    def validate_texts(self, texts: List[str]) -> List[bool]:
        """
        Efficient batch validation using the judge model.

        Args:
            texts: List of texts to validate

        Returns:
            List of boolean validity judgments
        """
        return self.llm_judge.judge_generations(texts)


class AutomaticLLMJudgeTask(GenerationTask):
    """
    Automatic LLM-as-judge task that creates an LLMJudge based on description and examples.
    This provides the original automatic functionality with task context formatting.
    """

    def __init__(
        self,
        name: str,
        examples: Optional[List[str] | List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int | None = None,
        judge_model_name: Optional[str] = None,
        judge_model: Optional[AutoModelForCausalLM] = None,
        judge_tokenizer: Optional[AutoTokenizer] = None,
        judge_prompt_template: Optional[List[Dict[str, str]]] = None,
        valid_text: str = "Valid",
        invalid_text: str = "Invalid",
        validity_threshold: float = 0.5,
        batch_size: int = 8,
    ):
        """
        Initialize the AutomaticLLMJudgeTask.

        Args:
            name: Task name
            examples: Examples of valid outputs
            description: Description of the task
            messages: Pre-formatted messages
            max_new_tokens: Maximum tokens for generation
            judge_model_name: Model name for lazy loading (mutually exclusive with judge_model/judge_tokenizer)
            judge_model: Pre-loaded model (mutually exclusive with judge_model_name)
            judge_tokenizer: Pre-loaded tokenizer (must be provided with judge_model)
            judge_prompt_template: Custom prompt template as chat messages (default uses automatic template)
            valid_text: Token representing valid judgment (default: "Valid")
            invalid_text: Token representing invalid judgment (default: "Invalid")
            validity_threshold: Threshold for p(valid) vs p(invalid) (default: 0.5)
            batch_size: Batch size for judge processing (default: 8)
        """
        super().__init__(name, examples, description, messages, max_new_tokens)

        # Create prompt template with task context
        self.judge_prompt_template = (
            judge_prompt_template or self._get_default_prompt_template()
        )
        prompt_template = self._create_prompt_template()

        # Create LLMJudge instance
        self.llm_judge = LLMJudge(
            prompt_template=prompt_template,
            valid_text=valid_text,
            invalid_text=invalid_text,
            validity_threshold=validity_threshold,
            judge_model_name=judge_model_name,
            model=judge_model,
            tokenizer=judge_tokenizer,
            batch_size=batch_size,
        )

    def _get_default_prompt_template(self) -> List[Dict[str, str]]:
        """Get the default prompt template for the LLM judge as chat messages."""
        return [
            {
                "role": "system",
                "content": 'You are an LLM-as-judge whose goal is to judge whether a piece of text is valid or not valid. You will be given some combination of a description (either vague or general), which will describe the kind of desired output, and examples of "in-class" good generations. If there is no description, base your judgment only on the examples - anything that seems like it was generated by the same process that made the examples is valid, and anything that does not is invalid.\n\nYou will be then be shown the candidate text, and be asked to judge whether or not it is "Valid" or "Invalid". ONLY respond with "Valid" or "Invalid".',
            },
            {
                "role": "user",
                "content": "{task_context}\n\nCandidate Text: {generation}",
            },
        ]

    def _format_task_context(self) -> str:
        """Format the task context (description and examples) for the judge prompt."""
        context_parts = []

        if self.description:
            context_parts.append(f"Description: {self.description}")

        if self.examples:
            for i, example in enumerate(self.examples, 1):
                if isinstance(example, str):
                    context_parts.append(f"Example {i}: {example}")
                elif isinstance(example, dict) and "output" in example:
                    context_parts.append(f"Example {i}: {example['output']}")

        return "\n".join(context_parts) if context_parts else ""

    def _create_prompt_template(self) -> List[Dict[str, str]]:
        """Create the prompt template with task context baked in."""
        task_context = self._format_task_context()

        # Create a template with task context pre-filled but keep {generation} placeholder
        template_with_context = []
        for msg in self.judge_prompt_template:
            content = msg["content"]
            if "{task_context}" in content:
                # Replace task_context but preserve {generation}
                content = content.replace("{task_context}", task_context)
            template_with_context.append({"role": msg["role"], "content": content})

        return template_with_context

    def _validate_impl(self, text: str) -> bool:
        """
        Single text validation using the judge model.
        Note: This is inefficient for batch validation - use validate_texts instead.
        """
        return self.validate_texts([text])[0]

    def validate_texts(self, texts: List[str]) -> List[bool]:
        """
        Efficient batch validation using the judge model.

        Args:
            texts: List of texts to validate

        Returns:
            List of boolean validity judgments
        """
        return self.llm_judge.judge_generations(texts)


if __name__ == "__main__":
    task = InclusionTask(
        name="rng_10",
        description="Generate a number between 1 and 10.",
        valid_strings=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    )
    print(task.validate("1"))
    print(task.validate("11"))

    # Example usage of AutomaticLLMJudgeTask (original functionality)
    # judge_task = LLMJudgeTask(
    #     name="number_judgment",
    #     description="Generate a number between 1 and 10.",
    #     examples=["5", "7", "2"],
    #     judge_model_name="google/gemma-3-1b-it"
    # )

    # PASSED w/ gemma 1b it
    # # Test batch validation
    # test_texts = ["5", "10", "42", "3", "invalid"]
    # results = judge_task.validate_texts(test_texts)
    # print(f"Validation results: {results}")

    # Example usage of LLMJudgeTask
    # judge_task = LLMJudgeTask(
    #     name="haiku",
    #     description="Generate a haiku.",
    #     judge_model_name="google/gemma-3-1b-it"
    # )

    # PASSED w/ gemma 1b it
    # # Test batch validation
    # test_texts = [
    #     "An old silent pond... / A frog jumps into the pond, / Splash! Silence again.",
    #     "An old silent pond...\nA frog jumps into the pond,\nSplash! Silence again.",
    #     "This is a poem.",
    #     "Apples are red.",
    #     "This will\nhave three\nlines and be haiku",
    # ]
    # results = judge_task.validate_texts(test_texts)
    # print(f"Validation results: {results}")

    # # FAILED w/ gemma 1b it
    # # WORKS w/ gemma 12b it
    # judge_task = LLMJudgeTask(
    #     name="gerund",
    #     description="Gerund",
    #     examples=[
    #         "Waiting",
    #         "Hoisting",
    #     ],
    #     judge_model_name="google/gemma-3-12b-it"
    # )

    # # Test batch validation
    # test_texts = [
    #     "The dog is running.", # invalid
    #     "Schlepping", # valid
    #     "Thinking", # valid
    #     "Walk", # invalid
    #     "This is a sentence.", # invalid
    # ]
    # results = judge_task.validate_texts(test_texts)
    # print(f"Validation results: {results}")

    # # 12b-it MOSTLY works at this point, but gets "Walk" wrong
    # judge_task = LLMJudgeTask(
    #     name="gerund",
    #     # description="Gerund",
    #     examples=[
    #         "Waiting",
    #         "Hoisting",
    #     ],
    #     # judge_model_name="google/gemma-3-12b-it"
    #     judge_model_name="google/gemma-3-27b-it"
    # )

    # # Test batch validation
    # test_texts = [
    #     "The dog is running.", # invalid
    #     "Schlepping", # valid
    #     "Thinking", # valid
    #     "Walk", # invalid
    #     "This is a sentence.", # invalid
    # ]
    # results = judge_task.validate_texts(test_texts)
    # print(f"Validation results: {results}")

    # 12b-it MOSTLY works at this point, but gets "Walk" wrong
    judge_task = AutomaticLLMJudgeTask(
        name="metaphor",
        description="Generate a metaphor.",
        # examples=[
        #     "The sun is a ball of fire.",
        #     "The sun is a ball of fire.",
        # ],
        judge_model_name="google/gemma-3-1b-it",
        # judge_model_name="google/gemma-3-12b-it"
        # judge_model_name="google/gemma-3-27b-it"
    )

    # Test batch validation
    test_texts = [
        "Time is a river.",  # valid
        "Time is a ticking clock",  # valid
        "The sun is a flying bird.",  # valid
        "Time is like a river.",  # invalid, simile
        "The sun is a ball of fire.",  # invalid, literal
    ]
    results = judge_task.validate_texts(test_texts)
    print(f"Validation results: {results}")

    # Example usage of generic LLMJudgeTask with custom LLMJudge instance
    # def custom_template(generation: str) -> List[Dict[str, str]]:
    #     return [
    #         {"role": "system", "content": "Decide whether the text is a color or not a color."},
    #         {"role": "user", "content": f"Is '{generation}' a valid color? Respond only with 'Color' or 'Not a color'."}
    #     ]
    custom_template = [
        {
            "role": "system",
            "content": "Decide whether the text is a color or not a color.",
        },
        {
            "role": "user",
            "content": "Is '{generation}' a valid color? Respond only with 'Color' or 'Not a color'.",
        },
    ]

    custom_judge = LLMJudge(
        prompt_template=custom_template,
        valid_text="Color",
        invalid_text="Not a color",
        judge_model_name="google/gemma-3-1b-it",
    )

    custom_judge_task = LLMJudgeTask(
        name="color_judge", description="Generate a color name.", llm_judge=custom_judge
    )

    # test batch validation
    test_texts = [
        "Red",
        "Yellow",
        "Purple Mountain Majesty",
        "Puce",
        "Car",
        "Bcekham",
    ]
    results = custom_judge_task.validate_texts(test_texts)
    print(f"Validation results: {results}")
