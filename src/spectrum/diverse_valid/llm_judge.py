import os
import sys
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMJudge:
    """
    Standalone LLM-as-judge for evaluating text validity using another LLM.
    Handles prompt templating internally and supports flexible model loading.

    This class has no dependencies on GenerationTask and can be used independently
    for any text judging needs.
    """

    def __init__(
        self,
        prompt_template: Union[
            str, List[Dict[str, str]], Callable[[str], List[Dict[str, str]]]
        ],
        valid_text: str,
        invalid_text: str,
        validity_threshold: float = 0.5,
        judge_model_name: Optional[str] = None,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        batch_size: int = 8,
    ):
        """
        Initialize the LLMJudge.

        Args:
            prompt_template: Template for generating prompts. Can be:
                - String with {generation} placeholder
                - List of chat messages with {generation} placeholder
                - Callable that takes generation and returns chat messages
            valid_text: Token representing valid judgment
            invalid_text: Token representing invalid judgment
            validity_threshold: Threshold for p(valid) vs p(invalid) (default: 0.5)
            judge_model_name: Model name for lazy loading (mutually exclusive with model/tokenizer)
            model: Pre-loaded model (mutually exclusive with judge_model_name)
            tokenizer: Pre-loaded tokenizer (must be provided with model)
            batch_size: Batch size for processing (default: 8)
        """
        # Validate input parameters
        if judge_model_name is not None and (
            model is not None or tokenizer is not None
        ):
            raise ValueError("Cannot specify both judge_model_name and model/tokenizer")

        if model is not None and tokenizer is None:
            raise ValueError("Must provide tokenizer when providing model")

        if judge_model_name is None and model is None:
            raise ValueError("Must provide either judge_model_name or model/tokenizer")

        # Store configuration
        self.prompt_template = prompt_template
        self.valid_text = valid_text
        self.invalid_text = invalid_text
        self.validity_threshold = validity_threshold
        self.batch_size = batch_size

        # Model setup
        self.judge_model_name = judge_model_name
        self._model = model
        self._tokenizer = tokenizer
        self._restrict_tokens = None

        # If model/tokenizer provided directly, set up tokens immediately
        if self._model is not None and self._tokenizer is not None:
            self._setup_restrict_tokens()

    def _load_judge_model(self):
        """Lazy load the judge model and tokenizer if needed."""
        if self._model is None and self.judge_model_name is not None:
            print(f"Loading judge model: {self.judge_model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.judge_model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.judge_model_name,
                device_map="auto",
            )
            self._setup_restrict_tokens()

    def _setup_restrict_tokens(self):
        """Set up restrict tokens for valid/invalid tokens."""
        if self._tokenizer is not None:
            valid_invalid_texts = [self.valid_text, self.invalid_text]
            self._restrict_tokens = get_restrict_tokens(
                valid_invalid_texts, self._tokenizer
            )

    def _apply_template(self, generation: str) -> str:
        """Apply the prompt template to a generation."""
        if callable(self.prompt_template):
            # Callable template - should return chat messages
            messages = self.prompt_template(generation)
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif isinstance(self.prompt_template, list):
            # Chat messages template
            messages = []
            for msg in self.prompt_template:
                content = msg["content"].format(generation=generation)
                messages.append({"role": msg["role"], "content": content})

            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # String template
            return self.prompt_template.format(generation=generation)

    def judge_generations(self, generations: List[str]) -> List[bool]:
        """
        Judge multiple generations for validity.

        Args:
            generations: List of text generations to judge

        Returns:
            List of boolean validity judgments (same order as input)
        """
        if not generations:
            return []

        # Check for empty/whitespace-only texts and handle them immediately
        # We'll build the final result list maintaining the exact input order
        final_results = [None] * len(generations)
        non_empty_generations = []
        non_empty_indices = []

        for i, gen in enumerate(generations):
            if not gen or not gen.strip():
                # Always return False for empty or whitespace-only texts
                final_results[i] = False
            else:
                non_empty_generations.append(gen)
                non_empty_indices.append(i)

        # If all generations were empty, return early
        if not non_empty_generations:
            return final_results

        # Ensure model is loaded
        self._load_judge_model()

        # Apply template to each non-empty generation
        judge_prompts = [self._apply_template(gen) for gen in non_empty_generations]

        # Get logprobs for valid/invalid tokens
        try:
            # Use all_logprobs to get probabilities efficiently
            results_df = all_logprobs(
                input_texts=judge_prompts,
                model=self._model,
                tokenizer=self._tokenizer,
                batch_size=min(self.batch_size, len(judge_prompts)),
                restrict_tokens=self._restrict_tokens,
            )

            # Extract validity judgments for non-empty texts
            non_empty_scores = []
            for _, row in results_df.iterrows():
                probs = row["probs"]  # This should be [p(valid_text), p(invalid_text)]
                p_valid = probs[0]  # First token is valid_text
                p_invalid = probs[1]  # Second token is invalid_text

                # Normalize probabilities
                total_prob = p_valid + p_invalid
                if total_prob > 0:
                    p_valid_normalized = p_valid / total_prob
                    non_empty_scores.append(
                        p_valid_normalized > self.validity_threshold
                    )
                else:
                    # Fallback to False if probabilities are invalid
                    non_empty_scores.append(False)

            # Place the non-empty results back in their original positions
            for i, score in enumerate(non_empty_scores):
                original_index = non_empty_indices[i]
                final_results[original_index] = score

            return final_results

        except Exception as e:
            print(f"Error in batch judging: {e}")
            raise e

    def judge_single(self, generation: str) -> bool:
        """
        Judge a single generation for validity.

        Args:
            generation: Text generation to judge

        Returns:
            Boolean validity judgment
        """
        return self.judge_generations([generation])[0]


if __name__ == "__main__":
    # Example usage

    # String template example
    string_template = "Is the following text a valid color name? Text: {generation}. Answer with 'Color' or 'Not a color'.\nAnswer:"

    judge = LLMJudge(
        prompt_template=string_template,
        valid_text=" Color",
        invalid_text=" Not a color",
        judge_model_name="google/gemma-3-1b-it",
    )

    test_generations = ["Red", "Blue", "Purple", "Banana", "Car", "Bcekham"]
    results = judge.judge_generations(test_generations)
    print(f"String template results: {results}")

    # Chat template example
    chat_template = [
        {
            "role": "system",
            "content": "You are a judge that determines if text is a valid color name.",
        },
        {
            "role": "user",
            "content": "Is '{generation}' a valid color name? Answer only 'Valid' or 'Invalid'.",
        },
    ]

    judge_chat = LLMJudge(
        prompt_template=chat_template,
        valid_text="Valid",
        invalid_text="Invalid",
        judge_model_name="google/gemma-3-1b-it",
    )

    results_chat = judge_chat.judge_generations(test_generations)
    print(f"Chat template results: {results_chat}")

    # Callable template example
    def custom_template(generation: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "Judge whether the text is a valid color name.",
            },
            {
                "role": "user",
                "content": f"Text: '{generation}'. Valid or Invalid color? Respond with 'Valid' or 'Invalid'.",
            },
        ]

    judge_callable = LLMJudge(
        prompt_template=custom_template,
        valid_text="Valid",
        invalid_text="Invalid",
        judge_model_name="google/gemma-3-1b-it",
    )

    results_callable = judge_callable.judge_generations(test_generations)
    print(f"Callable template results: {results_callable}")
