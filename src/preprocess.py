"""Dataset loading and preprocessing for GSM8K."""

import random
from pathlib import Path
from typing import List, Dict, Any
import re

from datasets import load_dataset


def extract_answer(answer_text: str) -> float:
    """Extract numeric answer from GSM8K answer string.

    GSM8K answers are formatted as explanations followed by #### <number>.

    Args:
        answer_text: The full answer string from GSM8K

    Returns:
        The numeric answer as a float
    """
    # GSM8K format: "explanation text #### 42"
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        # Remove commas from numbers like "1,234"
        return float(match.group(1).replace(",", ""))

    # Fallback: try to find any number in the string
    numbers = re.findall(r"-?\d+\.?\d*", answer_text)
    if numbers:
        return float(numbers[-1])

    raise ValueError(f"Could not extract numeric answer from: {answer_text}")


def load_gsm8k(
    cache_dir: str, split: str = "test", num_questions: int = 200, seed: int = 42
) -> List[Dict[str, Any]]:
    """Load and subsample GSM8K dataset.

    Args:
        cache_dir: Directory to cache downloaded dataset
        split: Dataset split to load (train/test)
        num_questions: Number of questions to subsample
        seed: Random seed for subsampling

    Returns:
        List of dictionaries with keys: question, answer, numeric_answer
    """
    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset from HuggingFace
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    # Convert to list of dicts
    questions = []
    for item in dataset:
        try:
            numeric_answer = extract_answer(item["answer"])
            questions.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "numeric_answer": numeric_answer,
                }
            )
        except ValueError as e:
            print(f"Warning: Skipping question due to parsing error: {e}")
            continue

    # Subsample if requested
    if num_questions and num_questions < len(questions):
        random.seed(seed)
        questions = random.sample(questions, num_questions)

    return questions


def format_question_prompt(question: str, method_type: str) -> str:
    """Format question with appropriate prompt for the method.

    Args:
        question: The math word problem
        method_type: 'bvcot' or 'self_consistency'

    Returns:
        Formatted prompt string
    """
    if method_type == "bvcot":
        return f"""Solve this math problem by breaking it into atomic, verifiable steps.

Question: {question}

Provide your answer in the following JSON format:
{{
  "steps": [
    {{"step": 1, "claim": "description", "equation": "if applicable", "checkable": true/false}},
    ...
  ],
  "answer": <numeric answer>,
  "confidence": <0.0 to 1.0>
}}

Requirements:
- Each step should contain exactly one operation or logical claim
- Include explicit equations when performing calculations
- Mark each step as checkable (true) if a human can verify it without additional context
- Keep the total number of steps minimal (ideally ≤ 10)
- Confidence should reflect your certainty in the final answer"""

    else:  # self_consistency
        return f"""Solve this math problem step by step.

Question: {question}

Provide your reasoning and then give your final numeric answer.
Format your final answer as: "The answer is: <number>" """
