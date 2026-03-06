"""LLM API interface for generating responses."""

import os
import json
from typing import List, Dict, Any, Optional
import time

from openai import OpenAI


class LLMInterface:
    """Interface for calling LLM APIs."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key_env: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        """Initialize LLM interface.

        Args:
            provider: API provider ('openai')
            model_name: Model identifier
            api_key_env: Environment variable name containing API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize client
        if provider == "openai":
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(
                    f"API key not found in environment variable: {api_key_env}"
                )
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a single response from the LLM.

        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Generated text response
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=max_tok,
            )
            return response.choices[0].message.content

        raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_multiple(
        self,
        prompt: str,
        num_samples: int,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate multiple responses from the LLM.

        Args:
            prompt: Input prompt
            num_samples: Number of samples to generate
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            List of generated text responses
        """
        responses = []
        for _ in range(num_samples):
            response = self.generate(prompt, temperature, max_tokens)
            responses.append(response)
            # Small delay to avoid rate limiting
            time.sleep(0.1)

        return responses


def parse_bvcot_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse BV-CoT JSON response.

    Args:
        response: Raw LLM response

    Returns:
        Parsed dictionary with steps, answer, confidence, or None if parsing fails
    """
    try:
        # Try to find JSON in the response
        start_idx = response.find("{")
        end_idx = response.rfind("}")

        if start_idx == -1 or end_idx == -1:
            return None

        json_str = response[start_idx : end_idx + 1]
        data = json.loads(json_str)

        # Validate required fields
        if "steps" not in data or "answer" not in data or "confidence" not in data:
            return None

        return data

    except (json.JSONDecodeError, ValueError):
        return None


def extract_numeric_answer(response: str) -> Optional[float]:
    """Extract numeric answer from self-consistency response.

    Args:
        response: Raw LLM response

    Returns:
        Extracted numeric answer, or None if extraction fails
    """
    import re

    # Look for "The answer is: <number>" pattern
    match = re.search(r"[Tt]he answer is:?\s*(-?[\d,]+\.?\d*)", response)
    if match:
        return float(match.group(1).replace(",", ""))

    # Fallback: look for numbers near the end of the response
    lines = response.strip().split("\n")
    for line in reversed(lines[-5:]):  # check last 5 lines
        numbers = re.findall(r"-?[\d,]+\.?\d*", line)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                continue

    return None
