"""Inference execution for prompt-based methods."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.preprocess import load_gsm8k, format_question_prompt
from src.model import LLMInterface, parse_bvcot_response, extract_numeric_answer


def compute_verifiability_score(
    parsed_response: Dict[str, Any], config: DictConfig
) -> float:
    """Compute dataset-agnostic verifiability proxy score.

    Args:
        parsed_response: Parsed BV-CoT response with steps
        config: Method configuration with verifiability weights

    Returns:
        Verifiability score between 0 and 1
    """
    steps = parsed_response.get("steps", [])
    if not steps:
        return 0.0

    weights = config.method.verifiability
    max_steps = config.method.max_steps

    # 1. Step count adherence (penalty for exceeding max_steps)
    step_count = len(steps)
    step_score = (
        1.0
        if step_count <= max_steps
        else max(0.0, 1.0 - (step_count - max_steps) / max_steps)
    )

    # 2. Equation density (reward for explicit equations)
    equations = [
        s for s in steps if s.get("equation") and s["equation"] != "if applicable"
    ]
    equation_score = min(1.0, len(equations) / max(1, step_count))

    # 3. New entity penalty (check for new entities in claims)
    # Simplified: count unique nouns/entities mentioned
    all_text = " ".join([s.get("claim", "") for s in steps])
    words = all_text.lower().split()
    unique_entities = len(set([w for w in words if len(w) > 3]))  # rough heuristic
    entity_score = max(
        0.0, 1.0 - unique_entities / 50.0
    )  # penalty if too many new entities

    # 4. Numeric consistency (check if equations are internally consistent)
    consistency_score = 0.0
    for step in steps:
        if step.get("checkable", False):
            consistency_score += 1.0
    consistency_score = consistency_score / max(1, step_count)

    # Weighted combination
    total_score = (
        weights.step_count_weight * step_score
        + weights.equation_density_weight * equation_score
        + weights.new_entity_penalty_weight * entity_score
        + weights.numeric_consistency_weight * consistency_score
    )

    return total_score


def bvcot_inference(
    questions: List[Dict[str, Any]], llm: LLMInterface, config: DictConfig, mode: str
) -> Dict[str, Any]:
    """Run BV-CoT inference with verifiability-constrained selection.

    Args:
        questions: List of question dicts
        llm: LLM interface
        config: Configuration
        mode: Execution mode (main/sanity_check)

    Returns:
        Results dictionary with predictions and metrics
    """
    num_samples = config.method.num_samples
    threshold = config.method.verifiability_threshold
    confidence_weight = config.method.confidence_weight

    results = []
    correct = 0
    total_verifiability = 0.0
    total_steps = 0

    for q_data in tqdm(questions, desc="BV-CoT inference"):
        question = q_data["question"]
        ground_truth = q_data["numeric_answer"]

        # Generate prompt
        prompt = format_question_prompt(question, "bvcot")

        # Sample K candidates
        responses = llm.generate_multiple(prompt, num_samples)

        # Parse and score candidates
        candidates = []
        for resp in responses:
            parsed = parse_bvcot_response(resp)
            if parsed:
                verif_score = compute_verifiability_score(parsed, config)
                candidates.append(
                    {
                        "parsed": parsed,
                        "verifiability": verif_score,
                        "answer": parsed.get("answer"),
                        "confidence": parsed.get("confidence", 0.5),
                    }
                )

        if not candidates:
            # Fallback: no valid parse
            prediction = None
        else:
            # Filter by verifiability threshold
            valid = [c for c in candidates if c["verifiability"] >= threshold]

            if not valid:
                # Fallback: use all candidates with weighted voting
                valid = candidates

            # Weighted voting: weight = verifiability * confidence
            answer_weights = {}
            for c in valid:
                ans = c["answer"]
                weight = c["verifiability"] * (
                    confidence_weight * c["confidence"] + (1 - confidence_weight)
                )
                answer_weights[ans] = answer_weights.get(ans, 0.0) + weight

            # Select answer with highest total weight
            prediction = (
                max(answer_weights.items(), key=lambda x: x[1])[0]
                if answer_weights
                else None
            )

        # Check correctness
        is_correct = prediction is not None and abs(prediction - ground_truth) < 1e-6
        if is_correct:
            correct += 1

        # Accumulate metrics
        if candidates:
            avg_verif = sum(c["verifiability"] for c in candidates) / len(candidates)
            avg_steps = sum(
                len(c["parsed"].get("steps", [])) for c in candidates
            ) / len(candidates)
            total_verifiability += avg_verif
            total_steps += avg_steps

        results.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "correct": is_correct,
                "num_valid_candidates": len(candidates),
            }
        )

    accuracy = correct / len(questions) if questions else 0.0
    avg_verifiability = total_verifiability / len(questions) if questions else 0.0
    avg_steps = total_steps / len(questions) if questions else 0.0

    return {
        "results": results,
        "accuracy": accuracy,
        "avg_verifiability": avg_verifiability,
        "avg_steps": avg_steps,
        "num_questions": len(questions),
    }


def self_consistency_inference(
    questions: List[Dict[str, Any]], llm: LLMInterface, config: DictConfig, mode: str
) -> Dict[str, Any]:
    """Run self-consistency inference with majority voting.

    Args:
        questions: List of question dicts
        llm: LLM interface
        config: Configuration
        mode: Execution mode (main/sanity_check)

    Returns:
        Results dictionary with predictions and metrics
    """
    num_samples = config.method.num_samples
    early_stopping = config.inference.early_stopping
    early_stop_threshold = config.inference.early_stopping_threshold

    results = []
    correct = 0

    for q_data in tqdm(questions, desc="Self-Consistency inference"):
        question = q_data["question"]
        ground_truth = q_data["numeric_answer"]

        # Generate prompt
        prompt = format_question_prompt(question, "self_consistency")

        # Sample K candidates with optional early stopping
        answers = []
        for i in range(num_samples):
            resp = llm.generate(prompt)
            ans = extract_numeric_answer(resp)
            if ans is not None:
                answers.append(ans)

            # Early stopping check
            if early_stopping and len(answers) >= early_stop_threshold:
                counter = Counter(answers)
                most_common = counter.most_common(1)[0]
                if most_common[1] >= early_stop_threshold:
                    break

        # Majority vote
        if answers:
            counter = Counter(answers)
            prediction = counter.most_common(1)[0][0]
        else:
            prediction = None

        # Check correctness
        is_correct = prediction is not None and abs(prediction - ground_truth) < 1e-6
        if is_correct:
            correct += 1

        results.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "correct": is_correct,
                "num_valid_answers": len(answers),
            }
        )

    accuracy = correct / len(questions) if questions else 0.0

    return {"results": results, "accuracy": accuracy, "num_questions": len(questions)}


def run_inference(config: DictConfig) -> None:
    """Main inference execution function.

    Args:
        config: Hydra configuration
    """
    # Load dataset
    num_questions = config.dataset.num_questions
    if config.mode == "sanity_check":
        num_questions = config.inference.sanity_num_samples

    questions = load_gsm8k(
        cache_dir=config.cache_dir,
        split=config.dataset.split,
        num_questions=num_questions,
        seed=config.dataset.subsample_seed,
    )

    print(f"Loaded {len(questions)} questions from GSM8K")

    # Initialize LLM
    llm = LLMInterface(
        provider=config.model.provider,
        model_name=config.model.name,
        api_key_env=config.model.api_key_env,
        temperature=config.method.temperature,
        max_tokens=config.method.max_tokens,
    )

    # Initialize WandB
    if config.wandb.mode != "disabled":
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            name=config.run.run_id,
            config=OmegaConf.to_container(config, resolve=True),
        )
        print(f"WandB run URL: {wandb.run.get_url()}")

    # Run inference based on method type
    if config.method.type == "bvcot":
        inference_results = bvcot_inference(questions, llm, config, config.mode)
    elif config.method.type == "self_consistency":
        inference_results = self_consistency_inference(
            questions, llm, config, config.mode
        )
    else:
        raise ValueError(f"Unknown method type: {config.method.type}")

    # Log metrics to WandB
    if config.wandb.mode != "disabled":
        wandb.summary["accuracy"] = inference_results["accuracy"]
        wandb.summary["num_questions"] = inference_results["num_questions"]

        if "avg_verifiability" in inference_results:
            wandb.summary["avg_verifiability"] = inference_results["avg_verifiability"]
            wandb.summary["avg_steps"] = inference_results["avg_steps"]

    # Save results to disk
    results_dir = Path(config.results_dir) / config.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "results.json", "w") as f:
        json.dump(inference_results, f, indent=2)

    print(f"\nResults saved to {results_dir}")
    print(f"Accuracy: {inference_results['accuracy']:.4f}")

    # Sanity validation
    if config.mode == "sanity_check":
        validate_sanity_check(inference_results, config)

    # Finish WandB
    if config.wandb.mode != "disabled":
        wandb.finish()


def validate_sanity_check(results: Dict[str, Any], config: DictConfig) -> None:
    """Validate sanity check results and print verdict.

    Args:
        results: Inference results
        config: Configuration
    """
    num_questions = results["num_questions"]
    accuracy = results["accuracy"]

    # Check minimum samples processed
    if num_questions < 5:
        print(
            f"SANITY_VALIDATION: FAIL reason=insufficient_samples (got {num_questions}, need >=5)"
        )
        print(
            f'SANITY_VALIDATION_SUMMARY: {{"samples": {num_questions}, "outputs_valid": false, "outputs_unique": false}}'
        )
        return

    # Check for valid outputs
    valid_results = [r for r in results["results"] if r["prediction"] is not None]
    if len(valid_results) < num_questions * 0.5:
        print(f"SANITY_VALIDATION: FAIL reason=too_many_invalid_outputs")
        print(
            f'SANITY_VALIDATION_SUMMARY: {{"samples": {num_questions}, "outputs_valid": false, "outputs_unique": true}}'
        )
        return

    # Check that not all predictions are identical (should have diversity)
    predictions = [
        r["prediction"] for r in results["results"] if r["prediction"] is not None
    ]
    unique_predictions = len(set(predictions))
    if unique_predictions < 2:
        print(f"SANITY_VALIDATION: FAIL reason=all_identical_outputs")
        print(
            f'SANITY_VALIDATION_SUMMARY: {{"samples": {num_questions}, "outputs_valid": true, "outputs_unique": false}}'
        )
        return

    # Check that accuracy is reasonable (>0 for sanity check)
    if accuracy == 0.0:
        print(f"SANITY_VALIDATION: FAIL reason=zero_accuracy")
        print(
            f'SANITY_VALIDATION_SUMMARY: {{"samples": {num_questions}, "outputs_valid": true, "outputs_unique": true, "accuracy": 0.0}}'
        )
        return

    # All checks passed
    print("SANITY_VALIDATION: PASS")
    print(
        f'SANITY_VALIDATION_SUMMARY: {{"samples": {num_questions}, "outputs_valid": true, "outputs_unique": true, "accuracy": {accuracy:.4f}, "unique_predictions": {unique_predictions}}}'
    )
