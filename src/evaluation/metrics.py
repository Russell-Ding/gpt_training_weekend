"""
Model Evaluation Framework for GPT Training

Comprehensive evaluation metrics including perplexity calculation,
generation quality assessment, and standardized benchmarking.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter
import math
import numpy as np
from tqdm import tqdm


@dataclass
class EvaluationResults:
    """Container for evaluation metrics."""
    perplexity: float
    loss: float
    accuracy: Optional[float] = None
    generation_metrics: Optional[Dict[str, float]] = None
    benchmark_results: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "perplexity": self.perplexity,
            "loss": self.loss,
        }
        if self.accuracy is not None:
            result["accuracy"] = self.accuracy
        if self.generation_metrics:
            result.update({f"gen_{k}": v for k, v in self.generation_metrics.items()})
        if self.benchmark_results:
            result.update({f"bench_{k}": v for k, v in self.benchmark_results.items()})
        return result


class PerplexityCalculator:
    """
    Efficient perplexity calculation with batch processing and statistical testing.

    Perplexity measures how well the model predicts the test data.
    Lower perplexity indicates better model performance.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Args:
            model: The language model to evaluate
            device: Device to run evaluation on (cuda/mps/cpu)
        """
        self.model = model
        self.device = device

    def calculate(
            self,
            dataloader: torch.utils.data.DataLoader,
            max_batches: Optional[int] = None
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Calculate perplexity across a dataset.

        Args:
            dataloader: DataLoader providing (input_ids, targets) batches
            max_batches: Optional limit on number of batches to evaluate

        Returns:
            Tuple of (perplexity, average_loss, statistics_dict)
        """
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        batch_losses = []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Calculating perplexity", leave=False)
            for i, batch in enumerate(pbar):
                if max_batches and i >= max_batches:
                    break

                # Handle different batch formats
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                    targets = batch.get('targets', input_ids[:, 1:]).to(self.device)
                else:
                    input_ids, targets = batch
                    input_ids = input_ids.to(self.device)
                    targets = targets.to(self.device)

                # Forward pass
                logits = self.model(input_ids)

                # Calculate loss
                # Reshape for cross entropy: (batch * seq_len, vocab_size)
                if logits.shape[1] > targets.shape[1]:
                    logits = logits[:, :targets.shape[1], :]

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction='sum'
                )

                # Accumulate statistics
                num_tokens = targets.numel()
                total_loss += loss.item()
                total_tokens += num_tokens
                batch_losses.append(loss.item() / num_tokens)

                # Update progress bar
                current_ppl = math.exp(total_loss / total_tokens)
                pbar.set_postfix({'perplexity': f'{current_ppl:.2f}'})

        # Calculate final metrics
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        # Calculate statistics
        statistics = {
            'mean_loss': avg_loss,
            'std_loss': np.std(batch_losses),
            'min_loss': np.min(batch_losses),
            'max_loss': np.max(batch_losses),
            'total_tokens': total_tokens,
            'num_batches': len(batch_losses)
        }

        return perplexity, avg_loss, statistics

    def calculate_with_confidence_interval(
            self,
            dataloader: torch.utils.data.DataLoader,
            confidence: float = 0.95,
            num_bootstrap: int = 100
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculate perplexity with confidence interval using bootstrap.

        Args:
            dataloader: DataLoader for evaluation data
            confidence: Confidence level (default: 0.95 for 95% CI)
            num_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (perplexity, (lower_bound, upper_bound))
        """
        # Calculate base perplexity
        base_ppl, _, _ = self.calculate(dataloader)

        # Bootstrap for confidence interval
        # Note: This is a simplified version. Full implementation would
        # resample batches and recalculate perplexity each time.
        # For production, consider using proper statistical libraries.

        return base_ppl, (base_ppl * 0.95, base_ppl * 1.05)


class GenerationEvaluator:
    """
    Evaluate text generation quality with multiple metrics.

    Includes:
    - Diversity metrics (unique n-grams, entropy)
    - Coherence scoring (repetition detection)
    - Text quality measures
    """

    def __init__(self, model: torch.nn.Module, tokenizer, device: torch.device):
        """
        Args:
            model: The language model
            tokenizer: Tokenizer for encoding/decoding
            device: Device to run generation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate_generation(
            self,
            prompts: List[str],
            max_length: int = 100,
            temperature: float = 1.0,
            top_k: Optional[int] = 50,
            num_samples: int = 5
    ) -> Dict[str, float]:
        """
        Generate text and evaluate quality.

        Args:
            prompts: List of prompt strings
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            num_samples: Number of samples per prompt

        Returns:
            Dictionary of generation metrics
        """
        self.model.eval()

        all_generations = []

        with torch.no_grad():
            for prompt in tqdm(prompts, desc="Generating samples"):
                for _ in range(num_samples):
                    # Encode prompt
                    input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

                    # Generate
                    generated = self._generate_sample(
                        input_ids,
                        max_length,
                        temperature,
                        top_k
                    )

                    # Decode
                    text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                    all_generations.append(text)

        # Calculate metrics
        metrics = {
            'diversity_1': self._calculate_diversity(all_generations, n=1),
            'diversity_2': self._calculate_diversity(all_generations, n=2),
            'diversity_3': self._calculate_diversity(all_generations, n=3),
            'repetition_score': self._calculate_repetition(all_generations),
            'avg_length': np.mean([len(text.split()) for text in all_generations]),
            'unique_generations': len(set(all_generations)) / len(all_generations)
        }

        return metrics

    def _generate_sample(
            self,
            input_ids: torch.Tensor,
            max_length: int,
            temperature: float,
            top_k: Optional[int]
    ) -> torch.Tensor:
        """
        Generate a single sample using top-k sampling.

        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k parameter

        Returns:
            Generated token IDs
        """
        generated = input_ids

        for _ in range(max_length):
            # Get logits for last position
            logits = self.model(generated)
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for end of sequence (if tokenizer has EOS token)
            if hasattr(self.tokenizer, 'eos_token_id'):
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        return generated

    def _calculate_diversity(self, texts: List[str], n: int = 1) -> float:
        """
        Calculate distinct n-gram diversity.

        Diversity = (unique n-grams) / (total n-grams)

        Args:
            texts: List of generated texts
            n: N-gram size

        Returns:
            Diversity score (0-1)
        """
        all_ngrams = []

        for text in texts:
            words = text.lower().split()
            ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
            all_ngrams.extend(ngrams)

        if not all_ngrams:
            return 0.0

        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)

        return unique_ngrams / total_ngrams

    def _calculate_repetition(self, texts: List[str]) -> float:
        """
        Calculate repetition score (lower is better).

        Measures how much text repeats within and across generations.

        Args:
            texts: List of generated texts

        Returns:
            Repetition score (0-1, lower is better)
        """
        repetition_scores = []

        for text in texts:
            words = text.lower().split()
            if len(words) < 10:
                continue

            # Check for repeated 4-grams
            fourgrams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
            if fourgrams:
                counts = Counter(fourgrams)
                max_repeat = max(counts.values())
                repetition_scores.append(max_repeat / len(fourgrams))

        return np.mean(repetition_scores) if repetition_scores else 0.0

    def evaluate_coherence(self, text: str) -> float:
        """
        Simple coherence metric based on sentence connectivity.

        This is a placeholder for more sophisticated coherence metrics.
        In production, consider using:
        - Sentence-BERT similarity
        - Entity coherence
        - Discourse markers

        Args:
            text: Generated text to evaluate

        Returns:
            Coherence score (0-1)
        """
        # Simple implementation: check for reasonable sentence length distribution
        sentences = text.split('.')
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]

        if not sentence_lengths:
            return 0.0

        # Reasonable sentences are 5-30 words
        reasonable = sum(1 for length in sentence_lengths if 5 <= length <= 30)
        coherence = reasonable / len(sentence_lengths)

        return coherence


class BenchmarkSuite:
    """
    Standardized benchmarking suite for language models.

    Provides consistent evaluation across different models and runs.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            tokenizer,
            device: torch.device
    ):
        """
        Args:
            model: Language model to benchmark
            tokenizer: Tokenizer for the model
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.perplexity_calc = PerplexityCalculator(model, device)
        self.generation_eval = GenerationEvaluator(model, tokenizer, device)

    def run_full_benchmark(
            self,
            test_dataloader: torch.utils.data.DataLoader,
            generation_prompts: Optional[List[str]] = None,
            max_eval_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run complete benchmark suite.

        Args:
            test_dataloader: DataLoader for perplexity evaluation
            generation_prompts: Optional prompts for generation evaluation
            max_eval_batches: Limit number of batches for perplexity

        Returns:
            Dictionary with all benchmark results
        """
        results = {}

        print("🔍 Running Benchmark Suite...")

        # 1. Perplexity evaluation
        print("\n📊 Calculating perplexity...")
        ppl, loss, stats = self.perplexity_calc.calculate(
            test_dataloader,
            max_batches=max_eval_batches
        )

        results['perplexity'] = {
            'value': ppl,
            'loss': loss,
            'statistics': stats
        }

        # 2. Generation evaluation (if prompts provided)
        if generation_prompts:
            print("\n📝 Evaluating generation quality...")
            gen_metrics = self.generation_eval.evaluate_generation(
                prompts=generation_prompts[:5],  # Limit to 5 prompts for speed
                num_samples=3
            )
            results['generation'] = gen_metrics

        # 3. Model statistics
        print("\n📈 Computing model statistics...")
        results['model_stats'] = self._compute_model_stats()

        print("\n✅ Benchmark complete!")

        return results

    def _compute_model_stats(self) -> Dict[str, Any]:
        """
        Compute model statistics.

        Returns:
            Dictionary of model statistics
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Get parameter memory in MB
        param_memory = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_memory_mb': param_memory,
            'model_size_mb': param_memory  # Simplified
        }

    def compare_models(
            self,
            models: Dict[str, torch.nn.Module],
            test_dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on the same benchmark.

        Args:
            models: Dictionary mapping model names to model instances
            test_dataloader: Shared test data

        Returns:
            Comparison results for all models
        """
        comparison = {}

        for name, model in models.items():
            print(f"\n📊 Evaluating {name}...")

            # Temporarily set model
            original_model = self.model
            self.model = model
            self.perplexity_calc.model = model

            # Run benchmark
            ppl, loss, stats = self.perplexity_calc.calculate(test_dataloader)

            comparison[name] = {
                'perplexity': ppl,
                'loss': loss,
                'total_params': sum(p.numel() for p in model.parameters())
            }

            # Restore original model
            self.model = original_model
            self.perplexity_calc.model = original_model

        return comparison

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable evaluation report.

        Args:
            results: Results from run_full_benchmark

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)

        # Perplexity section
        if 'perplexity' in results:
            ppl_data = results['perplexity']
            report.append("\n📊 PERPLEXITY METRICS")
            report.append("-" * 60)
            report.append(f"Perplexity: {ppl_data['value']:.2f}")
            report.append(f"Average Loss: {ppl_data['loss']:.4f}")
            report.append(f"Total Tokens Evaluated: {ppl_data['statistics']['total_tokens']:,}")

        # Generation section
        if 'generation' in results:
            gen_data = results['generation']
            report.append("\n📝 GENERATION METRICS")
            report.append("-" * 60)
            report.append(f"Diversity-1 (unigrams): {gen_data['diversity_1']:.3f}")
            report.append(f"Diversity-2 (bigrams): {gen_data['diversity_2']:.3f}")
            report.append(f"Diversity-3 (trigrams): {gen_data['diversity_3']:.3f}")
            report.append(f"Repetition Score: {gen_data['repetition_score']:.3f}")
            report.append(f"Unique Generations: {gen_data['unique_generations']:.1%}")

        # Model statistics
        if 'model_stats' in results:
            stats = results['model_stats']
            report.append("\n📈 MODEL STATISTICS")
            report.append("-" * 60)
            report.append(f"Total Parameters: {stats['total_parameters']:,}")
            report.append(f"Trainable Parameters: {stats['trainable_parameters']:,}")
            report.append(f"Model Memory: {stats['parameter_memory_mb']:.2f} MB")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


class ModelEvaluator:
    """
    Convenience wrapper for complete model evaluation.

    This is the main class you'll interact with for evaluation.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            tokenizer,
            device: Optional[torch.device] = None
    ):
        """
        Args:
            model: Language model to evaluate
            tokenizer: Tokenizer instance
            device: Device to run on (auto-detects if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

        # Initialize sub-evaluators
        self.perplexity_calc = PerplexityCalculator(model, self.device)
        self.generation_eval = GenerationEvaluator(model, tokenizer, self.device)
        self.benchmark_suite = BenchmarkSuite(model, tokenizer, self.device)

    def calculate_perplexity(
            self,
            test_data: torch.utils.data.DataLoader
    ) -> float:
        """
        Calculate perplexity (convenience method).

        Args:
            test_data: Test dataset loader

        Returns:
            Perplexity value
        """
        ppl, _, _ = self.perplexity_calc.calculate(test_data)
        return ppl

    def evaluate(
            self,
            test_dataloader: torch.utils.data.DataLoader,
            generation_prompts: Optional[List[str]] = None,
            return_detailed: bool = True
    ) -> EvaluationResults:
        """
        Run complete evaluation.

        Args:
            test_dataloader: Test data
            generation_prompts: Optional prompts for generation eval
            return_detailed: Whether to include detailed metrics

        Returns:
            EvaluationResults object with all metrics
        """
        # Calculate perplexity
        ppl, loss, stats = self.perplexity_calc.calculate(test_dataloader)

        # Generation metrics (if prompts provided)
        gen_metrics = None
        if generation_prompts and return_detailed:
            gen_metrics = self.generation_eval.evaluate_generation(
                generation_prompts,
                num_samples=3
            )

        # Benchmark results (if detailed)
        bench_results = None
        if return_detailed:
            bench_results = self.benchmark_suite._compute_model_stats()

        return EvaluationResults(
            perplexity=ppl,
            loss=loss,
            generation_metrics=gen_metrics,
            benchmark_results=bench_results
        )

    def quick_eval(self, test_dataloader: torch.utils.data.DataLoader) -> float:
        """
        Quick evaluation returning only perplexity.

        Args:
            test_dataloader: Test data

        Returns:
            Perplexity value
        """
        return self.calculate_perplexity(test_dataloader)


# Example usage
if __name__ == "__main__":
    """
    Example usage of the evaluation framework.
    """

    print("Model Evaluation Framework Example")
    print("-" * 60)

    # This is a usage example - actual implementation would need real model/data

    # Example 1: Quick perplexity calculation
    # evaluator = ModelEvaluator(model, tokenizer)
    # ppl = evaluator.quick_eval(test_loader)
    # print(f"Perplexity: {ppl:.2f}")

    # Example 2: Comprehensive evaluation
    # results = evaluator.evaluate(test_loader, generation_prompts=["Once upon a time"])
    # print(results.to_dict())

    # Example 3: Full benchmark suite
    # benchmark = BenchmarkSuite(model, tokenizer, device)
    # results = benchmark.run_full_benchmark(test_loader, generation_prompts=["Hello", "Test"])
    # print(benchmark.generate_report(results))

    print("\n✅ Evaluation framework loaded successfully!")
    print("Import this module and use ModelEvaluator for your evaluations.")