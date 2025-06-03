"""Pipeline metrics tracking for autobioml framework."""

import time
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from autogen_core.models import RequestUsage
from autogen_agentchat.base import TaskResult, Response


@dataclass
class PipelineMetrics:
    """Tracks comprehensive metrics for a pipeline run."""
    
    # Timing metrics
    start_time: float = 0.0
    end_time: float = 0.0
    total_duration_seconds: float = 0.0
    
    # Message metrics
    total_messages: int = 0
    team_a_messages: int = 0
    team_b_messages: int = 0
    
    # Token metrics
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    
    # Team-specific token metrics
    team_a_input_tokens: int = 0
    team_a_output_tokens: int = 0
    team_b_input_tokens: int = 0
    team_b_output_tokens: int = 0
    
    # Iteration metrics
    iterations_completed: int = 0
    completion_reason: str = ""
    
    # Evaluation metrics
    evaluation_completed: bool = False
    
    # Cost estimation (optional)
    estimated_cost_usd: float = 0.0


class PipelineMetricsTracker:
    """Tracks and aggregates metrics throughout a pipeline run."""
    
    def __init__(self):
        self.metrics = PipelineMetrics()
        self.token_rates = {
            # OpenAI GPT-4 pricing (per 1M tokens) - update as needed
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4.1": {"input": 2.00, "output": 8.00},
            "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        }
    
    def start_run(self) -> None:
        """Mark the start of a pipeline run."""
        self.metrics.start_time = time.time()
    
    def print_pipeline_start(self, task_name: str = "Pipeline") -> None:
        """Print a message when the pipeline starts."""
        print(f"\n{'='*60}")
        print(f"🚀 STARTING {task_name.upper()}")
        print(f"{'='*60}")
        print(f"⏰ Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.metrics.start_time))}")
        print(f"📊 Metrics tracking enabled - will report progress after each team")
        print(f"🎯 Automatic evaluation will run if predictions.arrow is generated")
        print(f"{'='*60}\n")
    
    def end_run(self, completion_reason: str = "completed") -> None:
        """Mark the end of a pipeline run."""
        self.metrics.end_time = time.time()
        self.metrics.total_duration_seconds = self.metrics.end_time - self.metrics.start_time
        self.metrics.completion_reason = completion_reason
        self.metrics.total_tokens = self.metrics.total_input_tokens + self.metrics.total_output_tokens
    
    def track_team_a_result(self, result: Response) -> None:
        """Track metrics from a Team A (planning) run."""
        if hasattr(result, 'inner_messages') and result.inner_messages:
            # Count all inner messages
            self.metrics.team_a_messages += len(result.inner_messages)
            
            # Aggregate token usage from all inner messages
            for msg in result.inner_messages:
                if hasattr(msg, 'models_usage') and msg.models_usage:
                    self.metrics.team_a_input_tokens += msg.models_usage.prompt_tokens
                    self.metrics.team_a_output_tokens += msg.models_usage.completion_tokens
        
        # Also count the final response message
        self.metrics.team_a_messages += 1
        if hasattr(result.chat_message, 'models_usage') and result.chat_message.models_usage:
            self.metrics.team_a_input_tokens += result.chat_message.models_usage.prompt_tokens
            self.metrics.team_a_output_tokens += result.chat_message.models_usage.completion_tokens
    
    def track_team_b_result(self, result: Response) -> None:
        """Track metrics from a Team B (implementation) run."""
        if hasattr(result, 'inner_messages') and result.inner_messages:
            # Count all inner messages
            self.metrics.team_b_messages += len(result.inner_messages)
            
            # Aggregate token usage from all inner messages
            for msg in result.inner_messages:
                if hasattr(msg, 'models_usage') and msg.models_usage:
                    self.metrics.team_b_input_tokens += msg.models_usage.prompt_tokens
                    self.metrics.team_b_output_tokens += msg.models_usage.completion_tokens
        
        # Also count the final response message
        self.metrics.team_b_messages += 1
        if hasattr(result.chat_message, 'models_usage') and result.chat_message.models_usage:
            self.metrics.team_b_input_tokens += result.chat_message.models_usage.prompt_tokens
            self.metrics.team_b_output_tokens += result.chat_message.models_usage.completion_tokens
    
    def complete_iteration(self) -> None:
        """Mark completion of one Team A -> Team B iteration."""
        self.metrics.iterations_completed += 1
    
    def mark_evaluation_completed(self, success: bool = True) -> None:
        """Mark whether automatic evaluation completed successfully."""
        self.metrics.evaluation_completed = success
    
    def print_team_progress(self, team_name: str, iteration: int) -> None:
        """Print progress after each team completes their work."""
        # Calculate current totals
        self.calculate_totals()
        
        # Calculate elapsed time
        current_time = time.time()
        elapsed = current_time - self.metrics.start_time
        
        # Team-specific metrics
        if team_name.upper().startswith("TEAM A"):
            team_messages = self.metrics.team_a_messages
            team_input_tokens = self.metrics.team_a_input_tokens
            team_output_tokens = self.metrics.team_a_output_tokens
            if "corrective" in team_name.lower():
                icon = "🔄"
                role = "Corrective Planning"
            else:
                icon = "🧠"
                role = "Planning"
        else:  # TEAM B
            team_messages = self.metrics.team_b_messages
            team_input_tokens = self.metrics.team_b_input_tokens
            team_output_tokens = self.metrics.team_b_output_tokens
            icon = "⚙️"
            role = "Implementation"
        
        team_total_tokens = team_input_tokens + team_output_tokens
        
        # Clean up team name for display
        display_name = team_name.upper()
        if "corrective" in team_name.lower():
            display_name = "TEAM A"  # Clean display name
        
        print(f"\n{'-'*50}")
        print(f"{icon} {display_name} ({role}) COMPLETED - Iteration {iteration}")
        print(f"{'-'*50}")
        print(f"⏱️  Elapsed Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
        print(f"📊 {display_name} Metrics:")
        print(f"   • Messages: {team_messages}")
        print(f"   • Tokens: {team_total_tokens:,} (input: {team_input_tokens:,}, output: {team_output_tokens:,})")
        print(f"🏃 Pipeline Totals:")
        print(f"   • Total Messages: {self.metrics.total_messages}")
        print(f"   • Total Tokens: {self.metrics.total_tokens:,}")
        print(f"   • Completed Iterations: {self.metrics.iterations_completed}")
        
        # Add cost estimate if we have meaningful token usage
        if self.metrics.total_tokens > 0:
            cost = self.estimate_cost("gpt-4.1-mini")
            print(f"💰 Estimated Cost So Far: ${cost:.4f}")
        
        print(f"{'-'*50}\n")
    
    def calculate_totals(self) -> None:
        """Calculate total metrics from team-specific metrics."""
        self.metrics.total_messages = self.metrics.team_a_messages + self.metrics.team_b_messages
        self.metrics.total_input_tokens = self.metrics.team_a_input_tokens + self.metrics.team_b_input_tokens
        self.metrics.total_output_tokens = self.metrics.team_a_output_tokens + self.metrics.team_b_output_tokens
        self.metrics.total_tokens = self.metrics.total_input_tokens + self.metrics.total_output_tokens
    
    def estimate_cost(self, model_name: str = "gpt-4.1-mini") -> float:
        """Estimate the cost of the run based on token usage with caching heuristic."""
        if model_name not in self.token_rates:
            return 0.0
        
        rates = self.token_rates[model_name]
        
        # Calculate base costs per 1M tokens
        input_cost = (self.metrics.total_input_tokens / 1_000_000) * rates["input"]
        output_cost = (self.metrics.total_output_tokens / 1_000_000) * rates["output"]
        
        base_cost = input_cost + output_cost
        
        # Apply caching heuristic: assume ~50% of input tokens are cached (charged at 1/4 price)
        # This means overall input cost is roughly: 50% * full_price + 50% * (1/4 * full_price) = 62.5% of full price
        # For simplicity, we'll use 60% of input cost to account for caching
        cached_input_cost = input_cost * 0.6
        
        # Output tokens are never cached, so full price
        total_estimated_cost = cached_input_cost + output_cost
        
        self.metrics.estimated_cost_usd = total_estimated_cost
        return total_estimated_cost
    
    def print_summary(self) -> None:
        """Print a human-readable summary of the metrics."""
        self.calculate_totals()
        
        print("\n" + "="*60)
        print("PIPELINE RUN SUMMARY")
        print("="*60)
        print(f"Total Duration: {self.metrics.total_duration_seconds:.2f} seconds ({self.metrics.total_duration_seconds/60:.1f} minutes)")
        print(f"Iterations Completed: {self.metrics.iterations_completed}")
        print(f"Completion Reason: {self.metrics.completion_reason}")
        print()
        print("MESSAGE COUNTS:")
        print(f"  Total Messages: {self.metrics.total_messages}")
        print(f"  Team A (Planning): {self.metrics.team_a_messages}")
        print(f"  Team B (Implementation): {self.metrics.team_b_messages}")
        print()
        print("TOKEN USAGE:")
        print(f"  Total Tokens: {self.metrics.total_tokens:,}")
        print(f"  Input Tokens: {self.metrics.total_input_tokens:,}")
        print(f"  Output Tokens: {self.metrics.total_output_tokens:,}")
        print()
        print("TOKEN BREAKDOWN BY TEAM:")
        print(f"  Team A - Input: {self.metrics.team_a_input_tokens:,}, Output: {self.metrics.team_a_output_tokens:,}")
        print(f"  Team B - Input: {self.metrics.team_b_input_tokens:,}, Output: {self.metrics.team_b_output_tokens:,}")
        
        if self.metrics.estimated_cost_usd > 0:
            print(f"\nEstimated Cost: ${self.metrics.estimated_cost_usd:.4f} USD")
        print("="*60)
    
    def save_to_json(self, filepath: Path) -> None:
        """Save metrics to a JSON file."""
        self.calculate_totals()
        
        # Convert to dictionary and add some computed fields
        metrics_dict = asdict(self.metrics)
        metrics_dict['total_duration_minutes'] = self.metrics.total_duration_seconds / 60
        metrics_dict['average_tokens_per_message'] = (
            self.metrics.total_tokens / self.metrics.total_messages 
            if self.metrics.total_messages > 0 else 0
        )
        metrics_dict['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.metrics.end_time))
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as a dictionary."""
        self.calculate_totals()
        return asdict(self.metrics) 