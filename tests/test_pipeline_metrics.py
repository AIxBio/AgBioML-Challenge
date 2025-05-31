"""Test the pipeline metrics tracking functionality."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

# import pytest  # Not needed - can run without pytest

from src.bioagents.pipeline.metrics import PipelineMetricsTracker, PipelineMetrics
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage
from autogen_core.models import RequestUsage


def create_mock_message(source: str, content: str, prompt_tokens: int, completion_tokens: int):
    """Helper to create a mock message with token usage."""
    message = TextMessage(content=content, source=source)
    message.models_usage = RequestUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )
    return message


def create_mock_response(chat_message, inner_messages=None):
    """Helper to create a mock Response object."""
    response = Mock(spec=Response)
    response.chat_message = chat_message
    response.inner_messages = inner_messages or []
    return response


def test_pipeline_metrics_tracker_basic():
    """Test basic functionality of the metrics tracker."""
    tracker = PipelineMetricsTracker()
    
    # Test initial state
    assert tracker.metrics.total_messages == 0
    assert tracker.metrics.total_tokens == 0
    assert tracker.metrics.iterations_completed == 0
    
    # Test start_run
    start_time = time.time()
    tracker.start_run()
    assert tracker.metrics.start_time >= start_time
    
    # Test end_run
    time.sleep(0.1)  # Small delay to ensure duration > 0
    tracker.end_run("test_completion")
    assert tracker.metrics.end_time > tracker.metrics.start_time
    assert tracker.metrics.total_duration_seconds > 0
    assert tracker.metrics.completion_reason == "test_completion"


def test_team_a_tracking():
    """Test Team A metrics tracking."""
    tracker = PipelineMetricsTracker()
    
    # Create mock Team A response with inner messages (group chat)
    principal_msg = create_mock_message("principal_scientist", "Let's plan this project...", 100, 50)
    bioinf_msg = create_mock_message("bioinformatics_expert", "I suggest we analyze...", 120, 80)
    ml_msg = create_mock_message("ml_expert", "For the model, we should...", 110, 70)
    final_msg = create_mock_message("principal_scientist", "Here's our final plan...", 90, 60)
    
    team_a_response = create_mock_response(
        chat_message=final_msg,
        inner_messages=[principal_msg, bioinf_msg, ml_msg]
    )
    
    # Track the response
    tracker.track_team_a_result(team_a_response)
    
    # Verify metrics
    assert tracker.metrics.team_a_messages == 4  # 3 inner + 1 final
    assert tracker.metrics.team_a_input_tokens == 100 + 120 + 110 + 90  # 420
    assert tracker.metrics.team_a_output_tokens == 50 + 80 + 70 + 60  # 260


def test_team_b_tracking():
    """Test Team B metrics tracking."""
    tracker = PipelineMetricsTracker()
    
    # Create mock Team B response (engineer + critic iterations)
    engineer_msg1 = create_mock_message("implementation_engineer", "I'll implement the solution...", 200, 150)
    critic_msg1 = create_mock_message("data_science_critic", "Please revise the approach...", 180, 100)
    engineer_msg2 = create_mock_message("implementation_engineer", "I've revised the code...", 220, 180)
    critic_msg2 = create_mock_message("data_science_critic", "APPROVE_ENGINEER", 150, 20)
    final_summary = create_mock_message("TEAM_B", "Implementation complete with approval", 50, 30)
    
    team_b_response = create_mock_response(
        chat_message=final_summary,
        inner_messages=[engineer_msg1, critic_msg1, engineer_msg2, critic_msg2]
    )
    
    # Track the response
    tracker.track_team_b_result(team_b_response)
    
    # Verify metrics
    assert tracker.metrics.team_b_messages == 5  # 4 inner + 1 final
    assert tracker.metrics.team_b_input_tokens == 200 + 180 + 220 + 150 + 50  # 800
    assert tracker.metrics.team_b_output_tokens == 150 + 100 + 180 + 20 + 30  # 480


def test_full_pipeline_simulation():
    """Test a complete pipeline simulation with multiple iterations."""
    tracker = PipelineMetricsTracker()
    tracker.start_run()
    
    # Simulate 3 iterations
    for i in range(3):
        # Team A planning
        team_a_final = create_mock_message("principal_scientist", f"Plan for iteration {i+1}", 100, 80)
        team_a_inner = [
            create_mock_message("principal_scientist", f"Starting iteration {i+1}", 80, 40),
            create_mock_message("bioinformatics_expert", f"Bio analysis {i+1}", 90, 60),
            create_mock_message("ml_expert", f"ML strategy {i+1}", 85, 55),
        ]
        team_a_response = create_mock_response(team_a_final, team_a_inner)
        tracker.track_team_a_result(team_a_response)
        
        # Team B implementation
        team_b_final = create_mock_message("TEAM_B", f"Implementation {i+1} complete", 50, 30)
        team_b_inner = [
            create_mock_message("implementation_engineer", f"Implementing {i+1}", 200, 150),
            create_mock_message("data_science_critic", f"Review {i+1}", 150, 80),
        ]
        team_b_response = create_mock_response(team_b_final, team_b_inner)
        tracker.track_team_b_result(team_b_response)
        
        # Complete iteration
        tracker.complete_iteration()
    
    tracker.end_run("task_completed")
    
    # Verify final metrics
    tracker.calculate_totals()
    
    # Expected: 3 iterations × (4 Team A + 3 Team B) = 21 total messages
    assert tracker.metrics.iterations_completed == 3
    assert tracker.metrics.total_messages == 21
    assert tracker.metrics.team_a_messages == 12  # 3 × 4
    assert tracker.metrics.team_b_messages == 9   # 3 × 3
    
    # Expected token totals
    expected_team_a_input = 3 * (100 + 80 + 90 + 85)  # 3 × 355 = 1065
    expected_team_a_output = 3 * (80 + 40 + 60 + 55)  # 3 × 235 = 705
    expected_team_b_input = 3 * (50 + 200 + 150)  # 3 × 400 = 1200
    expected_team_b_output = 3 * (30 + 150 + 80)  # 3 × 260 = 780
    
    assert tracker.metrics.team_a_input_tokens == expected_team_a_input
    assert tracker.metrics.team_a_output_tokens == expected_team_a_output
    assert tracker.metrics.team_b_input_tokens == expected_team_b_input
    assert tracker.metrics.team_b_output_tokens == expected_team_b_output
    
    assert tracker.metrics.total_input_tokens == expected_team_a_input + expected_team_b_input
    assert tracker.metrics.total_output_tokens == expected_team_a_output + expected_team_b_output
    assert tracker.metrics.total_tokens == tracker.metrics.total_input_tokens + tracker.metrics.total_output_tokens


def test_cost_estimation():
    """Test cost estimation functionality."""
    tracker = PipelineMetricsTracker()
    
    # Set some token usage
    tracker.metrics.total_input_tokens = 1_000_000  # 1M input tokens
    tracker.metrics.total_output_tokens = 500_000   # 500k output tokens
    
    # Test with gpt-4.1-mini: input $0.40/1M, output $1.60/1M
    cost = tracker.estimate_cost("gpt-4.1-mini")
    
    # Expected calculation:
    # Input cost with caching: (1M / 1M) * $0.40 * 0.6 = $0.24
    # Output cost (no caching): (0.5M / 1M) * $1.60 = $0.80
    # Total: $0.24 + $0.80 = $1.04
    expected_cost = 1.04
    assert abs(cost - expected_cost) < 0.001
    assert tracker.metrics.estimated_cost_usd == cost
    
    # Test with unknown model
    cost_unknown = tracker.estimate_cost("unknown-model")
    assert cost_unknown == 0.0


def test_json_output():
    """Test JSON output functionality."""
    tracker = PipelineMetricsTracker()
    tracker.start_run()
    
    # Add some test data
    tracker.metrics.team_a_messages = 10
    tracker.metrics.team_b_messages = 15
    tracker.metrics.team_a_input_tokens = 5000
    tracker.metrics.team_a_output_tokens = 3000
    tracker.metrics.team_b_input_tokens = 7000
    tracker.metrics.team_b_output_tokens = 4000
    tracker.metrics.iterations_completed = 2
    
    time.sleep(0.1)  # Small delay
    tracker.end_run("test_completion")
    tracker.estimate_cost("gpt-4.1-mini")
    
    # Test JSON output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    
    try:
        tracker.save_to_json(tmp_path)
        
        # Verify file was created and contains expected data
        assert tmp_path.exists()
        
        with open(tmp_path, 'r') as f:
            data = json.load(f)
        
        # Check key fields
        assert data['total_messages'] == 25
        assert data['team_a_messages'] == 10
        assert data['team_b_messages'] == 15
        assert data['total_input_tokens'] == 12000
        assert data['total_output_tokens'] == 7000
        assert data['total_tokens'] == 19000
        assert data['iterations_completed'] == 2
        assert data['completion_reason'] == "test_completion"
        assert 'total_duration_minutes' in data
        assert 'average_tokens_per_message' in data
        assert 'timestamp' in data
        assert data['average_tokens_per_message'] == 19000 / 25
        
    finally:
        # Clean up
        if tmp_path.exists():
            tmp_path.unlink()


def test_print_summary():
    """Test the print summary functionality."""
    tracker = PipelineMetricsTracker()
    tracker.start_run()
    
    # Add test data
    tracker.metrics.total_messages = 20
    tracker.metrics.team_a_messages = 8
    tracker.metrics.team_b_messages = 12
    tracker.metrics.total_input_tokens = 15000
    tracker.metrics.total_output_tokens = 10000
    tracker.metrics.iterations_completed = 3
    tracker.metrics.estimated_cost_usd = 0.0345
    
    time.sleep(0.1)
    tracker.end_run("task_completed")
    
    # Call print_summary
    print("\n--- Testing print_summary output ---")
    tracker.print_summary()
    print("--- End print_summary test ---\n")


def test_progress_tracking():
    """Test the new progress tracking functionality."""
    tracker = PipelineMetricsTracker()
    tracker.start_run()
    
    # Test pipeline start message
    print("\n--- Testing pipeline start message ---")
    tracker.print_pipeline_start("Test Challenge")
    
    # Test Team A progress
    team_a_msg = create_mock_message("principal_scientist", "Planning complete", 100, 80)
    team_a_response = create_mock_response(team_a_msg)
    tracker.track_team_a_result(team_a_response)
    
    print("\n--- Testing Team A progress ---")
    tracker.print_team_progress("Team A", 1)
    
    # Test Team B progress
    team_b_msg = create_mock_message("TEAM_B", "Implementation complete", 150, 120)
    team_b_response = create_mock_response(team_b_msg)
    tracker.track_team_b_result(team_b_response)
    
    print("\n--- Testing Team B progress ---")
    tracker.print_team_progress("Team B", 1)
    
    # Test corrective Team A progress
    corrective_msg = create_mock_message("principal_scientist", "Corrective feedback", 80, 60)
    corrective_response = create_mock_response(corrective_msg)
    tracker.track_team_a_result(corrective_response)
    
    print("\n--- Testing corrective Team A progress ---")
    tracker.print_team_progress("Team A (Corrective)", 1)
    
    print("\n--- End progress tracking test ---\n")


def test_edge_cases():
    """Test edge cases and error conditions."""
    tracker = PipelineMetricsTracker()
    
    # Test with no messages
    tracker.calculate_totals()
    assert tracker.metrics.total_messages == 0
    assert tracker.metrics.total_tokens == 0
    
    # Test with response that has no token usage
    message_no_usage = TextMessage(content="Test", source="test")
    # Don't set models_usage - should be None
    response_no_usage = create_mock_response(message_no_usage)
    
    tracker.track_team_a_result(response_no_usage)
    assert tracker.metrics.team_a_messages == 1
    assert tracker.metrics.team_a_input_tokens == 0
    assert tracker.metrics.team_a_output_tokens == 0
    
    # Test metrics dict
    metrics_dict = tracker.get_metrics_dict()
    assert isinstance(metrics_dict, dict)
    assert 'total_messages' in metrics_dict


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("RUNNING COMPREHENSIVE PIPELINE METRICS TESTS")
    print("=" * 60)
    
    tests = [
        test_pipeline_metrics_tracker_basic,
        test_team_a_tracking,
        test_team_b_tracking,
        test_full_pipeline_simulation,
        test_cost_estimation,
        test_json_output,
        test_print_summary,
        test_progress_tracking,
        test_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print(f"Running {test_func.__name__}...")
            test_func()
            print(f"✅ {test_func.__name__} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {failed} tests failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    # Run all tests when executed directly
    success = run_all_tests()
    exit(0 if success else 1) 