"""Main pipeline runner for biomlagents framework."""

import os
import asyncio
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from ..config.schemas import TaskConfig
from ..config.loader import format_task_context, get_private_test_file
from ..agents.factory import create_team_a, create_team_b
from ..tools import initialize_notebook, read_notebook, write_notebook, set_evaluator, read_notebook_summary
from ..evaluation.public_evaluator import PublicEvaluator
from ..agents.base import truncate_message_for_budget
from .metrics import PipelineMetricsTracker


logger = logging.getLogger(__name__)


def cleanup_temp_files(directory: str = ".") -> int:
    """
    Remove temporary code files created during execution.
    
    Args:
        directory: Directory to clean (defaults to current directory)
        
    Returns:
        Number of files removed
    """
    import glob
    
    patterns = ["tmp_code_*", "*.pyc", "__pycache__"]
    
    total_removed = 0
    os.makedirs("agent_code", exist_ok=True)

    for pattern in patterns:
        for file_path in glob.glob(os.path.join(directory, pattern)):
            try:
                if os.path.isfile(file_path):
                    dest_path = os.path.join("agent_code", os.path.basename(file_path))
                    shutil.move(file_path, dest_path)
                    logger.debug(f"Moved to agent_code: {file_path}")
                    total_removed += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    logger.debug(f"Removed directory: {file_path}")
                    total_removed += 1
            except Exception as e:
                logger.warning(f"Error cleaning {file_path}: {e}")
    
    logger.info(f"Cleanup complete. Removed {total_removed} temporary files/directories.")
    return total_removed


def format_prompt(
    notebook_content: str,
    last_team_b_output: Optional[str],
    task_config: TaskConfig,
    enable_public_evaluation: bool = True
) -> str:
    """
    Format a prompt with notebook content and last Team B output.
    
    Args:
        notebook_content: Content of lab notebook
        last_team_b_output: Optional last output from Team B
        task_config: Task configuration
        enable_public_evaluation: Whether public evaluation is enabled
        
    Returns:
        Formatted prompt with all context
    """
    
    prompt_parts = []
    
    # Add task context with conditional public evaluation
    task_context = format_task_context(task_config, enable_public_evaluation=enable_public_evaluation)
    prompt_parts.append("# TASK CONTEXT")
    prompt_parts.append(task_context)
    prompt_parts.append("")
    
    # Add notebook content (TRUNCATED to prevent token explosion)
    if notebook_content:
        # Truncate notebook content to reasonable size
        truncated_notebook = truncate_message_for_budget(notebook_content, max_tokens=20000)
        prompt_parts.append("# LAB NOTEBOOK CONTENT (RECENT ENTRIES)")
        prompt_parts.append(truncated_notebook)
        prompt_parts.append("")
    
    # Add last Team B output if provided (ALREADY SUMMARIZED by EngineerSociety)
    if last_team_b_output:
        # Team B output is already summarized, but apply safety truncation
        truncated_team_b = truncate_message_for_budget(last_team_b_output, max_tokens=12000)
        prompt_parts.append("# LATEST IMPLEMENTATION SUMMARY FROM TEAM B")
        prompt_parts.append(truncated_team_b)
        prompt_parts.append("")
    
    # Add instruction for Team A
    prompt_parts.append("# YOUR CURRENT TASK")
    prompt_parts.append("Review the lab notebook and the latest implementation (if any). Based on the overall goal and current progress:")
    prompt_parts.append("1. Discuss the current state of the project.")
    prompt_parts.append("2. Identify the next logical step to advance the project.")
    prompt_parts.append("3. Create a detailed specification for Team B to implement this next step.")
    prompt_parts.append("4. The Principal Scientist should summarize the discussion and provide the final specification.")
    
    # Add performance reminders if applicable
    if hasattr(task_config, 'evaluation') and task_config.evaluation.metrics:
        prompt_parts.append("")
        prompt_parts.append("# PERFORMANCE TARGETS REMINDER")
        prompt_parts.append("The project must achieve these performance criteria:")
        for metric in task_config.evaluation.metrics:
            prompt_parts.append(f"- {metric.name}: {metric.threshold} on {metric.dataset}")
    
    return "\n".join(prompt_parts)


async def run_pipeline(
    task_config: TaskConfig,
    task_dir: Path,
    run_dir: Path,
    agent_configs: Dict[str, Any],
    tools: Dict[str, Any],
    model_name: str = "gpt-4.1-mini",
    max_iterations: int = 25,
    docker_config: Dict[str, Any] = None,
    gpu_spec: str = "all",
    team_config: Dict[str, Any] = None,
    resource_config: Dict[str, Any] = None,
    enable_public_evaluation: bool = False,
    require_public_evaluation: bool = True
) -> None:
    """
    Main pipeline execution.
    
    Args:
        task_config: Validated task configuration
        task_dir: Path to the challenge directory
        run_dir: Path to the run directory (where we're executing)
        agent_configs: Agent configurations
        tools: Available tools
        model_name: Model to use
        max_iterations: Maximum Team A-B iterations
        docker_config: Docker configuration dictionary
        gpu_spec: GPU specification
        team_config: Team composition configuration
        resource_config: Resource configuration for the deployment
        enable_public_evaluation: Whether to enable public test evaluation
        require_public_evaluation: Whether to require public evaluation before final submission
                                   (only relevant if enable_public_evaluation=True)
    """
    # Initialize metrics tracker
    metrics_tracker = PipelineMetricsTracker()
    metrics_tracker.start_run()
    
    # Print pipeline start message
    metrics_tracker.print_pipeline_start(task_config.display_name)
    
    logger.info(f"Starting pipeline for task: {task_config.display_name}")
    logger.info(f"Run directory: {run_dir}")
    
    # Filter tools based on configuration
    # Remove evaluation tools if public evaluation is disabled
    if not enable_public_evaluation:
        # Create a copy of tools without evaluation tools
        filtered_tools = {k: v for k, v in tools.items() if k != "evaluate_on_public_test"}
        tools = filtered_tools
        logger.info("Public evaluation DISABLED - evaluation tools removed from agent toolset")
    
    # Initialize evaluator if enabled and test labels exist
    evaluator = None
    private_sample_ids = None
    if enable_public_evaluation:
        eval_dir = task_dir / "data/eval"
        public_labels_path = eval_dir / "meta_heldout_public.arrow"
        private_labels_path = eval_dir / "meta_heldout_private.arrow"
        
        if public_labels_path.exists():
            logger.info("Initializing public evaluator")
            evaluator = PublicEvaluator(public_labels_path)
            set_evaluator(evaluator)
            logger.info(f"Public evaluation {'REQUIRED' if require_public_evaluation else 'OPTIONAL'} before final submission")
            
            # Load private sample IDs for validation (NOT labels)
            if private_labels_path.exists():
                import pandas as pd
                private_df = pd.read_feather(private_labels_path)
                private_sample_ids = set(private_df.index.tolist())
                del private_df  # Immediately delete to ensure no labels in memory
        else:
            logger.warning("Evaluation data not found, evaluation tools will be disabled")
            # Remove evaluation tools if data not found
            filtered_tools = {k: v for k, v in tools.items() if k != "evaluate_on_public_test"}
            tools = filtered_tools
    else:
        logger.info("Public evaluation DISABLED - agents will work with private test set only")
    
    # Copy task data to run directory
    for data_file in task_config.available_data.agent_data:
        # Skip public test data files if public evaluation is disabled
        if not enable_public_evaluation and 'public' in data_file.path.lower():
            logger.info(f"Skipping public test data file (public evaluation disabled): {data_file.path}")
            continue
            
        src = task_dir / data_file.path
        dst = run_dir / Path(data_file.path).name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            logger.info(f"Copied data file: {src} -> {dst}")
    
    # Initialize notebook in run directory
    notebook_path = run_dir / "lab_notebook.md"
    initialize_notebook(notebook_path)
    
    # Create task context for agents with resource config
    task_context = format_task_context(task_config, resource_config, enable_public_evaluation)
    

    # Ensure work_dir is set to run_dir if not provided
    if "work_dir" not in docker_config:
        docker_config["work_dir"] = str(run_dir)
    
    # Initialize teams with enhanced configuration
    team_a = await create_team_a(
        agent_configs=agent_configs,
        tools=tools,
        model_name=model_name,
        task_context=task_context,
        team_composition=team_config['planning_team'],
        enable_public_evaluation=enable_public_evaluation
    )
    
    team_b, docker_executor = await create_team_b(
        agent_configs=agent_configs,
        tools=tools,
        model_name=model_name,
        task_context=task_context,
        docker_config=docker_config,
        gpu_spec=gpu_spec,
        working_dir=str(run_dir),
        team_composition={
            "implementation_team": team_config['implementation_team'],
            "review_team": team_config['review_team']
        },
        enable_public_evaluation=enable_public_evaluation
    )
    
    try:
        # Run main loop
        iteration = 0
        last_team_b_output = None
        project_complete = False
        completion_token = "TASK_COMPLETE"  # Generic completion token
        completion_reason = "max_iterations_reached"
        
        while iteration < max_iterations and not project_complete:
            iteration += 1
            logger.info(f"Starting iteration {iteration}/{max_iterations}")
            
            # Step 1: Team A Planning Phase
            logger.info(f"Iteration {iteration}: Team A Planning Phase")
            
            # Read current notebook content (TRUNCATED to prevent token explosion)
            notebook_content = read_notebook_summary(notebook_path, max_chars=15000)
            
            # Format the prompt for Team A with current context
            team_a_prompt = format_prompt(
                notebook_content=notebook_content,
                last_team_b_output=last_team_b_output,
                task_config=task_config,
                enable_public_evaluation=enable_public_evaluation
            )
            
            # Run Team A to get next steps plan
            team_a_message = TextMessage(content=team_a_prompt, source="User")
            team_a_response = await team_a.on_messages([team_a_message], CancellationToken())
            
            # Track Team A metrics
            metrics_tracker.track_team_a_result(team_a_response)
            
            # Print Team A progress
            metrics_tracker.print_team_progress("Team A", iteration)
            
            # Check if the project is complete
            if completion_token in team_a_response.chat_message.content:
                # Additional validation: ensure this is actually a completion and not mid-discussion
                response_content = team_a_response.chat_message.content
                
                # Check if this looks like a real completion (mentions actual results/files)
                # TODO: This is a bit janky... We should have a structured checklist we look for.
                completion_indicators = [
                    "all task requirements have been met",
                    "predictions.arrow file has been validated",
                    "performance targets have been met",
                    "evaluation results have been documented in output report",
                    "all deliverables are complete and verified"
                ]
                
                has_completion_evidence = any(indicator.lower() in response_content.lower() 
                                             for indicator in completion_indicators)
                
                if not has_completion_evidence:
                    # This looks like premature completion signaling
                    logger.warning(f"Team A used TASK_COMPLETE but message doesn't indicate actual completion. Ignoring completion signal.")
                    
                    # Create detailed message about what's missing
                    missing_evidence_msg = f"""WARNING: Team A used TASK_COMPLETE but the message lacks evidence of actual completion.

Expected completion evidence includes affirming ALL of the following in your message:
- "All task requirements have been met."
- "predictions.arrow file has been validated."
- "Performance targets have been met."
- "Evaluation results have been documented in output report."
- "All deliverables are complete and verified."

Found not all of these indicators are present in the completion message. The project appears incomplete.

REQUIREMENTS CHECKLIST:
□ Generate predictions.arrow file with correct format
□ Achieve performance targets
□ Document evaluation results with specific metrics
□ Ensure all required files are created
{'□ Complete public evaluation using evaluate_on_public_test() tool' if enable_public_evaluation and require_public_evaluation else ''}

Team A should verify all requirements are met before using TASK_COMPLETE."""
                    
                    write_notebook(
                        entry=missing_evidence_msg,
                        entry_type="NOTE",
                        source="pipeline_system",
                        notebook_path=notebook_path
                    )
                    # Continue normally, ignore the TASK_COMPLETE signal
                else:
                    # Check if predictions.arrow exists and has correct format
                    predictions_file = run_dir / "predictions.arrow"
                    target_column = task_config.evaluation.target_column or "predicted_value"
                    
                    # Validate predictions file
                    is_valid, validation_message = validate_predictions_file(predictions_file, private_sample_ids, target_column)
                    
                    # Check if public evaluation was required but not used
                    public_eval_check_passed = True
                    public_eval_message = ""
                    
                    if enable_public_evaluation and require_public_evaluation and evaluator:
                        if evaluator.get_public_attempts() == 0:
                            public_eval_check_passed = False
                            public_eval_message = "Public evaluation was REQUIRED but never used. You must validate your model on the public test set before final submission."
                        else:
                            public_eval_message = f"✓ Public evaluation requirement satisfied ({evaluator.get_public_attempts()} attempts used)"
                    
                    if is_valid and public_eval_check_passed:
                        # File exists, has correct format, and public eval requirements met
                        logger.info(f"Project completion verified: {validation_message}. {public_eval_message} Project is complete.")
                        project_complete = True
                        completion_reason = "task_completed"
                        completion_entry = f"PROJECT COMPLETED: All requirements have been satisfied. {validation_message}"
                        if public_eval_message:
                            completion_entry += f" {public_eval_message}"
                        write_notebook(
                            entry=completion_entry,
                            entry_type="COMPLETION",
                            source="principal_scientist",
                            notebook_path=notebook_path
                        )
                    else:
                        # File missing, invalid format, or public eval requirements not met
                        error_reasons = []
                        if not is_valid:
                            error_reasons.append(f"predictions.arrow validation failed: {validation_message}")
                        if not public_eval_check_passed:
                            error_reasons.append(public_eval_message)
                        
                        combined_error = "; ".join(error_reasons)
                        logger.warning(f"Team A declared completion but requirements not met: {combined_error}")
                        write_notebook(
                            entry=f"WARNING: Team A declared completion but requirements not met: {combined_error}",
                            entry_type="NOTE",
                            source="pipeline_system",
                            notebook_path=notebook_path
                        )
                        
                        # Create detailed corrective message for Team A
                        corrective_parts = ["THE TASK IS NOT COMPLETE."]
                        
                        if not is_valid:
                            corrective_parts.append(f"Predictions file validation failed: {validation_message}")
                            corrective_parts.append(f"""
REQUIRED FORMAT for predictions.arrow:
- Must be a valid Arrow/Feather file
- Must contain exactly these columns: 'sample_id' and '{target_column}'
- sample_id: unique identifiers for each sample (no duplicates, no null values)
- {target_column}: numeric predictions (no null values)
- Must contain predictions for ALL samples in {get_private_test_file(task_config)} (private test set ONLY)
- Do NOT include predictions for public test set samples if public evaluation is enabled""")
                        
                        if not public_eval_check_passed:
                            corrective_parts.append(public_eval_message)
                            corrective_parts.append("You MUST use the evaluate_on_public_test() tool to validate your model before declaring completion.")
                        
                        if enable_public_evaluation:
                            corrective_parts.append("""
REMINDER OF WORKFLOW:
1. Build model using betas.arrow and metadata.arrow (training data)
2. Test on public set using evaluate_on_public_test() (REQUIRED)
3. Generate final predictions.arrow using betas_heldout_private.arrow
4. Ensure predictions.arrow contains predictions for ALL private test samples""")
                        else:
                            corrective_parts.append("""
REMINDER OF WORKFLOW:
1. Build model using betas.arrow and metadata.arrow (training data)
2. Generate final predictions.arrow using betas_heldout_private.arrow
3. Ensure predictions.arrow contains predictions for ALL private test samples""")
                        
                        corrective_parts.append("DO NOT mark 'TASK_COMPLETE' until all requirements are satisfied.")
                        
                        corrective_message = TextMessage(
                            content="\n\n".join(corrective_parts),
                            source="User"
                        )
                        
                        # Restart Team A with the corrective message
                        team_a_response = await team_a.on_messages([corrective_message], CancellationToken())
                        
                        # Track additional Team A metrics from corrective run
                        metrics_tracker.track_team_a_result(team_a_response)
                        
                        # Print corrective Team A progress
                        metrics_tracker.print_team_progress("Team A (Corrective)", iteration)
            else:
                # No completion token found - continue normally
                pass
            
            # Log and record Team A's plan
            logger.info(f"Team A planning complete for iteration {iteration}")
            
            plan_content = team_a_response.chat_message.content
            
            write_notebook(
                entry=plan_content,
                entry_type="PLAN",
                source="principal_scientist",
                notebook_path=notebook_path
            )
            
            # If the project is complete, skip Team B implementation
            if project_complete:
                logger.info("Project is complete. Skipping Team B implementation.")
                break
            
            # Step 2: Team B Implementation Phase
            logger.info(f"Iteration {iteration}: Team B Implementation Phase")
            
            # Forward Team A's plan to Team B
            team_b_message = team_a_response.chat_message
            team_b_response = await team_b.on_messages([team_b_message], CancellationToken())
            
            # Track Team B metrics
            metrics_tracker.track_team_b_result(team_b_response)
            
            # Print Team B progress
            metrics_tracker.print_team_progress("Team B", iteration)
            
            # Store Team B's output for next iteration
            last_team_b_output = team_b_response.chat_message.content
            
            # Get the last message from Team B (usually the critic's review)
            if hasattr(team_b_response, 'inner_messages') and team_b_response.inner_messages:
                critic_review = team_b_response.inner_messages[-1]
            else:
                critic_review = team_b_response.chat_message
            
            # Clean up temporary files
            cleanup_temp_files(str(run_dir))
            
            # Log completion of iteration
            logger.info(f"Completed iteration {iteration}/{max_iterations}")
            
            # Mark iteration as complete in metrics
            metrics_tracker.complete_iteration()
            
            # Write iteration summary to notebook
            write_notebook(
                entry=f"Completed iteration {iteration}. Team B implementation summary recorded.",
                entry_type="OUTPUT",
                source="TEAM_B",
                notebook_path=notebook_path
            )
        
        # Set completion reason if not already set
        if not project_complete:
            completion_reason = "max_iterations_reached"
        
        logger.info("Pipeline completed successfully")
        
    finally:
        # End metrics tracking
        metrics_tracker.end_run(completion_reason)
        
        # Estimate cost
        metrics_tracker.estimate_cost(model_name)
        
        # Print summary to console
        metrics_tracker.print_summary()
        
        # Save metrics to JSON file
        output_stats_path = run_dir / "output_stats.json"
        metrics_tracker.save_to_json(output_stats_path)
        logger.info(f"Pipeline metrics saved to: {output_stats_path}")
        
        # Run automatic evaluation if predictions.arrow exists
        predictions_file = run_dir / "predictions.arrow"
        if predictions_file.exists():
            logger.info("🎯 Predictions file found - running automatic evaluation...")
            evaluation_success = run_evaluation_script(predictions_file, task_dir, run_dir)
            metrics_tracker.mark_evaluation_completed(evaluation_success)
            if evaluation_success:
                print("\n🎉 Automatic evaluation completed! Check the final_eval_results directory for detailed results.")
            else:
                print("\n⚠️ Automatic evaluation failed. You can run it manually using the evaluation script.")
        else:
            logger.info("📝 No predictions.arrow file found - skipping automatic evaluation")
            metrics_tracker.mark_evaluation_completed(False)
        
        # Close the docker executor
        await docker_executor.stop()
        logger.info("Docker executor stopped")


def validate_predictions_file(predictions_path: Path, expected_sample_ids: set = None, target_column: str = "predicted_value") -> tuple[bool, str]:
    """
    Validate that predictions.arrow has the correct format and contains predictions for the private test set.
    
    The final predictions.arrow file should contain predictions ONLY for the private test set samples.
    Agents should have generated these predictions using betas_heldout_private.arrow.
    
    Args:
        predictions_path: Path to the predictions.arrow file
        expected_sample_ids: Set of expected sample IDs from private test set (optional)
        target_column: Target column for evaluation
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        import pandas as pd
        
        # Check if file exists
        if not predictions_path.exists():
            return False, "predictions.arrow file does not exist"
        
        # Try to load the file
        try:
            df = pd.read_feather(predictions_path)
        except Exception as e:
            return False, f"Cannot read predictions.arrow file: {str(e)}"
        
        # Check required columns
        required_columns = {'sample_id', target_column}
        actual_columns = set(df.columns)
        
        if not required_columns.issubset(actual_columns):
            missing_columns = required_columns - actual_columns
            return False, f"Missing required columns: {missing_columns}. Found columns: {list(actual_columns)}"
        
        # Check data types and content
        if df.empty:
            return False, "predictions.arrow file is empty"
        
        # Check for null values in required columns
        if df['sample_id'].isnull().any():
            return False, "sample_id column contains null values"
        
        if df[target_column].isnull().any():
            return False, f"{target_column} column contains null values"
        
        # Check that predicted_age is numeric
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            return False, f"{target_column} column must be numeric"
        
        # Check for duplicate sample_ids
        if df['sample_id'].duplicated().any():
            return False, "sample_id column contains duplicate values"
        
        # Check that sample_id values are strings and look like valid IDs
        # Sample IDs should be strings, not numeric values
        sample_ids = df['sample_id']
        
        # Check if sample_id is numeric type (this is wrong)
        if pd.api.types.is_numeric_dtype(sample_ids):
            return False, "sample_id column contains numeric values instead of string identifiers. Sample IDs should be strings like 'GSM123456', not numbers."
        
        # Check that all sample IDs are strings
        non_string_ids = []
        for idx, sid in enumerate(sample_ids):
            if not isinstance(sid, str):
                non_string_ids.append((idx, sid, type(sid).__name__))
                if len(non_string_ids) >= 5:  # Show only first 5 examples
                    break
        
        if non_string_ids:
            examples = "; ".join([f"row {idx}: {sid} (type: {t})" for idx, sid, t in non_string_ids])
            return False, f"sample_id column must contain string identifiers, found non-string values: {examples}"
        
        # Check that sample IDs look like valid identifiers (not just numeric strings)
        # Valid sample IDs typically start with letters (e.g., GSM, E-GEOD, etc.)
        numeric_looking_ids = []
        for sid in sample_ids[:10]:  # Check first 10
            try:
                # If we can convert to float, it's probably not a valid sample ID
                float(sid)
                numeric_looking_ids.append(sid)
            except (ValueError, TypeError):
                # Good - it's not a pure number
                pass
        
        if numeric_looking_ids:
            examples = ", ".join(numeric_looking_ids[:5])
            return False, f"sample_id values appear to be numeric strings: {examples}. Expected identifiers like 'GSM123456', 'E-GEOD-12345', etc."
        
        # Check sample IDs match expected if provided
        if expected_sample_ids is not None:
            predicted_sample_ids = set(df['sample_id'])
            missing_samples = expected_sample_ids - predicted_sample_ids
            extra_samples = predicted_sample_ids - expected_sample_ids
            
            if missing_samples:
                return False, f"Missing predictions for {len(missing_samples)} expected samples from held-out test set"
            
            if extra_samples:
                return False, f"Found {len(extra_samples)} unexpected sample IDs not in held-out test set"
        
        logger.info(f"Predictions file validation passed: {len(df)} predictions with columns {list(df.columns)}")
        return True, f"Valid predictions file with {len(df)} predictions"
        
    except Exception as e:
        return False, f"Unexpected error validating predictions file: {str(e)}"


def run_evaluation_script(predictions_path: Path, task_dir: Path, run_dir: Path) -> bool:
    """
    Run the evaluation script for the given predictions file.
    
    Args:
        predictions_path: Path to the predictions.arrow file
        task_dir: Path to the challenge directory 
        run_dir: Path to the run directory (for working directory)
        
    Returns:
        True if evaluation succeeded, False otherwise
    """
    try:
        # Find the evaluation script in the task directory
        eval_script_path = task_dir / "scripts" / "evaluate.py"
        if not eval_script_path.exists():
            logger.warning(f"Evaluation script not found at: {eval_script_path}")
            return False
        
        # Set up evaluation data path
        eval_data_path = task_dir / "data" / "eval"
        if not eval_data_path.exists():
            logger.warning(f"Evaluation data directory not found at: {eval_data_path}")
            return False
        
        logger.info(f"Running automatic evaluation...")
        logger.info(f"Evaluation script: {eval_script_path}")
        logger.info(f"Predictions file: {predictions_path}")
        logger.info(f"Evaluation data: {eval_data_path}")
        
        # Construct the command to run the evaluation script
        command = [
            "python",
            str(eval_script_path),
            "--predictions", str(predictions_path),
            "--eval-data", str(eval_data_path)
        ]
        
        # Run the evaluation script in the run directory
        result = subprocess.run(
            command, 
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Log the output
        if result.stdout:
            logger.info(f"Evaluation output:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Evaluation stderr:\n{result.stderr}")
        
        # Check if the script exited with a success code
        if result.returncode == 0:
            logger.info("✅ Automatic evaluation completed successfully!")
            return True
        else:
            logger.error(f"❌ Evaluation script failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Evaluation script timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Error running evaluation script: {str(e)}")
        return False 