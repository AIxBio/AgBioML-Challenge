"""Main pipeline runner for bioagents framework."""

import os
import asyncio
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from ..config.schemas import TaskConfig
from ..config.loader import format_task_context
from ..agents.factory import create_team_a, create_team_b
from ..tools import initialize_notebook, read_notebook, write_notebook


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
    for pattern in patterns:
        for file_path in glob.glob(os.path.join(directory, pattern)):
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Removed: {file_path}")
                    total_removed += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    logger.debug(f"Removed directory: {file_path}")
                    total_removed += 1
            except Exception as e:
                logger.warning(f"Error removing {file_path}: {e}")
    
    logger.info(f"Cleanup complete. Removed {total_removed} temporary files/directories.")
    return total_removed


def format_prompt(
    notebook_content: str,
    last_team_b_output: Optional[str],
    task_config: TaskConfig
) -> str:
    """
    Format a prompt with notebook content and last Team B output.
    
    Args:
        notebook_content: Content of lab notebook
        last_team_b_output: Optional last output from Team B
        task_config: Task configuration
        
    Returns:
        Formatted prompt with all context
    """
    prompt_parts = []
    
    # Add task context
    task_context = format_task_context(task_config)
    prompt_parts.append("# TASK CONTEXT")
    prompt_parts.append(task_context)
    prompt_parts.append("")
    
    # Add notebook content
    if notebook_content:
        prompt_parts.append("# LAB NOTEBOOK CONTENT")
        prompt_parts.append(notebook_content)
        prompt_parts.append("")
    
    # Add last Team B output if provided
    if last_team_b_output:
        prompt_parts.append("# LATEST IMPLEMENTATION REPORT FROM TEAM B")
        prompt_parts.append(last_team_b_output)
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
    model_name: str = "gpt-4.1",
    max_iterations: int = 25,
    docker_config: Dict[str, Any] = None,
    gpu_spec: str = "all",
    team_config: Dict[str, Any] = None,
    resource_config: Dict[str, Any] = None
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
    """
    logger.info(f"Starting pipeline for task: {task_config.display_name}")
    logger.info(f"Run directory: {run_dir}")
    
    # Copy task data to run directory
    for data_file in task_config.available_data.agent_data:
        src = task_dir / data_file.path
        dst = run_dir / Path(data_file.path).name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            logger.info(f"Copied data file: {src} -> {dst}")
    
    # Initialize notebook in run directory
    notebook_path = run_dir / "lab_notebook.md"
    initialize_notebook(notebook_path)
    
    # Create task context for agents with resource config
    task_context = format_task_context(task_config, resource_config)
    
    # Set default configurations if not provided
    if docker_config is None:
        docker_config = {
            "image": "millerh1/bioagents:latest",
            "timeout": 3600,
            "work_dir": str(run_dir)
        }
    
    if team_config is None:
        team_config = {
            "planning_team": ["principal_scientist", "bioinformatics_expert", "ml_expert"],
            "implementation_team": ["implementation_engineer"],
            "review_team": ["data_science_critic"]
        }
    
    # Initialize teams with enhanced configuration
    team_a = await create_team_a(
        agent_configs=agent_configs,
        tools=tools,
        model_name=model_name,
        task_context=task_context,
        team_composition=team_config.get("planning_team", ["principal_scientist", "bioinformatics_expert", "ml_expert"])
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
            "implementation_team": team_config.get("implementation_team", ["implementation_engineer"]),
            "review_team": team_config.get("review_team", ["data_science_critic"])
        }
    )
    
    try:
        # Run main loop
        iteration = 0
        last_team_b_output = None
        project_complete = False
        completion_token = "TASK_COMPLETE"  # Generic completion token
        
        while iteration < max_iterations and not project_complete:
            iteration += 1
            logger.info(f"Starting iteration {iteration}/{max_iterations}")
            
            # Step 1: Team A Planning Phase
            logger.info(f"Iteration {iteration}: Team A Planning Phase")
            
            # Read current notebook content
            notebook_content = read_notebook(notebook_path)
            
            # Format the prompt for Team A with current context
            team_a_prompt = format_prompt(
                notebook_content=notebook_content,
                last_team_b_output=last_team_b_output,
                task_config=task_config
            )
            
            # Run Team A to get next steps plan
            team_a_message = TextMessage(content=team_a_prompt, source="User")
            team_a_response = await team_a.on_messages([team_a_message], CancellationToken())
            
            # Check if the project is complete
            if completion_token in team_a_response.chat_message.content:
                logger.info("Detected project completion token. All requirements have been met.")
                project_complete = True
                write_notebook(
                    entry="PROJECT COMPLETED: All requirements have been satisfied. The Principal Scientist has verified completion.",
                    entry_type="COMPLETION",
                    source="principal_scientist",
                    notebook_path=notebook_path
                )
            
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
            
            # Write iteration summary to notebook
            write_notebook(
                entry=f"Completed iteration {iteration}. Team B implementation summary recorded.",
                entry_type="OUTPUT",
                source="TEAM_B",
                notebook_path=notebook_path
            )
        
        logger.info("Pipeline completed successfully")
        
    finally:
        # Close the docker executor
        await docker_executor.stop()
        logger.info("Docker executor stopped") 