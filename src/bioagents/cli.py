"""Command-line interface for bioagents using Hydra."""

import asyncio
import logging
import sys
from pathlib import Path

import hydra
import hydra.core.hydra_config
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir

from .config.loader import load_and_validate_task, load_agents_config
from .tools import get_all_tools
from .pipeline.runner import run_pipeline


logger = logging.getLogger(__name__)


# Get the absolute path to the conf directory
CONF_DIR = Path(__file__).parent.parent.parent / "conf"


@hydra.main(config_path=str(CONF_DIR.absolute()), config_name="config", version_base=None)
def bioagents(cfg: DictConfig) -> None:
    """Main entry point for bioagents CLI."""
    # Convert to dictionary for easier handling
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Get the original working directory from Hydra
    from hydra.core.hydra_config import HydraConfig
    hydra_cfg = HydraConfig.get()
    original_cwd = Path(hydra_cfg.runtime.cwd)
    
    # Get task directory relative to original working directory
    task_dir_str = config['task_dir']
    task_dir = (original_cwd / task_dir_str).resolve()
    if not task_dir.exists():
        raise ValueError(f"Task directory not found: {task_dir}")
    
    # Load task configuration
    task_yaml = task_dir / "task.yaml"
    if not task_yaml.exists():
        raise ValueError(f"task.yaml not found in {task_dir}")
    
    # Load and validate task config
    task_config = load_and_validate_task(task_dir)
    
    # Get Hydra's output directory
    run_dir = Path.cwd()  # Hydra changes to output directory
    logger.info(f"Running in directory: {run_dir}")
    
    # Load agents configuration
    agents_config_path = config.get('agents_config')
    if agents_config_path:
        agents_config_file = Path(agents_config_path)
    else:
        # Default to the package's agents.yaml
        agents_config_file = Path(__file__).parent / "config" / "agents.yaml"
    
    if not agents_config_file.exists():
        raise ValueError(f"Agents configuration not found: {agents_config_file}")
    
    # Load resource configuration if available
    resource_config = config.get('resources', {})
    
    # Load agents config with resource substitution
    agent_configs = load_agents_config(agents_config_file, resource_config)
    
    # Initialize tools
    tools = get_all_tools()
    
    # Get other configuration values
    model_name = config.get('model', 'gpt-4.1')
    max_iterations = config.get('max_iterations', 25)
    dry_run = config.get('dry_run', False)
    
    # Docker configuration with resource-aware timeout
    docker_config = config.get('docker', {})
    if resource_config and 'timeout' in resource_config:
        docker_config['timeout'] = resource_config['timeout'].get('docker_execution', 3600)
    
    gpu_spec = config.get('gpus', 'all')
    
    # Team configuration
    team_config = config.get('teams', None)
    
    logger.info(f"Task: {task_config.display_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Max iterations: {max_iterations}")
    logger.info(f"Docker image: {docker_config.get('image', 'millerh1/bioagents:latest')}")
    logger.info(f"GPU specification: {gpu_spec}")
    logger.info(f"Resource profile: {resource_config.get('compute', {}).get('gpu', {}).get('type', 'default')}")
    
    if dry_run:
        logger.info("Dry run mode - configuration validated successfully")
        print("\n=== Configuration Summary ===")
        print(f"Task: {task_config.display_name}")
        print(f"Version: {task_config.version}")
        print(f"Model: {model_name}")
        print(f"Max iterations: {max_iterations}")
        print(f"Output directory: {run_dir}")
        print(f"Resource profile: CPU={resource_config.get('compute', {}).get('cpu', {}).get('cores', 'N/A')} cores, "
              f"RAM={resource_config.get('compute', {}).get('cpu', {}).get('memory_gb', 'N/A')}GB, "
              f"GPU={resource_config.get('compute', {}).get('gpu', {}).get('type', 'none')}")
        print("\n=== Task Requirements ===")
        for metric in task_config.evaluation.metrics:
            print(f"- {metric.name}: {metric.threshold} on {metric.dataset}")
        return
    
    # Run the pipeline with resource configuration
    asyncio.run(run_pipeline(
        task_config=task_config,
        task_dir=task_dir,
        run_dir=run_dir,
        agent_configs=agent_configs,
        tools=tools,
        model_name=model_name,
        max_iterations=max_iterations,
        docker_config=docker_config,
        gpu_spec=gpu_spec,
        team_config=team_config,
        resource_config=resource_config
    ))


def main():
    """Entry point for the bioagents CLI."""
    bioagents()


if __name__ == "__main__":
    main() 