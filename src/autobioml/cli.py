"""Command-line interface for autobioml using Hydra."""

import asyncio
import logging
import sys
from pathlib import Path

import hydra
import hydra.core.hydra_config
from omegaconf import DictConfig, OmegaConf

from .config.loader import load_and_validate_task, load_agents_config
from .tools import get_all_tools
from .pipeline.runner import run_pipeline


logger = logging.getLogger(__name__)


# Get the absolute path to the conf directory
CONF_DIR = Path(__file__).parent.parent.parent / "conf"


@hydra.main(config_path=str(CONF_DIR.absolute()), config_name="config", version_base=None)
def autobioml(cfg: DictConfig) -> None:
    """Main entry point for autobioml CLI."""
    # Convert to dictionary for easier handling
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Get the original working directory from Hydra
    from hydra.core.hydra_config import HydraConfig
    hydra_cfg = HydraConfig.get()
    original_cwd = Path(hydra_cfg.runtime.cwd)
    
    # Get task directory relative to original working directory
    task_dir = (original_cwd / config['task_dir']).resolve()
    if not task_dir.exists():
        raise ValueError(f"Task directory not found: {task_dir}")
    
    # Load task configuration
    task_yaml = task_dir / "task.yaml"
    if not task_yaml.exists():
        raise ValueError(f"task.yaml not found in {task_dir}")
    
    # Load and validate task config
    task_config = load_and_validate_task(task_dir)
    
    # Get Hydra's actual output/run directory
    run_dir = Path(hydra_cfg.runtime.output_dir)
    logger.info(f"Running in directory: {run_dir}")
    
    # Load agents configuration
    if config.get('agents_config'):
        agents_config_file = Path(config['agents_config'])
    else:
        # Default to the package's agents.yaml
        agents_config_file = Path(__file__).parent / "config" / "agents.yaml"
    
    if not agents_config_file.exists():
        raise ValueError(f"Agents configuration not found: {agents_config_file}")
    
    # Load agents config with resource substitution
    agent_configs = load_agents_config(agents_config_file, config['resources'])
    
    # Initialize tools
    tools = get_all_tools()
    
    # Docker configuration with resource-aware timeout
    if 'timeout' in config['resources']:
        config['docker']['timeout'] = config['resources']['timeout']['docker_execution']
    
    logger.info(f"Task: {task_config.display_name}")
    logger.info(f"Model: {config['model']}")
    logger.info(f"Max iterations: {config['max_iterations']}")
    logger.info(f"Docker image: {config['docker']['image']}")
    logger.info(f"GPU specification: {config['gpus']}")
    logger.info(f"Resource profile: {config['resources']['compute']['gpu']['type']}")
    logger.info(f"Public evaluation enabled: {config['enable_public_evaluation']}")
    
    if config['enable_public_evaluation']:
        logger.info(f"Public evaluation requirement: {'REQUIRED' if config['require_public_evaluation'] else 'OPTIONAL'}")
    
    if config['dry_run']:
        logger.info("Dry run mode - configuration validated successfully")
        print("\n=== Configuration Summary ===")
        print(f"Task: {task_config.display_name}")
        print(f"Version: {task_config.version}")
        print(f"Model: {config['model']}")
        print(f"Max iterations: {config['max_iterations']}")
        print(f"Output directory: {run_dir}")
        
        print(f"Resource profile: CPU={config['resources']['compute']['cpu']['cores']} cores, "
              f"RAM={config['resources']['compute']['cpu']['memory_gb']}GB, "
              f"GPU={config['resources']['compute']['gpu']['type']}")
        
        if config['enable_public_evaluation']:
            public_eval_text = f"Enabled (5 attempts, {'REQUIRED' if config['require_public_evaluation'] else 'OPTIONAL'})"
        else:
            public_eval_text = "Disabled"
        print(f"Public evaluation: {public_eval_text}")
        print("\n=== Task Requirements ===")
        for metric in task_config.evaluation.metrics:
            print(f"- {metric.name}: {metric.threshold} on {metric.dataset}")
        return
        
    # Run the pipeline
    asyncio.run(run_pipeline(
        task_config=task_config,
        task_dir=task_dir,
        run_dir=run_dir,
        agent_configs=agent_configs,
        tools=tools,
        model_name=config['model'],
        max_iterations=config['max_iterations'],
        docker_config=config['docker'],
        gpu_spec=config['gpus'],
        team_config=config['team_composition'],
        resource_config=config['resources'],
        enable_public_evaluation=config['enable_public_evaluation'],
        require_public_evaluation=config['require_public_evaluation']
    ))


def main():
    """Entry point for the autobioml CLI."""
    autobioml()


if __name__ == "__main__":
    main() 