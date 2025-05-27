"""Configuration loading and validation for bioagents."""

import yaml
from pathlib import Path
from pydantic import ValidationError
from typing import Dict, Any, Optional
import logging

from .schemas import TaskConfig, AgentsConfig, AutonomousWorkflow, ReferenceSection, AvailableData, DataFile, EvaluationConfig, EvaluationMetric, DockerConfig


logger = logging.getLogger(__name__)


def load_and_validate_task(task_dir: Path) -> TaskConfig:
    """
    Load and validate task configuration.
    
    Args:
        task_dir: Path to the challenge directory
        
    Returns:
        Validated TaskConfig instance
        
    Raises:
        FileNotFoundError: If task.yaml not found
        ValueError: If task configuration is invalid
    """
    task_yaml_path = task_dir / "task.yaml"
    
    if not task_yaml_path.exists():
        raise FileNotFoundError(f"task.yaml not found in {task_dir}")
    
    with open(task_yaml_path, 'r') as f:
        task_data = yaml.safe_load(f)
    
    # Convert nested dictionaries to proper schema objects
    if 'autonomous_workflow' in task_data and isinstance(task_data['autonomous_workflow'], dict):
        task_data['autonomous_workflow'] = AutonomousWorkflow(**task_data['autonomous_workflow'])
    
    if 'reference' in task_data and isinstance(task_data['reference'], dict):
        reference_sections = {}
        for key, value in task_data['reference'].items():
            if isinstance(value, dict) and 'text' in value:
                reference_sections[key] = ReferenceSection(**value)
            else:
                reference_sections[key] = ReferenceSection(text=str(value))
        task_data['reference'] = reference_sections
    
    if 'available_data' in task_data and isinstance(task_data['available_data'], dict):
        agent_data = [DataFile(**item) for item in task_data['available_data'].get('agent_data', [])]
        eval_data = [DataFile(**item) for item in task_data['available_data'].get('eval_data', [])]
        task_data['available_data'] = AvailableData(agent_data=agent_data, eval_data=eval_data)
    
    if 'evaluation' in task_data and isinstance(task_data['evaluation'], dict):
        metrics = [EvaluationMetric(**m) for m in task_data['evaluation'].get('metrics', [])]
        required_outputs = task_data['evaluation'].get('required_outputs', [])
        task_data['evaluation'] = EvaluationConfig(metrics=metrics, required_outputs=required_outputs)
    
    if 'docker' in task_data and isinstance(task_data['docker'], dict):
        task_data['docker'] = DockerConfig(**task_data['docker'])
    
    # Validate using Pydantic schema
    try:
        task_config = TaskConfig(**task_data)
    except ValidationError as e:
        raise ValueError(f"Invalid task configuration: {e}")
    
    # Validate data files exist
    for data_file in task_config.available_data.agent_data:
        file_path = task_dir / data_file.path
        if not file_path.exists():
            raise FileNotFoundError(f"Agent data file not found: {file_path}")
    
    for data_file in task_config.available_data.eval_data:
        file_path = task_dir / data_file.path
        if not file_path.exists():
            raise FileNotFoundError(f"Evaluation data file not found: {file_path}")
    
    logger.info(f"Successfully loaded and validated task: {task_config.display_name}")
    return task_config


def load_agent_configs(config_path: Path) -> Dict[str, Any]:
    """
    Load agent configurations from YAML file.
    
    Args:
        config_path: Path to agents.yaml file
        
    Returns:
        Dictionary of agent configurations
        
    Raises:
        FileNotFoundError: If config file not found
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Agent config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    # Basic validation
    if 'agents' not in configs:
        raise ValueError("Agent config must contain 'agents' key")
    
    logger.info(f"Loaded {len(configs['agents'])} agent configurations")
    return configs


def substitute_config_variables(config_str: str, variables: Dict[str, Any]) -> str:
    """
    Substitute variables in configuration strings.
    
    Args:
        config_str: Configuration string with ${variable} placeholders
        variables: Dictionary of variable values to substitute
        
    Returns:
        String with variables substituted
    """
    import re
    
    def replace_var(match):
        var_path = match.group(1)
        parts = var_path.split('.')
        value = variables
        
        try:
            for part in parts:
                value = value[part]
            return str(value)
        except (KeyError, TypeError):
            # If variable not found, leave as is
            return match.group(0)
    
    # Find and replace ${variable.path} patterns
    return re.sub(r'\$\{([^}]+)\}', replace_var, config_str)


def load_agents_config(config_path: Path, resource_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load agent configurations from YAML file with resource variable substitution.
    
    Args:
        config_path: Path to the agents YAML configuration file
        resource_config: Optional resource configuration for variable substitution
        
    Returns:
        Loaded and processed agent configurations
    """
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Substitute resource variables if provided
    if resource_config:
        content = substitute_config_variables(content, {'resources': resource_config})
    
    # Parse the YAML
    config = yaml.safe_load(content)
    
    return config


def format_task_context(task_config: TaskConfig, resource_config: Optional[Dict[str, Any]] = None) -> str:
    """
    Format task configuration into context string for agents.
    
    Args:
        task_config: The task configuration
        resource_config: Optional resource configuration for variable substitution
        
    Returns:
        Formatted task context string
    """
    # Format available data descriptions - use just filenames since they're copied to working directory
    data_descriptions = []
    for data_file in task_config.available_data.agent_data:
        # Extract just the filename from the path
        filename = Path(data_file.path).name
        data_descriptions.append(f"- {filename}: {data_file.description}")
    
    context = f"""You are working on: {task_config.display_name}

Task Description:
{task_config.task_description}

Project Goal:
{task_config.project_goal}

Project Context:
{task_config.project_context}

Available Data (in current working directory):
{chr(10).join(data_descriptions)}

Data Completeness:
{task_config.data_completeness}

Autonomous Workflow Approach:
{task_config.autonomous_workflow.approach}

Lab Notebook Guidelines:
{task_config.lab_notebook_guidelines}

Remember to document your findings in the lab notebook and create an analysis report
once you meet the performance criteria."""
    
    # Substitute resource variables if provided
    if resource_config:
        context = substitute_config_variables(context, {'resources': resource_config})
    
    return context 