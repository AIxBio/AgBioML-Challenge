"""Configuration loading and validation for biomlagents."""

import yaml
from pathlib import Path
from pydantic import ValidationError
from typing import Dict, Any, Optional, List
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
    
    
    if 'available_data' in task_data and isinstance(task_data['available_data'], dict):
        agent_data = [DataFile(**item) for item in task_data['available_data'].get('agent_data', [])]
        eval_data = [DataFile(**item) for item in task_data['available_data'].get('eval_data', [])]
        task_data['available_data'] = AvailableData(agent_data=agent_data, eval_data=eval_data)
    
    if 'evaluation' in task_data and isinstance(task_data['evaluation'], dict):
        metrics = [EvaluationMetric(**m) for m in task_data['evaluation'].get('metrics', [])]
        required_outputs = task_data['evaluation'].get('required_outputs', [])
        process = task_data['evaluation'].get('process', None)
        task_data['evaluation'] = EvaluationConfig(
            process=process, 
            metrics=metrics, 
            required_outputs=required_outputs,
            workflow_template=task_data['evaluation'].get('workflow_template', None),
            target_column=task_data['evaluation'].get('target_column', None),
            custom_workflow=task_data['evaluation'].get('custom_workflow', None)
        )
    
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


def format_task_context(task_config: TaskConfig, resource_config: Dict[str, Any] = None, enable_public_evaluation: bool = True) -> str:
    """
    Format the task configuration into a readable context string for agents.
    
    Args:
        task_config: The task configuration
        resource_config: Optional resource configuration dict
        enable_public_evaluation: Whether public evaluation is enabled
        
    Returns:
        Formatted string containing all task information
    """
    context_parts = []
    
    # Basic task information
    context_parts.append(f"# Task: {task_config.display_name}")
    context_parts.append(f"## Task Description\n{task_config.task_description}")
    
    if task_config.project_goal:
        context_parts.append(f"## Project Goal\n{task_config.project_goal}")

    # Available data
    if task_config.available_data and task_config.available_data.agent_data:
        context_parts.append("\n## Available Data")
        for data_file in task_config.available_data.agent_data:
            # Skip public test data files if public evaluation is disabled
            if not enable_public_evaluation and 'public' in data_file.path.lower():
                continue
            context_parts.append(f"- **{data_file.path}**: {data_file.description}")
    
    # Add data completeness note if present
    if hasattr(task_config, 'data_details') and task_config.data_details:
        context_parts.append(f"\n## Data Details\n{task_config.data_details}")
    
    # Prediction requirements - use workflow templates
    if task_config.evaluation and (task_config.evaluation.workflow_template or task_config.evaluation.custom_workflow):
        context_parts.append("\n## Prediction Requirements")
        
        if task_config.evaluation.workflow_template:
            # Use template-based approach
            templates = load_workflow_templates()
            template_name = task_config.evaluation.workflow_template
            
            if template_name not in templates:
                logger.warning(f"Workflow template '{template_name}' not found. Available templates: {list(templates.keys())}")
                context_parts.append("Workflow template not found. Please specify prediction requirements manually.")
            else:
                template = templates[template_name]
                
                # Determine which workflow to use based on public evaluation setting
                if enable_public_evaluation:
                    # Combine all sections for public evaluation workflow
                    workflow_sections = []
                    if 'training_section' in template:
                        workflow_sections.append(template['training_section'])
                    if 'public_evaluation_section' in template:
                        workflow_sections.append(template['public_evaluation_section'])
                    if 'final_prediction_section' in template:
                        workflow_sections.append(template['final_prediction_section'])
                    workflow_text = "\n\n".join(workflow_sections)
                else:
                    # Use private-only workflow
                    workflow_text = template.get('private_only_workflow', 'No private-only workflow defined for this template.')
                
                # Substitute variables in the workflow text
                substitutions = {
                    'training_files': get_training_files(task_config),
                    'public_test_file': get_public_test_file(task_config),
                    'private_test_file': get_private_test_file(task_config),
                    'target_column': task_config.evaluation.target_column or 'predicted_value'
                }
                
                try:
                    workflow_text = workflow_text.format(**substitutions)
                    context_parts.append(workflow_text)
                except KeyError as e:
                    logger.warning(f"Template substitution failed for variable: {e}")
                    context_parts.append(f"Template processing error. Missing variable: {e}")
        
        elif task_config.evaluation.custom_workflow:
            # Use custom workflow text directly
            if enable_public_evaluation:
                context_parts.append(task_config.evaluation.custom_workflow)
            else:
                # For custom workflows, we need to modify them to remove public evaluation
                # This is a fallback - ideally custom workflows should be template-aware
                custom_text = task_config.evaluation.custom_workflow
                # Simple removal of public evaluation sections (basic approach)
                lines = custom_text.split('\n')
                filtered_lines = []
                skip_section = False
                
                for line in lines:
                    if any(phrase in line.upper() for phrase in ['PUBLIC EVALUATION', 'evaluate_on_public_test']):
                        skip_section = True
                        continue
                    if skip_section and (line.strip() == '' or line.startswith('## ') or line.startswith('# ')):
                        skip_section = False
                    if not skip_section:
                        filtered_lines.append(line)
                
                modified_text = '\n'.join(filtered_lines)
                if modified_text != custom_text:
                    modified_text += "\n\nNote: Public evaluation sections have been removed as public evaluation is disabled."
                context_parts.append(modified_text)
    
    # Evaluation criteria
    if task_config.evaluation:
        context_parts.append("\n## Evaluation")
        if task_config.evaluation.process:
            context_parts.append(f"### Process\n{task_config.evaluation.process}")
        if task_config.evaluation.metrics:
            context_parts.append("### Performance Targets")
            for metric in task_config.evaluation.metrics:
                context_parts.append(f"- **{metric.name}**: {metric.threshold} (on {metric.dataset})")
        if task_config.evaluation.required_outputs:
            context_parts.append("### Required Outputs")
            for output in task_config.evaluation.required_outputs:
                context_parts.append(f"- {output}")
    
    # Model preservation
    if hasattr(task_config, 'model_preservation') and task_config.model_preservation:
        context_parts.append("\n## Model Preservation")
        if hasattr(task_config.model_preservation, 'requirements'):
            context_parts.append(task_config.model_preservation.requirements)
    
    # Report requirements
    if hasattr(task_config, 'report_requirements') and task_config.report_requirements:
        context_parts.append("\n## Report Requirements")
        if hasattr(task_config.report_requirements, 'structure'):
            context_parts.append(task_config.report_requirements.structure)
    
    # Resource configuration if provided
    if resource_config:
        context_parts.append("\n## Available Resources")
        if 'compute' in resource_config:
            compute = resource_config['compute']
            if 'cpu' in compute:
                context_parts.append(f"### CPU")
                context_parts.append(f"- Cores: {compute['cpu'].get('cores', 'Unknown')}")
                context_parts.append(f"- Memory: {compute['cpu'].get('memory_gb', 'Unknown')} GB")
            if 'gpu' in compute and compute['gpu'].get('available'):
                context_parts.append(f"### GPU")
                context_parts.append(f"- Type: {compute['gpu'].get('type', 'Unknown')}")
                context_parts.append(f"- Memory: {compute['gpu'].get('memory_gb', 'Unknown')} GB VRAM")
        if 'timeout' in resource_config:
            context_parts.append(f"### Time Limits")
            context_parts.append(f"- Code execution timeout: {resource_config['timeout'].get('code_execution', 300)} seconds")
    
    return "\n\n".join(context_parts)


def load_workflow_templates() -> Dict[str, Any]:
    """
    Load workflow templates from the templates YAML file.
    
    Returns:
        Dictionary of workflow templates
    """
    templates_path = Path(__file__).parent / "workflow_templates.yaml"
    
    if not templates_path.exists():
        logger.warning(f"Workflow templates file not found: {templates_path}")
        return {}
    
    with open(templates_path, 'r') as f:
        templates = yaml.safe_load(f)
    
    logger.debug(f"Loaded {len(templates)} workflow templates")
    return templates


def get_training_files(task_config: TaskConfig) -> str:
    """Extract training data files from task config."""
    training_files = []
    for data_file in task_config.available_data.agent_data:
        # Files that contain training data (not test sets)
        if not any(keyword in data_file.path.lower() for keyword in ['heldout', 'test', 'validation']):
            training_files.append(data_file.path)
    
    if len(training_files) == 1:
        return training_files[0]
    elif len(training_files) > 1:
        return f"{', '.join(training_files[:-1])} and {training_files[-1]}"
    else:
        return "training data"


def get_public_test_file(task_config: TaskConfig) -> str:
    """Extract public test data file from task config."""
    for data_file in task_config.available_data.agent_data:
        if 'public' in data_file.path.lower() and 'heldout' in data_file.path.lower():
            return data_file.path
    return "public test data"


def get_private_test_file(task_config: TaskConfig) -> str:
    """Extract private test data file from task config."""
    for data_file in task_config.available_data.agent_data:
        if 'private' in data_file.path.lower() and 'heldout' in data_file.path.lower():
            return data_file.path
    return "private test data" 