"""Configuration schemas for biomlagents using Pydantic."""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from pathlib import Path


class DataFile(BaseModel):
    """Schema for a data file specification."""
    path: str
    description: str


class AvailableData(BaseModel):
    """Schema for available data specification."""
    agent_data: List[DataFile]
    eval_data: List[DataFile]


class EvaluationMetric(BaseModel):
    """Schema for an evaluation metric."""
    name: str
    threshold: float
    dataset: str


class EvaluationConfig(BaseModel):
    """Schema for evaluation configuration."""
    process: Optional[str] = None
    metrics: List[EvaluationMetric]
    required_outputs: List[str]
    
    # New workflow template system
    workflow_template: Optional[str] = None  # e.g., "standard_ml_prediction", "time_series_prediction"
    target_column: Optional[str] = None      # e.g., "predicted_age", "score", "class"
    custom_workflow: Optional[str] = None    # For tasks that need completely custom workflows


class DockerConfig(BaseModel):
    """Schema for Docker configuration."""
    gpu_required: bool = False
    base_image: str = "millerh1/biomlagents:latest"
    additional_packages: Optional[List[str]] = None


class AutonomousWorkflow(BaseModel):
    """Schema for autonomous workflow configuration."""
    approach: str
    methodology: str
    expected_outcomes: List[str]


class ReferenceSection(BaseModel):
    """Schema for a reference section."""
    text: str


class TaskConfig(BaseModel):
    """Schema for complete task configuration."""
    name: str
    display_name: str
    version: str
    task_description: str
    project_goal: str
    available_data: AvailableData
    data_details: str
    evaluation: EvaluationConfig
    docker: DockerConfig
    
    @validator('name')
    def validate_name(cls, v):
        """Validate task name contains only alphanumeric and underscore."""
        if not v.replace('_', '').isalnum():
            raise ValueError('Task name must contain only alphanumeric characters and underscores')
        return v
    
    @validator('version')
    def validate_version(cls, v):
        """Validate version format."""
        parts = v.split('.')
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError('Version must be in format X.Y where X and Y are integers')
        return v


class AgentConfig(BaseModel):
    """Schema for an individual agent configuration."""
    name: str
    role: str
    prompt: str
    termination_token: Optional[str] = None
    approval_token: Optional[str] = None
    revision_token: Optional[str] = None


class AgentsConfig(BaseModel):
    """Schema for all agents configuration."""
    agents: Dict[str, AgentConfig]


class HydraConfig(BaseModel):
    """Schema for Hydra configuration."""
    task_dir: str = Field(..., description="Path to challenge directory")
    max_iterations: int = Field(25, description="Maximum number of Team A-B exchanges")
    docker_timeout: int = Field(3600, description="Docker execution timeout in seconds")
    gpus: str = Field("all", description="GPU specification: 'all', 'none', or comma-separated indices")
    working_dir: str = Field("./workdir", description="Base directory for run outputs")
    dry_run: bool = Field(False, description="Validate configuration without running")
    model: str = Field("gpt-4.1-mini", description="LLM model to use")
    agents_config: Optional[str] = Field(None, description="Path to custom agents configuration") 