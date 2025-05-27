# BioAgents Framework Architecture

BioAgents is a framework for deploying autonomous agent teams to solve biomedical machine learning tasks. It uses the AutoGen framework to orchestrate multi-agent collaboration between planning and implementation teams.

## 🏗️ Core Architecture

### Agent Teams

**Team A (Planning)**
- **Principal Scientist**: Project coordinator and strategic decision maker
- **Bioinformatics Expert**: Domain expertise in biological data processing and analysis
- **ML Expert**: Machine learning architecture, methodology, and best practices guidance

**Team B (Implementation)**
- **Implementation Engineer**: Executes plans from Team A with code development
- **Code Executor**: Docker-based secure code execution environment
- **Data Science Critic**: Reviews implementations and provides quality assurance

### Workflow Orchestration

1. **Planning Phase**: Team A analyzes task context and creates detailed implementation specifications
2. **Implementation Phase**: Team B develops and executes the solution
3. **Review Phase**: Critical evaluation with iterative improvement cycles
4. **Completion**: Verified solution with model checkpoint and evaluation metrics

## 📁 Module Structure

```
src/bioagents/
├── agents/              # Agent implementations
│   ├── base.py          # Core agent classes (TeamAPlanning, EngineerSociety)
│   ├── factory.py       # Agent initialization and team creation
│   └── __init__.py
├── config/              # Configuration management
│   ├── schemas.py       # Pydantic schemas for validation
│   ├── loader.py        # Configuration loading utilities
│   ├── agents.yaml      # Default agent configurations
│   └── __init__.py
├── tools/               # Agent tools and utilities
│   ├── base.py          # Basic tools (calculator, file operations)
│   ├── notebook.py      # Lab notebook management
│   ├── research.py      # Research tools (Perplexity, web parsing)
│   └── __init__.py
├── pipeline/            # Execution pipeline
│   ├── runner.py        # Main orchestration logic
│   └── __init__.py
├── cli.py               # Hydra-based command-line interface
├── __init__.py          # Package initialization with .env loading
└── __main__.py          # Module execution entry point
```

## ⚙️ Configuration System

### Hydra Integration

The framework uses [Hydra](https://hydra.cc/) for sophisticated configuration management:

```yaml
# conf/config.yaml
defaults:
  - _self_
  - agents: default
  - docker: default
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

task_dir: ???  # Required parameter
model: gpt-4
max_iterations: 25
dry_run: false
```

### Configuration Groups

**Agent Groups** (`conf/agents/`)
- `default.yaml`: Full team composition with standard settings
- `minimal.yaml`: Reduced team for faster testing

**Docker Groups** (`conf/docker/`)
- `default.yaml`: GPU-enabled with 1-hour timeout
- `cpu_only.yaml`: CPU-only execution
- `extended_timeout.yaml`: 2-hour timeout for complex tasks

**Experiment Groups** (`conf/experiment/`)
- `quick_test.yaml`: Fast testing preset (minimal agents + CPU-only)
- `production.yaml`: Full production preset (default agents + extended timeout)

### Configuration Composition

```bash
# Use configuration groups
bioagents task_dir=challenges/01_basic_epigenetic_clock agents=minimal docker=cpu_only

# Use experiment presets
bioagents task_dir=challenges/01_basic_epigenetic_clock +experiment=quick_test

# Override specific parameters
bioagents task_dir=challenges/01_basic_epigenetic_clock model=gpt-4o max_iterations=10
```

## 🔧 Agent Factory System

### Team Creation

```python
# Team A (Planning)
team_a = await create_team_a(
    agent_configs=agent_configs,
    tools=tools,
    model_name=model_name,
    task_context=task_context,
    team_composition=["principal_scientist", "bioinformatics_expert", "ml_expert"]
)

# Team B (Implementation)
team_b, docker_executor = await create_team_b(
    agent_configs=agent_configs,
    tools=tools,
    model_name=model_name,
    task_context=task_context,
    docker_config=docker_config,
    gpu_spec=gpu_spec,
    working_dir=str(run_dir),
    team_composition={
        "implementation_team": ["implementation_engineer"],
        "review_team": ["data_science_critic"]
    }
)
```

### Agent Initialization

Each agent receives:
- **Task Context**: Specific challenge information and requirements
- **Tool Access**: Complete toolkit for file operations, calculations, research
- **Team Awareness**: Information about other agents and their roles
- **Notebook Access**: Read/write access to persistent lab notebook

## 🛠️ Tool System

### Available Tools

**File Operations**
- `read_text_file`: Read various text formats (txt, csv, json, etc.)
- `write_text_file`: Write text files with proper encoding
- `read_arrow_file`: Read Arrow/Parquet files as DataFrames
- `search_directory`: Find files with pattern matching

**Analysis Tools**
- `calculator`: Mathematical expressions and statistical calculations
- `analyze_plot`: Vision-based plot analysis and description
- `perplexity_search`: AI-powered research queries
- `webpage_parser`: Extract content from web pages

**Notebook Management**
- `read_notebook`: Access lab notebook history
- `write_notebook`: Document decisions, results, and observations

### Tool Integration

```python
# Tools are automatically injected into agent system prompts
system_prompt += f"""
**TOOLS**

You have access to the following tools:
{chr(10).join([f"{tool.name}: {tool.description}" for tool in agent_tools])}

**YOU ARE HIGHLY ENCOURAGED TO USE TOOLS WHEN APPROPRIATE.**
"""
```

## 📊 Pipeline Execution

### Main Loop

```python
while iteration < max_iterations and not project_complete:
    # Team A Planning Phase
    team_a_prompt = format_prompt(notebook_content, last_team_b_output, task_config)
    team_a_response = await team_a.on_messages([team_a_message], CancellationToken())
    
    # Check for completion
    if completion_token in team_a_response.chat_message.content:
        project_complete = True
        break
    
    # Team B Implementation Phase
    team_b_response = await team_b.on_messages([team_a_response.chat_message], CancellationToken())
    last_team_b_output = team_b_response.chat_message.content
    
    # Document iteration
    write_notebook(entry=f"Completed iteration {iteration}", entry_type="OUTPUT", source="TEAM_B")
```

### Output Management

**Hydra Directory Structure**
```
outputs/
├── runs/YYYY-MM-DD/HH-MM-SS/        # Standard runs
├── tests/YYYY-MM-DD/HH-MM-SS/       # Quick test runs
└── production/YYYY-MM-DD/HH-MM-SS/  # Production runs
```

**Run Directory Contents**
- `config.yaml`: Complete configuration used for the run
- `lab_notebook.md`: Agent collaboration and decision log
- `.hydra/`: Hydra metadata for full reproducibility
- Generated files: Models, plots, data files, evaluation results

## 🔍 Task Configuration Schema

### Task Definition

```yaml
# challenges/task_name/task.yaml
name: "task_name"
version: "1.0"
display_name: "Human Readable Task Name"
description: "Detailed task description"

available_data:
  agent_data:
    - path: "data/agent/training_data.arrow"
      description: "Training dataset"
      format: "arrow"
  eval_data:
    - path: "data/eval/test_data.arrow"
      description: "Held-out test data"
      format: "arrow"

evaluation:
  metrics:
    - name: "pearson_correlation"
      threshold: 0.9
      dataset: "test_set"
    - name: "mae"
      threshold: 10.0
      dataset: "test_set"

requirements:
  success_criteria:
    - "Model checkpoint saved as 'model.pkl'"
    - "Prediction script 'predict.py' created"
    - "Performance targets achieved"

docker:
  image: "millerh1/bioagents:latest"
  timeout: 3600
```

## 🧪 Development and Testing

### Creating New Challenges

```bash
# Generate challenge template
python scripts/create_challenge.py my_challenge "My Challenge Name"

# Validate challenge structure
python scripts/validate_challenges.py --challenge my_challenge

# Test challenge configuration
bioagents task_dir=challenges/my_challenge dry_run=true
```

### Testing Framework

```bash
# Run comprehensive tests
python test_updated_refactoring.py

# Test specific configuration
bioagents task_dir=challenges/01_basic_epigenetic_clock +experiment=quick_test

# Validate all challenges
python scripts/validate_challenges.py
```

## 🔧 Extensibility

### Custom Agents

```python
class CustomAgent(BaseChatAgent):
    def __init__(self, name: str, specialized_tools: List[Tool]):
        super().__init__(name, description="Custom agent description")
        self.tools = specialized_tools
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        # Custom agent logic
        pass
```

### Custom Tools

```python
async def custom_tool(parameter: str) -> str:
    """Custom tool implementation."""
    # Tool logic here
    return result

# Register tool
custom_tool_instance = FunctionTool(
    custom_tool,
    name="custom_tool",
    description="Description of what the tool does"
)
```

### Custom Configuration Groups

```yaml
# conf/agents/custom.yaml
# @package _global_
agents_config: null

team_composition:
  planning_team:
    - custom_planning_agent
  implementation_team:
    - custom_implementation_agent
  review_team:
    - custom_review_agent

planning_max_turns: 10
implementation_max_turns: 50
review_max_iterations: 2
```

## 🐛 Troubleshooting

### Common Issues

**Configuration Problems**
```bash
# Validate configuration
bioagents --cfg job task_dir=challenges/01_basic_epigenetic_clock

# Check Hydra configuration
bioagents --cfg hydra task_dir=challenges/01_basic_epigenetic_clock
```

**Docker Issues**
```bash
# Use CPU-only mode
bioagents task_dir=challenges/01_basic_epigenetic_clock docker=cpu_only

# Increase timeout
bioagents task_dir=challenges/01_basic_epigenetic_clock docker=extended_timeout
```

**Path Resolution**
- Hydra changes working directory; framework handles this automatically
- Task directories are resolved relative to original working directory
- All outputs go to Hydra-managed output directories

### Debugging

**Dry Run Mode**
```bash
bioagents task_dir=challenges/01_basic_epigenetic_clock dry_run=true
```

**Verbose Logging**
```bash
bioagents task_dir=challenges/01_basic_epigenetic_clock hydra.verbose=true
```

**Configuration Inspection**
```bash
# View effective configuration
bioagents --cfg job task_dir=challenges/01_basic_epigenetic_clock

# View available options
bioagents --help
```

## 📚 References

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [AutoGen Framework](https://github.com/microsoft/autogen)
- [Pydantic Validation](https://pydantic.dev/)
- [OmegaConf Configuration](https://omegaconf.readthedocs.io/)