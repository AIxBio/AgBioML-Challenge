# BioAgents Framework

An autonomous agent framework for biomedical machine learning research. BioAgents enables teams of AI agents to collaboratively solve complex biomedical ML challenges through planning, implementation, and critical review cycles.

## 🚀 Quick Start

### Installation

1. **Install UV package manager** (recommended):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone and setup**:
```bash
git clone <repository-url>
cd flock-ag
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

3. **Set up API keys** (create `.env` file):
```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key  # Optional
PERPLEXITY_API_KEY=your_perplexity_key  # Optional
```

### Basic Usage

```bash
# Quick test run
bioagents task_dir=challenges/01_basic_epigenetic_clock +experiment=quick_test

# Production run
bioagents task_dir=challenges/01_basic_epigenetic_clock +experiment=production

# Custom configuration
bioagents task_dir=challenges/01_basic_epigenetic_clock \
    model=gpt-4o \
    max_iterations=15 \
    agents=minimal
```

## 🏗️ Framework Architecture

### Agent Teams

**Team A (Planning)**
- **Principal Scientist**: Strategic oversight and decision-making
- **Bioinformatics Expert**: Domain expertise in computational biology
- **ML Expert**: Machine learning methodology and best practices

**Team B (Implementation)**
- **Implementation Engineer**: Code development and execution
- **Data Science Critic**: Quality assurance and validation

### Workflow

1. **Planning Phase**: Team A analyzes the task and creates implementation specifications
2. **Implementation Phase**: Team B develops and executes the solution
3. **Review Phase**: Critical evaluation and iterative improvement
4. **Completion**: Verified solution with model checkpoint and evaluation

## 🎯 Public Evaluation System

BioAgents includes a **mandatory** public/private evaluation system that mirrors real ML competition workflows:

**How It Works:**
1. **Data Split**: Test data is split 50/50 into public and private subsets
2. **Public Testing**: Agents MUST evaluate on the public subset (up to 5 times)
3. **Final Scoring**: Final evaluation uses the private subset

**For Agents:**
- First evaluate models on `betas_heldout_public.arrow` → `predictions_public.arrow`
- Use the `evaluate_on_public_test` tool to get performance metrics (REQUIRED)
- Iterate and improve based on public results (5 attempts max)
- Generate final predictions on `betas_heldout_private.arrow` → `predictions.arrow`

**Default Behavior:**
- Public evaluation is **enabled by default** and required
- To disable (not recommended): `enable_public_evaluation=false`

This system prevents overfitting to the test set while ensuring agents validate their approach before final submission.

## 🎛️ Configuration System

BioAgents uses [Hydra](https://hydra.cc/) for flexible, composable configuration management.

### Configuration Groups

**Agent Configurations**
```bash
agents=default    # Full team (3 planning + 1 implementation + 1 review)
agents=minimal    # Reduced team (1 planning + 1 implementation + 1 review)
```

**Docker Configurations**
```bash
docker=default           # GPU-enabled, 1-hour timeout
docker=cpu_only          # CPU-only execution
docker=extended_timeout  # 2-hour timeout for complex tasks
```

**Experiment Presets**
```bash
+experiment=quick_test   # Fast testing (gpt-4o-mini, 3 iterations, CPU-only)
+experiment=production   # Full production (gpt-4, 25 iterations, extended timeout)
```

### Command Line Examples

**Development & Testing**
```bash
# Quick development test
bioagents task_dir=challenges/01_basic_epigenetic_clock +experiment=quick_test

# CPU-only testing
bioagents task_dir=challenges/01_basic_epigenetic_clock agents=minimal docker=cpu_only

# Dry run validation
bioagents task_dir=challenges/01_basic_epigenetic_clock dry_run=true
```

**Production Runs**
```bash
# Full production run
bioagents task_dir=challenges/01_basic_epigenetic_clock +experiment=production

# Custom production setup
bioagents task_dir=challenges/01_basic_epigenetic_clock \
    agents=default \
    docker=extended_timeout \
    model=gpt-4 \
    max_iterations=25
```

**Parameter Overrides**
```bash
# Custom model and iterations
bioagents task_dir=challenges/01_basic_epigenetic_clock model=gpt-4o max_iterations=10

# Mixed configuration
bioagents task_dir=challenges/01_basic_epigenetic_clock \
    agents=minimal \
    model=gpt-4o-mini \
    docker=cpu_only
```

### Configuration Discovery

```bash
# View all available options
bioagents --help

# View current configuration
bioagents --cfg job task_dir=challenges/01_basic_epigenetic_clock

# View Hydra configuration
bioagents --cfg hydra task_dir=challenges/01_basic_epigenetic_clock
```

## 📁 Output Management

BioAgents automatically organizes outputs using Hydra's directory management:

```
outputs/
├── runs/YYYY-MM-DD/HH-MM-SS/        # Standard runs
├── tests/YYYY-MM-DD/HH-MM-SS/       # Quick test runs  
└── production/YYYY-MM-DD/HH-MM-SS/  # Production runs
```

Each run directory contains:
- `config.yaml` - Complete configuration used
- `lab_notebook.md` - Agent collaboration log
- `.hydra/` - Hydra metadata for reproducibility
- Generated outputs (models, plots, data files)

## 🧪 Available Challenges

| Challenge | Status | Description |
|-----------|--------|-------------|
| [01_basic_epigenetic_clock](challenges/01_basic_epigenetic_clock/) | ✅ Ready | Age prediction from DNA methylation data |
| [02_rsna_breast_cancer_detection](challenges/02_rsna_breast_cancer_detection/) | 🚧 Template | Breast cancer detection from mammograms |
| [03_tdc_admet_benchmark](challenges/03_tdc_admet_benchmark/) | 🚧 Template | ADMET property prediction from SMILES |
| [04_cellmap_challenge](challenges/04_cellmap_challenge/) | 🚧 Template | EM tissue image segmentation |
| [05_dream_target_2035](challenges/05_dream_target_2035/) | 🚧 Template | Small molecule activity prediction |
| [06_biomarkers_of_aging](challenges/06_biomarkers_of_aging/) | 🚧 Template | Health outcome prediction |

## 🛠️ Development

### Repository Structure

```
├── src/bioagents/           # Core framework
│   ├── agents/              # Agent implementations
│   ├── config/              # Configuration schemas
│   ├── pipeline/            # Execution pipeline
│   ├── tools/               # Agent tools and utilities
│   └── cli.py               # Command-line interface
├── conf/                    # Hydra configuration
│   ├── agents/              # Agent configuration groups
│   ├── docker/              # Docker configuration groups
│   ├── experiment/          # Experiment presets
│   └── config.yaml          # Main configuration
├── challenges/              # Challenge implementations
└── experiments/             # Research experiments
```

### Creating New Challenges

```bash
# Create challenge template
python scripts/create_challenge.py my_challenge "My Challenge Name"

# Validate challenge structure
python scripts/validate_challenges.py --challenge my_challenge

# Test challenge
bioagents task_dir=challenges/my_challenge dry_run=true
```

### Running Tests

```bash
# Test framework functionality
python test_updated_refactoring.py

# Validate all challenges
python scripts/validate_challenges.py
```

## 🔧 Advanced Usage

### Multi-run Experiments

```bash
# Parameter sweeps
bioagents --multirun task_dir=challenges/01_basic_epigenetic_clock +experiment=multirun_comparison model=gpt-4.1,gpt-4.1-mini,gpt-4o,gpt-4o-mini enable_public_evaluation=true,false +seed=0,1,2,3,4,5
```

### Custom Docker Images

```bash
# Use custom Docker image
bioagents task_dir=challenges/01_basic_epigenetic_clock docker.image=my-custom-image:latest
```

### Environment Variables

The framework automatically loads environment variables from `.env`:
- `OPENAI_API_KEY` - Required for OpenAI models
- `ANTHROPIC_API_KEY` - Optional for Claude models
- `PERPLEXITY_API_KEY` - Optional for research tools

## 📚 Documentation

- [Challenge Creation Guide](challenges/README.md)
- [Framework Architecture](src/README.md)
- [Configuration Reference](UPDATED_REFACTORING_RESULTS.md)
- [Hydra Documentation](https://hydra.cc/docs/intro/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
uv add --dev pytest black isort mypy

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/
```

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

Built with:
- [Hydra](https://hydra.cc/) for configuration management
- [AutoGen](https://github.com/microsoft/autogen) for agent orchestration
- [Pydantic](https://pydantic.dev/) for data validation
- [UV](https://github.com/astral-sh/uv) for package management







