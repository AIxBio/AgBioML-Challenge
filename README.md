# Agentic BioML Challenge + AutoBioML Framework

## 📝 Challenge Description

The AgBioML Challenge is a competition to develop AI agents that can solve complex biomedical machine learning challenges through planning, \implementation, and critical review cycles.

| Challenge | Status | Description |
|-----------|--------|-------------|
| [01_basic_epigenetic_clock](challenges/01_basic_epigenetic_clock/) | ✅ Ready | Age prediction from DNA methylation data |
| [02_rsna_breast_cancer_detection](challenges/02_rsna_breast_cancer_detection/) | 🚧 Template | Breast cancer detection from mammograms |
| [03_tdc_admet_benchmark](challenges/03_tdc_admet_benchmark/) | 🚧 Template | ADMET property prediction from SMILES |
| [04_cellmap_challenge](challenges/04_cellmap_challenge/) | 🚧 Template | EM tissue image segmentation |
| [05_dream_target_2035](challenges/05_dream_target_2035/) | 🚧 Template | Small molecule activity prediction |
| [06_biomarkers_of_aging](challenges/06_biomarkers_of_aging/) | 🚧 Template | Disease prediction from DNA methylation data |

See full challenge description [here](https://livingmachines.substack.com/p/the-agentic-bioml-challenge).

## 🤖 AutoBioML Framework

An autonomous agent framework for biomedical machine learning. autobioml enables teams of AI agents to collaboratively solve complex biomedical ML challenges through planning, implementation, and critical review cycles.

### 🚀 Quick Start

#### Installation

1. **Install UV package manager** (recommended):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone and setup**:
```bash
git clone https://github.com/AIxBio/AgBioML-Challenge.git
cd AgBioML-Challenge
uv sync
```

3. **Set up API keys** (create `.env` file):
```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key  # Optional
PERPLEXITY_API_KEY=your_perplexity_key  # Optional
```

#### Basic Usage

```bash
# Run the challenge
autobioml task_dir=challenges/01_basic_epigenetic_clock

# Run the challenge with a different model
autobioml task_dir=challenges/01_basic_epigenetic_clock model=gpt-4o

# Custom configuration with fewer iterations
autobioml task_dir=challenges/01_basic_epigenetic_clock \
    model=gpt-4o \
    max_iterations=15
```

### 🏗️ Framework Architecture

#### Agent Teams

**Team A (Planning)**
- **Principal Scientist**: Strategic oversight and decision-making
- **Bioinformatics Expert**: Domain expertise in computational biology
- **ML Expert**: Machine learning methodology and best practices

**Team B (Implementation)**
- **Implementation Engineer**: Code development and execution
- **Data Science Critic**: Quality assurance and validation

#### Workflow

1. **Planning Phase**: Team A analyzes the task and creates implementation specifications
2. **Implementation Phase**: Team B develops and executes the solution
3. **Review Phase**: Critical evaluation and iterative improvement
4. **Completion**: Verified solution with model checkpoint and evaluation

### 🎯 Public Evaluation System

autobioml includes a public/private evaluation system that mirrors real ML competition workflows:

**How It Works:**
1. **Data Split**: Test data is split 50/50 into public and private subsets
2. **Public Testing**: Agents MUST evaluate on the public subset (up to X times)
3. **Final Scoring**: Final evaluation uses the private subset

**For Agents:**
- Generate predictions from public test set (e.g. `betas_heldout_public.arrow` → `predictions_public.arrow`)
- Use the `evaluate_on_public_test` tool to get performance metrics
- Iterate and improve based on public results (X attempts max)
- Generate final predictions on private test set (e.g. `betas_heldout_private.arrow` → `predictions.arrow`) 
- After concluding the agent loop, evaluation script will be run on predictions from private test set to generate final performance metrics

**Default Behavior:**
- Public evaluation is **enabled by default** 
- To disable (not recommended): `enable_public_evaluation=false`

### 🎛️ Configuration System

autobioml uses [Hydra](https://hydra.cc/) for configuration management.

```bash
# View all available options
autobioml --help

# View current configuration
autobioml --cfg job task_dir=challenges/01_basic_epigenetic_clock

# View Hydra configuration
autobioml --cfg hydra task_dir=challenges/01_basic_epigenetic_clock
```

### Command Line Examples

**Development & Testing**
```bash
# Dry run validation
autobioml task_dir=challenges/01_basic_epigenetic_clock dry_run=true
```



### 📁 Output Management

autobioml automatically organizes outputs using Hydra's directory management:

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


### 🛠️ Development

#### Repository Structure

```
├── src/autobioml/           # Core framework
│   ├── agents/              # Agent implementations
│   ├── config/              # Configuration schemas
│   ├── pipeline/            # Execution pipeline
│   ├── tools/               # Agent tools and utilities
│   └── cli.py               # Command-line interface
├── conf/                    # Hydra configuration
│   ├── agents/              # Agent configuration groups
│   ├── docker/              # Docker configuration groups
│   ├── experiment/          # Experiment presets
│   ├── resources/           # Configuration of compute resources
│   └── config.yaml          # Main configuration
├── challenges/              # Challenge implementations
└── experiments/             # Research experiments
```

### Multi-run Experiments

autobioml supports parallel execution of multirun experiments using Hydra's joblib launcher:

```bash
# Parallel parameter sweeps (runs 4 jobs in parallel by default)
autobioml --multirun +experiment=multirun_comparison task_dir=challenges/01_basic_epigenetic_clock model=gpt-4.1,gpt-4o enable_public_evaluation=true,false +seed=0,1,2,3 hydra.launcher.n_jobs=4
```
