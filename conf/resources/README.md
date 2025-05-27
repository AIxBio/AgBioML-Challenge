# Resource Configuration

The bioagents framework supports different resource profiles to run efficiently on various hardware configurations.

## Available Resource Profiles

1. **default** - Conservative settings for basic machines
   - 4 CPU cores, 16GB RAM
   - No GPU
   - Small batch sizes (32-1024)
   - 5 minute code execution timeout

2. **high_performance** - For powerful workstations/servers
   - 30 CPU cores, 220GB RAM
   - A10 GPU with 24GB VRAM
   - Large batch sizes (256-4096)
   - 5 minute code execution timeout

3. **cloud_gpu** - For typical cloud GPU instances
   - 8 CPU cores, 32GB RAM
   - T4 GPU with 16GB VRAM
   - Medium batch sizes (128-2048)
   - 5 minute code execution timeout

4. **long_running** - For experiments requiring extended execution times
   - 16 CPU cores, 64GB RAM
   - V100 GPU with 32GB VRAM
   - Medium batch sizes (128-2048)
   - 15 minute code execution timeout
   - 4 hour docker container timeout

## Using Resource Profiles

To use a specific resource profile, override it via command line:

```bash
# Use default resources
bioagents --task challenges/01_basic_epigenetic_clock

# Use high-performance resources
bioagents --task challenges/01_basic_epigenetic_clock resources=high_performance

# Use cloud GPU resources
bioagents --task challenges/01_basic_epigenetic_clock resources=cloud_gpu
```

## Creating Custom Resource Profiles

Create a new YAML file in this directory with your custom settings:

```yaml
# conf/resources/my_custom.yaml
compute:
  cpu:
    cores: 16
    memory_gb: 64
  gpu:
    available: true
    type: "V100"
    memory_gb: 32
    count: 2
  
optimization_hints:
  prefer_gpu: true
  min_batch_size: 512
  max_batch_size: 8192
  parallel_workers: 8
  
timeout:
  code_execution: 600
  docker_execution: 7200
```

Then use it:

```bash
bioagents --task challenges/01_basic_epigenetic_clock resources=my_custom
```

## Resource Variables in Agent Prompts

The agent prompts automatically adapt based on the resource configuration:
- CPU cores and memory are shown to the implementation engineer
- GPU availability and type are communicated
- Batch size recommendations are adjusted
- Parallel worker counts are configured
- Timeout values are injected into runtime estimation guidelines

This ensures agents make optimal use of available hardware without hardcoding assumptions.

## Timeout Configuration

The timeout settings are particularly important for controlling execution behavior:

### Code Execution Timeout
- Controls how long individual code blocks can run
- Agent is informed of this limit and plans accordingly
- Runtime estimation guidelines use 60% of this value as a safety threshold

### Docker Execution Timeout  
- Controls the overall Docker container lifetime
- Should be set higher than code execution timeout
- Allows for multiple code executions within a single container session

### Examples

For quick experiments (default profile):
- Code timeout: 300 seconds (5 minutes)
- Safe threshold: 180 seconds (3 minutes)
- Docker timeout: 3600 seconds (1 hour)

For long-running ML training (long_running profile):
- Code timeout: 900 seconds (15 minutes)
- Safe threshold: 540 seconds (9 minutes)
- Docker timeout: 14400 seconds (4 hours)

The agent will automatically adjust its approach based on these limits, breaking down long computations into smaller chunks when necessary. 