"""Agent factory for initializing agents based on configurations."""

import os
import datetime
from typing import Dict, List, Optional, Any

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_core.model_context import BufferedChatCompletionContext
from docker.types import DeviceRequest

from .base import TeamAPlanning, EngineerSociety


def remove_public_evaluation_references(prompt: str) -> str:
    """
    Remove references to public evaluation from agent prompts when it's disabled.
    
    Args:
        prompt: Original agent prompt
        
    Returns:
        Modified prompt without public evaluation references
    """
    # Remove the entire public evaluation workflow section
    lines = prompt.split('\n')
    filtered_lines = []
    skip_section = False
    
    for line in lines:
        # Start skipping when we find the evaluation workflow section
        if "**EVALUATION WORKFLOW - REQUIRED STEPS**" in line:
            skip_section = True
            # Add alternative instructions
            filtered_lines.append("      **EVALUATION WORKFLOW**")
            filtered_lines.append("      Generate your final predictions directly for the private test set:")
            filtered_lines.append("      ")
            filtered_lines.append("      1. **TRAIN YOUR MODEL:**")
            filtered_lines.append("         - Use betas.arrow and metadata.arrow to build and train your model")
            filtered_lines.append("         - Apply appropriate preprocessing and feature selection")
            filtered_lines.append("         ")
            filtered_lines.append("      2. **GENERATE FINAL PREDICTIONS:**")
            filtered_lines.append("         - Load betas_heldout_private.arrow")
            filtered_lines.append("         - Apply the same preprocessing as training")
            filtered_lines.append("         - Generate predictions and save as 'predictions.arrow'")
            filtered_lines.append("         ")
            filtered_lines.append("      Note: Public evaluation is not available. Focus on building the best model possible.")
            filtered_lines.append("      ")
            continue
            
        # Stop skipping after the example workflow
        if skip_section and line.strip() == "```" and "final_pred_df.to_feather('predictions.arrow')" in '\n'.join(filtered_lines[-20:]):
            skip_section = False
            filtered_lines.append(line)
            continue
            
        # Skip lines in the evaluation workflow section
        if skip_section:
            continue
            
        # Remove other references to public evaluation
        if "evaluate_on_public_test" in line:
            continue
        if "PUBLIC EVALUATION (REQUIRED FIRST)" in line:
            continue
        if "predictions_public.arrow" in line and "Stage 1:" not in line:
            continue
            
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def initialize_agents(
    agent_configs: Dict[str, Any],
    tools: Dict[str, Any],
    selected_agents: Optional[List[str]] = None,
    model_name: str = "gpt-4.1-mini",
    task_context: Optional[str] = None,
    enable_public_evaluation: bool = True
) -> Dict[str, AssistantAgent]:
    """
    Initialize agents based on configurations.
    
    Args:
        agent_configs: Dictionary of agent configurations
        tools: Dictionary of available tools
        selected_agents: List of agent names to initialize (None = all)
        model_name: Name of the model to use
        task_context: Optional task-specific context to inject
        enable_public_evaluation: Whether public evaluation is enabled
        
    Returns:
        Dictionary of initialized agents
    """
    model_client = OpenAIChatCompletionClient(model=model_name, stream_options={"include_usage": True})
    
    # Get current date once for this initialization
    today_date = datetime.date.today().isoformat()
    
    # Extract the agents dictionary from configs
    agents_dict = agent_configs.get("agents", {})
    
    # Get all agent names and descriptions
    agent_names = []
    agent_descriptions = []
    
    for name, config in agents_dict.items():
        agent_names.append(name)
        # Use role as the description if available
        if "role" in config:
            agent_descriptions.append(config["role"])
        else:
            agent_descriptions.append(config.get("name", name))
    
    # Ensure all selected agents are in the agent_configs
    if selected_agents:
        for agent_name in selected_agents:
            if agent_name not in agent_names:
                raise ValueError(f"Agent {agent_name} not found in agent_configs")
    
    agents = {}
    for name, config in agents_dict.items():
        # Skip agents not in the selected list if a selection is provided
        if selected_agents and name not in selected_agents:
            continue
        
        other_agents_info = "\n".join([
            f"{n}: {d}" for n, d in zip(agent_names, agent_descriptions) 
            if n != name
        ])
        
        # Get tools for this agent - all agents get all tools
        agent_tools = list(tools.values())
        
        # Use the prompt field as the system prompt
        system_prompt = config.get("prompt", "")
        
        # Remove public evaluation references if disabled
        if not enable_public_evaluation and name == "implementation_engineer":
            system_prompt = remove_public_evaluation_references(system_prompt)
        
        # Add date context at the beginning
        system_prompt = f"CONTEXT: Today's date is {today_date}.\n\n" + system_prompt
        
        # Add task context if provided
        if task_context:
            system_prompt = f"{task_context}\n\n{system_prompt}"
        
        if system_prompt:  # Add other context only if base prompt exists
            system_prompt += f"""
            The other agents on your team are: 
            {other_agents_info}

            The agents in the current conversation are:
            {', '.join(selected_agents or agent_names)}
            """
        
        # Add info about the tools available to the agent
        system_prompt += f"""
        \n\n**TOOLS**

        Whenever you encounter a question or task cannot be solved purely through reasoning, 
        or which would benefit from access to other data sources or resources,
        you should use one of the following tools to assist you:

        Here are their descriptions:
        {chr(10).join([f"{tool.name}: {tool.description}" for tool in agent_tools])}

        **YOU ARE HIGHLY ENCOURAGED TO USE TOOLS WHEN APPROPRIATE.**

        **NOTE ABOUT FILE LOCATIONS**
        
        You are executing code in your current working directory. All data files mentioned in the 
        task description are available in this directory. For example, if the task mentions 
        "betas.arrow", the file is available as "betas.arrow" in your current directory.
        Always use relative paths (./filename or just filename) or the current directory.

        **NOTE ABOUT OUTPUT FILES**
        
        All output files (plots, models, results) should be saved to the current directory.
        Use simple filenames like "plot.png", "model.pkl", "results.arrow" etc.
        These files will be automatically available after execution.

        **NOTE ABOUT LAB NOTEBOOK**

        The Lab notebook is a record of all the decisions, observations, and results of the project.
        You should read from it frequently if you cannot recall observations or results from previous steps.
        You should also update with your observations, tips, heuristics, and lessons learned. This will mean
        that, in the future, you will be able to improve your performance by reusing these observations.
        
        When writing to the notebook, you should ALWAYS use the following arguments:
        - entry: <text of the entry to append>
        - entry_type: <PLAN | NOTE | OUTPUT >
        - source: <name of the agent that wrote the entry, i.e. your name>

        **NOTE ABOUT OUTPUTS**
        Whenever you generate a file, result, plot, etc. You MUST make a note of it in the notebook. 
        Specifically, you should use the OUTPUT entry type. 
        Make sure to include the file name and a description of the contents if there is a file.
        For results, make sure to describe the numerical values of results (in markdown table format).

        **NOTE ABOUT FILE FORMATS**
        NEVER WRITE LARGE CSV FILES. ALWAYS USE ARROW FILES INSTEAD.
        GOOD EXAMPLES:
        - data_trainsplit.arrow
        - data_valsplit.arrow
        - data_testsplit.arrow
        - model_evaluation.arrow
        - model_predictions.arrow
        - model_coefficients.arrow
        BAD EXAMPLES:
        - data_trainsplit.csv
        - data_valsplit.csv
        - data_testsplit.csv
        """
        
        # Set appropriate buffer size based on agent role
        if name == "implementation_engineer":
            buffer_size = 30  # Engineers need more context for implementation details
        elif name == "data_science_critic":
            buffer_size = 20  # Critics need sufficient context to review
        else:
            buffer_size = 15  # Planning agents need moderate context
        
        agents[name] = AssistantAgent(
            name=config.get("name", name),
            system_message=system_prompt,
            model_client=model_client,
            model_context=BufferedChatCompletionContext(buffer_size=buffer_size),
            tools=agent_tools,
            model_client_stream=True,
            reflect_on_tool_use=True
        )
    
    return agents


def get_agent_token(agent_configs: Dict[str, Any], agent_name: str, token_type: str = "termination_token") -> Optional[str]:
    """
    Get a token for an agent from the agent configs.
    
    Args:
        agent_configs: Agent configurations dictionary
        agent_name: Name of the agent to find
        token_type: Type of token to retrieve (default: "termination_token")
        
    Returns:
        The token value, or None if the agent or token doesn't exist
    """
    agents_dict = agent_configs.get("agents", {})
    agent_config = agents_dict.get(agent_name, {})
    return agent_config.get(token_type)


async def create_team_a(
    agent_configs: Dict[str, Any],
    tools: Dict[str, Any],
    model_name: str = "gpt-4.1-mini",
    task_context: Optional[str] = None,
    max_turns: int = 15,
    team_composition: Optional[List[str]] = None,
    enable_public_evaluation: bool = True
) -> TeamAPlanning:
    """
    Create Team A (Planning) with the required agents.
    
    Args:
        agent_configs: Agent configurations
        tools: Available tools
        model_name: Model to use
        task_context: Task-specific context
        max_turns: Maximum turns for internal group chat
        team_composition: List of agent names for the team
        enable_public_evaluation: Whether public evaluation is enabled
        
    Returns:
        Initialized TeamAPlanning instance
    """
    # Initialize the agents needed for Team A
    team_a_agents = team_composition or ['principal_scientist', 'bioinformatics_expert', 'ml_expert']
    agents = initialize_agents(
        agent_configs=agent_configs,
        selected_agents=team_a_agents,
        tools=tools,
        model_name=model_name,
        task_context=task_context,
        enable_public_evaluation=enable_public_evaluation
    )
    
    # Get the principal scientist's termination token
    principal_scientist_termination_token = get_agent_token(agent_configs, "principal_scientist")
    
    # Initialize TeamAPlanning
    team_a = TeamAPlanning(
        name="team_a_planning",
        principal_scientist=agents['principal_scientist'],
        ml_expert=agents['ml_expert'],
        bioinformatics_expert=agents['bioinformatics_expert'],
        principal_scientist_termination_token=principal_scientist_termination_token,
        max_turns=max_turns
    )
    
    return team_a


async def create_team_b(
    agent_configs: Dict[str, Any],
    tools: Dict[str, Any],
    model_name: str = "gpt-4.1-mini",
    task_context: Optional[str] = None,
    docker_config: Optional[Dict[str, Any]] = None,
    gpu_spec: str = "all",
    working_dir: str = ".",
    max_messages_to_return: int = 25,
    team_composition: Optional[Dict[str, List[str]]] = None,
    enable_public_evaluation: bool = True
) -> tuple[EngineerSociety, DockerCommandLineCodeExecutor]:
    """
    Create Team B (Engineering) with the required agents and code executor.
    
    Args:
        agent_configs: Agent configurations
        tools: Available tools
        model_name: Model to use
        task_context: Task-specific context
        docker_config: Docker configuration dictionary
        gpu_spec: GPU specification ("all" or comma-separated indices)
        working_dir: Working directory for code execution
        max_messages_to_return: Maximum messages to return from team
        team_composition: Dictionary with implementation_team and review_team lists
        enable_public_evaluation: Whether public evaluation is enabled
        
    Returns:
        Tuple of (EngineerSociety instance, DockerCommandLineCodeExecutor instance)
    """
    # Set default team composition if not provided
    if team_composition is None:
        team_composition = {
            "implementation_team": ["implementation_engineer"],
            "review_team": ["data_science_critic"]
        }
    
    # Set default docker config if not provided
    if docker_config is None:
        docker_config = {
            "image": "millerh1/bioagents:latest",
            "timeout": 3600,
            "work_dir": working_dir
        }
    
    # Initialize engineer
    implementation_agents = team_composition.get("implementation_team", ["implementation_engineer"])
    engineer_agent = initialize_agents(
        agent_configs=agent_configs,
        selected_agents=implementation_agents,
        tools=tools,
        model_name=model_name,
        task_context=task_context,
        enable_public_evaluation=enable_public_evaluation
    )[implementation_agents[0]]  # Use first agent as primary engineer
    
    engineer_termination_token = get_agent_token(agent_configs, "implementation_engineer")
    
    # Set up code executor
    device_requests = []
    if gpu_spec == 'all':
        device_requests = [DeviceRequest(count=-1, capabilities=[["gpu"]])]
    elif gpu_spec and gpu_spec != 'none':
        # Parse comma-separated GPU indices
        gpu_indices = [int(idx.strip()) for idx in gpu_spec.split(',')]
        device_requests = [DeviceRequest(
            device_ids=[str(idx) for idx in gpu_indices],
            capabilities=[["gpu"]]
        )]
    
    code_executor = LimitedOutputDockerExecutor(
        max_output_chars=3000,  # Limit output to 3000 characters
        image=docker_config.get("image", "millerh1/bioagents:latest"),
        work_dir=docker_config.get("work_dir", working_dir),
        timeout=docker_config.get("timeout", 3600),
        device_requests=device_requests
    )
    await code_executor.start()
    
    code_executor_agent = CodeExecutorAgent('code_executor', code_executor=code_executor)
    
    # Create the engineer team as a round-robin group chat
    engineer_team = RoundRobinGroupChat(
        participants=[engineer_agent, code_executor_agent],
        termination_condition=TextMentionTermination(engineer_termination_token),
        max_turns=50  # Reduced from 75 to prevent excessively long conversations
    )

    # Initialize critic team
    review_agents = team_composition.get("review_team", ["data_science_critic"])
    critic_agent = initialize_agents(
        agent_configs=agent_configs,
        selected_agents=review_agents,
        tools=tools,
        model_name=model_name,
        task_context=task_context,
        enable_public_evaluation=enable_public_evaluation
    )[review_agents[0]]  # Use first agent as primary critic
    
    critic_termination_token = get_agent_token(agent_configs, "data_science_critic")
    critic_team = RoundRobinGroupChat(
        participants=[critic_agent],
        termination_condition=TextMentionTermination(critic_termination_token)
    )
    
    # Get tokens for EngineerSociety
    critic_approve_token = get_agent_token(agent_configs, "data_science_critic", "approval_token")
    critic_revise_token = get_agent_token(agent_configs, "data_science_critic", "revision_token")
    
    # Initialize EngineerSociety
    team_b = EngineerSociety(
        name="team_b_engineering",
        engineer_team=engineer_team,
        critic_team=critic_team,
        critic_approve_token=critic_approve_token,
        engineer_terminate_token=engineer_termination_token,
        critic_terminate_token=critic_termination_token,
        critic_revise_token=critic_revise_token,
        max_messages_to_return=max_messages_to_return
    )
    
    # Set the output directory to the current working directory
    team_b._output_dir = "."
    
    return team_b, code_executor


class LimitedOutputDockerExecutor(DockerCommandLineCodeExecutor):
    """Docker code executor that limits output to prevent token explosion."""
    
    def __init__(self, max_output_chars: int = 3000, **kwargs):
        super().__init__(**kwargs)
        self.max_output_chars = max_output_chars
    
    async def _execute_code_dont_check_setup(self, code_blocks, cancellation_token):
        """Execute code blocks with output limiting."""
        result = await super()._execute_code_dont_check_setup(code_blocks, cancellation_token)
        
        # Limit the output length
        if len(result.output) > self.max_output_chars:
            truncated_output = result.output[-self.max_output_chars:]
            final_output = f"[OUTPUT TRUNCATED - showing last {self.max_output_chars} characters]\n\n{truncated_output}"
            
            # Create a new result with truncated output
            from autogen_ext.code_executors._common import CommandLineCodeResult
            return CommandLineCodeResult(
                exit_code=result.exit_code,
                output=final_output,
                code_file=result.code_file
            )
        
        return result 