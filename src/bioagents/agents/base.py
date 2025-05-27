"""Base agent implementations for the bioagents framework."""

import time
from typing import Sequence, List, Dict, Any, Optional

from autogen_agentchat.agents import BaseChatAgent, AssistantAgent
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.base import Response
from autogen_core import CancellationToken
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console


def estimate_tokens(messages):
    """Estimate the number of tokens in a list of messages."""
    total_words = sum(len(message.content.split()) for message in messages)
    total_tokens = total_words * 3
    return total_tokens


class TeamAPlanning(BaseChatAgent):
    """
    A custom agent that manages collaboration between the principal scientist,
    machine learning expert, and bioinformatics expert for planning and analysis.
    """
    def __init__(
        self, 
        name: str, 
        principal_scientist: AssistantAgent, 
        ml_expert: AssistantAgent, 
        bioinformatics_expert: AssistantAgent,
        principal_scientist_termination_token: str,
        max_turns: int = 15
    ) -> None:
        """
        Initialize the TeamAPlanning agent.
        
        Args:
            name: The name of the agent.
            principal_scientist: The principal scientist agent.
            ml_expert: The machine learning expert agent.
            bioinformatics_expert: The bioinformatics expert agent.
            principal_scientist_termination_token: Token that the principal scientist uses to terminate discussion.
            max_turns: Maximum number of turns in the internal group chat.
        """
        super().__init__(name, description="Team A that handles planning, analysis, and decisions")
        
        # Store the agents
        self._principal_scientist = principal_scientist
        self._ml_expert = ml_expert
        self._bioinformatics_expert = bioinformatics_expert
        
        # Create the internal group chat
        from autogen_agentchat.conditions import TextMentionTermination
        
        self._termination_token = principal_scientist_termination_token
        self._termination_condition = TextMentionTermination(principal_scientist_termination_token)
        
        self._group_chat = RoundRobinGroupChat(
            participants=[principal_scientist, bioinformatics_expert, ml_expert],
            termination_condition=self._termination_condition,
            max_turns=max_turns
        )
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """
        Process messages through an internal group chat of planning experts.
        
        Args:
            messages: Incoming messages.
            cancellation_token: Token to cancel the operation.
            
        Returns:
            Response object containing the result of the group chat.
        """
        # Log received messages (for debugging)
        print(f"TeamAPlanning - Received {len(messages)} messages")
        print(f"TOKEN ESTIMATE: {estimate_tokens(messages)}")
        
        # Run the internal group chat
        result = await Console(
            self._group_chat.run_stream(task=messages, cancellation_token=cancellation_token), 
            output_stats=True
        )
        
        # Extract the final message (summary/plan from the Principal Scientist)
        # Remove the termination token
        final_message = result.messages[-1]
        if isinstance(final_message, TextMessage) and self._termination_token in final_message.content:
            final_message.content = final_message.content.replace(self._termination_token, "").strip()
        
        # Return only the final message as the response
        return Response(chat_message=final_message)
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the agent, clearing any internal state."""
        await self._group_chat.reset()
    
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        """Return the message types this agent can produce."""
        return (TextMessage,)


class EngineerSociety(BaseChatAgent):
    """
    A custom agent that manages the interaction between an engineer team and a critic team.
    
    Features:
    - Returns the last N messages instead of a summary
    - Records metrics and results to the notebook
    """
    def __init__(
        self, 
        name: str, 
        engineer_team: RoundRobinGroupChat, 
        critic_team: RoundRobinGroupChat, 
        critic_approve_token: str, 
        engineer_terminate_token: str, 
        critic_terminate_token: str, 
        critic_revise_token: str, 
        max_messages_to_return: int = 25
    ) -> None:
        """
        Initialize the EngineerSociety agent.
        
        Args:
            name: The name of the agent.
            engineer_team: The engineer team round-robin group chat.
            critic_team: The critic team round-robin group chat.
            critic_approve_token: Token used by the critic to approve the engineer's work.
            engineer_terminate_token: Token used by the engineer to terminate their work.
            critic_terminate_token: Token used by the critic to terminate their review.
            critic_revise_token: Token used by the critic to request revisions.
            max_messages_to_return: Maximum number of messages to return in the response.
        """
        super().__init__(name, description="Team B that handles implementation with critical feedback.")
        self._engineer_team = engineer_team
        self._critic_team = critic_team
        self._engineer_terminate_token = engineer_terminate_token
        self._critic_terminate_token = critic_terminate_token
        self._critic_approve_token = critic_approve_token
        self._critic_revise_token = critic_revise_token
        self._output_dir = "."
        self._max_messages_to_return = max_messages_to_return
        self.all_messages = []  # Track all messages for context and selection

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """
        Process messages through the engineer team and critic team.
        
        Args:
            messages: Incoming messages.
            cancellation_token: Token to cancel the operation.
            
        Returns:
            Response object containing the last N messages from the interaction.
        """
        NUM_LAST_MESSAGES = min(self._max_messages_to_return, 50)  # Cap at 50 for safety
        original_messages = messages
        print(f"TOKEN ESTIMATE: engineer society: {estimate_tokens(messages)}")
        print(f"NUM MESSAGES: {len(messages)}")
        time.sleep(2)
        
        # Reset message tracking for this run
        self.all_messages = []
        
        # Add instruction for the engineer to save files in the correct directory
        engineer_directory_instruction = TextMessage(
            content=f"""IMPORTANT FILE PATH INSTRUCTIONS:

ALL output files (plots, data, etc.) MUST be saved in this exact directory:
{self._output_dir}

Examples of correct file paths:
- plt.savefig('{self._output_dir}/histogram.png')
- df.to_csv('{self._output_dir}/results.csv')
- np.save('{self._output_dir}/array_data.npy')

Do NOT save files to the current directory or any other location. Always use '{self._output_dir}/' as the path prefix.
""",
            source="User"
        )
        
        # Add engineering heuristics and best practices
        engineering_heuristics = TextMessage(
            content="""ENGINEERING BEST PRACTICES AND TROUBLESHOOTING HEURISTICS:

When implementing your solution, follow these heuristics:

1. DATA UNDERSTANDING:
   - Always start by exploring and understanding the data structure (column names, data types, missing values)
   - Print shapes, descriptive statistics, and a few sample rows first
   - Check for missing data, outliers, or unusual distributions before proceeding

2. TROUBLESHOOTING APPROACH:
   - When you encounter an error, simplify your code to isolate the problem
   - Test individual components separately before combining them
   - Print intermediate results to verify each step works as expected
   - When debugging, start with the simplest possible version of your code

3. DEVELOPMENT STRATEGY:
   - Start small with atomic, focused steps that do one thing well
   - Test each component separately before combining them
   - Build up complexity incrementally, verifying at each step
   - Use intermediate data files to break complex processes into manageable stages

4. PERFORMANCE AND QUALITY:
   - Use sampling for initial testing when working with large datasets
   - Monitor memory usage and optimize for large data processing
   - Create clear, informative visualizations with proper labels and titles
   - Add useful comments explaining WHY, not just WHAT your code does

5. DATA SPLITTING BEST PRACTICES:
   - Always maintain stratification for important variables when splitting
   - Check the distributions in your train/test splits to ensure they're representative
   - Verify there's no data leakage between splits
   - Use k-fold cross-validation when appropriate to ensure stable results

6. OUTPUT VALIDATION:
   - Generate summary statistics for each data split and compare them
   - Create plots showing distributions across splits to visually confirm balance
   - Use statistical tests to determine the level of similarity between splits
   - Create clear tables showing counts and percentages of key variables across splits

Remember to check your results at each step and build up complexity gradually.
""",
            source="User"
        )
        
        # Add lab notebook reminder
        notebook_reminder = TextMessage(
            content='''IMPORTANT: DOCUMENT YOUR WORK

At the end of your implementation, use the write_notebook tool to document your work in the lab notebook:
1. Record important data insights 
2. Record significant implementation decisions
3. Record key metrics and evaluation results

Example:
write_notebook(
    entry=f"""Implemented data splitting with stratification. Results:

    | Variable | Chi-Square Statistic | P-Value |
    |----------|----------------------|---------|
    | Age      | 12.5                 | 0.005   |
    | Tissue   | 15.2                 | 0.001   |

    """,
    entry_type="OUTPUT",
    source="implementation_engineer"
)

The notebook is a critical scientific record that Team A will use to plan the next steps.
''',
            source="User"
        )
        
        # Add the instructions to the messages
        engineer_messages_with_instructions = list(messages) + [engineer_directory_instruction, engineering_heuristics, notebook_reminder]
        
        # Run the engineer team with the given messages
        result_engineer = await Console(self._engineer_team.run_stream(task=engineer_messages_with_instructions, cancellation_token=cancellation_token), output_stats=True)
        
        engineer_messages = result_engineer.messages
        engineer_messages = [message for message in engineer_messages if isinstance(message, TextMessage)]
        engineer_messages = [message for message in engineer_messages if not "error" in message.content.lower()]
        print(f"TOKEN ESTIMATE: engineer team: {estimate_tokens(engineer_messages)}")
        print(f"NUM MESSAGES: {len(engineer_messages)}")
        time.sleep(2)
        
        # In the last message, remove the engineer_terminate_token
        if len(engineer_messages) > 0:
            engineer_messages[-1].content = engineer_messages[-1].content.replace(self._engineer_terminate_token, "")
        
        if len(engineer_messages) > NUM_LAST_MESSAGES:
            last_messages_engineer = engineer_messages[-NUM_LAST_MESSAGES:]
        else:
            last_messages_engineer = engineer_messages
        
        # Store the engineer messages
        self.all_messages.extend(last_messages_engineer)
        
        last_message_critic = None
        revision_counter = 0
        while True:
            # Run the critic team with the updated messages
            if last_message_critic is not None:
                messages_for_critic = original_messages + [last_message_critic] + last_messages_engineer
            else:
                messages_for_critic = original_messages + last_messages_engineer
            
            # Add explicit instruction for critic to use tools
            tool_instruction_message = TextMessage(
                content=f"""TOOLS AVAILABLE FOR YOUR REVIEW:

The following tools can help you evaluate the implementation:
- search_directory("{self._output_dir}", "*.png") to find visualization files
- analyze_plot("{self._output_dir}/filename.png") to examine any visualizations of interest
- search_directory("{self._output_dir}", "*") to see all output files
- read_notebook() to see the project history and context

You can use these tools as needed to support your assessment. Tools are particularly helpful for examining visualizations that seem relevant to your evaluation. In your first review, examining some visualizations is recommended but not mandatory.

In follow-up reviews, you can focus primarily on whether the engineer addressed your previous feedback and only analyze plots that are new or relevant to the changes.

IMPORTANT: After completing your review, if you APPROVE the implementation, also document significant metrics and results in the lab notebook:

Example:
write_notebook(
    entry='''
    Model evaluation results:
    key results table:

    | Metric | Value |
    |--------|-------|
    | Pearson correlation | 0.87 |
    | MAE | 3.2 years |

    Key finding: model performs well on blood samples but shows higher error on brain tissue samples.

    Files generated:
    - model_evaluation.png
    - model_evaluation.arrow
    ''',
    entry_type="OUTPUT",
    source="data_science_critic"
)
""",
                source="User"
            )
            messages_for_critic.append(tool_instruction_message)
            
            print(f"TOKEN ESTIMATE: critic team before run {revision_counter}: {estimate_tokens(messages_for_critic)}")
            print(f"NUM MESSAGES: {len(messages_for_critic)}")
            time.sleep(2)
            result_critic = await Console(self._critic_team.run_stream(task=messages_for_critic, cancellation_token=cancellation_token), output_stats=True)
            critic_messages = result_critic.messages
            
            critic_messages = [message for message in critic_messages if isinstance(message, TextMessage)]
            print(f"TOKEN ESTIMATE: critic team after run {revision_counter}: {estimate_tokens(critic_messages)}")
            print(f"NUM MESSAGES: {len(critic_messages)}")
            time.sleep(2)

            # Store the last message
            last_message_critic = critic_messages[-1]
            
            # Check for approval BEFORE removing tokens
            approves = self._critic_approve_token in last_message_critic.content
            revises = self._critic_revise_token in last_message_critic.content
            
            # Remove tokens after checking
            last_message_critic.content = last_message_critic.content.replace(self._critic_terminate_token, "")
            last_message_critic.content = last_message_critic.content.replace(self._critic_revise_token, "")
            last_message_critic.content = last_message_critic.content.replace(self._critic_approve_token, "")
            
            # Add the critic message to the collection
            self.all_messages.append(last_message_critic)
            
            # Check if critic approves the work
            if approves:
                break
            elif not revises:
                print(f"Warning: Critic didn't provide a clear approval or revision token")
                # Continue anyway with revision
                
            revision_counter += 1
            if revision_counter > 3:
                break

            # Run the engineer team with the updated messages
            print(f"TOKEN ESTIMATE: engineer team before run {revision_counter}: {estimate_tokens(last_messages_engineer + [last_message_critic])}")
            print(f"NUM MESSAGES: {len(last_messages_engineer + [last_message_critic])}")
            time.sleep(2)
            
            # Add directory and notebook reminders before running engineer again
            directory_reminder = TextMessage(
                content=f"""IMPORTANT REMINDER: ALL output files (plots, data, etc.) MUST be saved in:
{self._output_dir}

Examples of correct paths:
- plt.savefig('{self._output_dir}/histogram.png')
- df.to_csv('{self._output_dir}/results.csv')""",
                source="User"
            )
            
            # Add troubleshooting reminder
            troubleshooting_reminder = TextMessage(
                content="""TROUBLESHOOTING REMINDER:

1. When fixing errors or addressing feedback:
   - Start by understanding exactly what's not working or what feedback needs to be addressed
   - Break down the problem into smaller parts
   - Test each part separately to find which component needs fixing
   - Make one change at a time and test its effect

2. For visualization issues:
   - Add proper titles, labels, and legends to all plots
   - Use appropriate color schemes
   - Include statistical context in the visualization
   - Save all plots to the correct output directory
""",
                source="User"
            )
            
            # Add feedback acknowledgment requirement
            feedback_acknowledgment_reminder = TextMessage(
                content="""CRITICAL REQUIREMENT: Once you receive feedback from the critic, you MUST explicitly acknowledge each point of feedback before implementing changes.

Your response MUST begin with:

"I acknowledge the following feedback points from the data science critic:
1. [Restate first feedback point from the critic]
2. [Restate second feedback point from the critic]
3. [Restate third feedback point from the critic]
...etc.

My implementation plan to address each point:
1. [Your plan to address the first point]
2. [Your plan to address the second point]
3. [Your plan to address the third point]
...etc."

DO NOT proceed with code implementation until you have explicitly acknowledged each feedback point from the critic.
""",
                source="User"
            )
            
            # Combine the messages with reminders
            engineer_iteration_messages = original_messages + last_messages_engineer + [last_message_critic, directory_reminder, troubleshooting_reminder, feedback_acknowledgment_reminder]
            
            # Run the engineer team with updated messages
            result_engineer = await Console(self._engineer_team.run_stream(task=engineer_iteration_messages, cancellation_token=cancellation_token), output_stats=True)
            engineer_messages = result_engineer.messages
            
            engineer_messages = [message for message in engineer_messages if isinstance(message, TextMessage)]
            engineer_messages = [message for message in engineer_messages if not "error" in message.content.lower()]
            print(f"TOKEN ESTIMATE: engineer team after run {revision_counter}: {estimate_tokens(engineer_messages)}")
            print(f"NUM MESSAGES: {len(engineer_messages)}")
            time.sleep(2)
            
            # Process the engineer messages
            if len(engineer_messages) > 0:
                # in the last message, remove the engineer_terminate_token
                engineer_messages[-1].content = engineer_messages[-1].content.replace(self._engineer_terminate_token, "")
                if len(engineer_messages) > NUM_LAST_MESSAGES:
                    last_messages_engineer = engineer_messages[-NUM_LAST_MESSAGES:]
                else:
                    last_messages_engineer = engineer_messages
                self.all_messages.extend(last_messages_engineer)

        # Instead of using a summarizer, return the last N messages
        # Get the last N messages from the full interaction
        if len(self.all_messages) > self._max_messages_to_return:
            last_n_messages = self.all_messages[-self._max_messages_to_return:]
        else:
            last_n_messages = self.all_messages
        
        # Combine the messages into a single message for the response
        combined_content = self._format_message_history(last_n_messages)
        final_message = TextMessage(content=combined_content, source=self.name)
        
        return Response(chat_message=final_message, inner_messages=self.all_messages)
    
    def _format_message_history(self, messages):
        """Format a list of messages into a readable history."""
        formatted_history = "# TEAM B IMPLEMENTATION REPORT\n\n"
        
        for i, message in enumerate(messages):
            agent_name = message.source
            formatted_history += f"\n## Message {i+1} from {agent_name}\n"
            formatted_history += f"{message.content}\n"
            formatted_history += f"{'=' * 80}\n"
        
        return formatted_history

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the agent."""
        # Reset the inner teams
        await self._engineer_team.reset()
        await self._critic_team.reset()
        # Clear message history
        self.all_messages = []

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        """Return the message types this agent can produce."""
        return (TextMessage,) 