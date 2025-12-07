"""
AIME Agent with tool calling support.
"""

import json
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from .tools import execute_python_code, get_tool_definition

# Load environment variables from .env file
load_dotenv()


class AIMEAgent:
    """Agent for solving AIME problems with optional Python tool support."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "deepseek-chat",
        temperature: float = 0.6,
        max_steps: int = 20,
        use_tools: bool = True,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the AIME agent.
        
        Args:
            api_key: API key for the LLM service
            base_url: Base URL for the API
            model: Model name to use
            temperature: Sampling temperature
            max_steps: Maximum number of reasoning steps
            use_tools: Whether to enable Python code execution tool
            max_tokens: Maximum tokens to generate (None for no limit)
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model
        self.temperature = temperature
        self.max_steps = max_steps
        self.use_tools = use_tools
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Tool definitions
        self.tools = [get_tool_definition()] if use_tools else None
        
    def get_system_prompt(self, use_tools: bool = True) -> str:
        """Get the system prompt for the agent."""
        if use_tools:
            return """You are solving AIME (American Invitational Mathematics Examination) problems.

You have access to Python code execution via the execute_python_code tool. Use it to perform calculations, verify solutions, or explore patterns.

Put your final answer in \\boxed{} format.

Available Python modules: math, fractions, itertools, sympy, numpy

Think step by step and use Python to verify your calculations when needed."""
        else:
            return """You are solving AIME (American Invitational Mathematics Examination) problems. Put your final answer in \\boxed{} format."""
    
    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """
        Solve an AIME problem using the agent.
        
        Args:
            problem: The problem statement
            
        Returns:
            Dictionary containing:
            - response: Final response text
            - messages: Full conversation history
            - steps: Number of steps taken
        """
        # Initialize conversation
        messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(self.use_tools)
            },
            {
                "role": "user",
                "content": problem
            }
        ]
        
        steps = 0
        
        # Agent loop
        while steps < self.max_steps:
            steps += 1
            
            # Call LLM
            try:
                # Prepare common parameters
                api_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature
                }
                if self.max_tokens is not None:
                    api_params["max_tokens"] = self.max_tokens
                
                if self.use_tools and self.tools:
                    api_params["tools"] = self.tools
                    api_params["tool_choice"] = "auto"
                    response = self.client.chat.completions.create(**api_params)
                else:
                    response = self.client.chat.completions.create(**api_params)
                    
                message = response.choices[0].message
                
                # Add assistant message to history
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls if hasattr(message, 'tool_calls') else None
                })
                
                # Check if there are tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        if tool_call.function.name == "execute_python_code":
                            # Parse arguments
                            args = json.loads(tool_call.function.arguments)
                            code = args.get("code", "")
                            
                            # Execute code
                            result = execute_python_code(code)
                            
                            # Format result for LLM
                            result_text = ""
                            if result['success']:
                                if result['stdout']:
                                    result_text += f"Output:\n{result['stdout']}\n"
                                if result['return_value']:
                                    result_text += f"Return value: {result['return_value']}\n"
                                if not result_text:
                                    result_text = "Code executed successfully (no output)"
                            else:
                                result_text = f"Error:\n{result['error']}"
                            
                            # Add tool response to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result_text
                            })
                else:
                    # No tool calls, we're done
                    break
                    
            except Exception as e:
                # Handle API errors
                error_message = f"Error calling LLM: {str(e)}"
                messages.append({
                    "role": "assistant",
                    "content": error_message
                })
                break
        
        # Extract final response
        final_response = ""
        for msg in reversed(messages):
            if msg["role"] == "assistant" and msg.get("content"):
                final_response = msg["content"]
                break
        
        return {
            "response": final_response,
            "messages": messages,
            "steps": steps
        }
    
    def solve_problem_simple(self, problem: str) -> str:
        """
        Solve a problem with a single turn (no tool use).
        
        Args:
            problem: The problem statement
            
        Returns:
            The response text
        """
        messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(use_tools=False)
            },
            {
                "role": "user",
                "content": problem
            }
        ]
        
        try:
            api_params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "timeout": 60.0
            }
            if self.max_tokens is not None:
                api_params["max_tokens"] = self.max_tokens
            
            response = self.client.chat.completions.create(**api_params)
            return response.choices[0].message.content
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return f"Error: {str(e)}\nDetails: {error_detail}"
