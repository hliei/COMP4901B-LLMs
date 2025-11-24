"""
LLM client for DeepSeek API.

This module provides functions to interact with DeepSeek LLM via OpenAI-compatible API.
"""

import os
from openai import OpenAI
from typing import List, Dict, Any, Optional

# Load environment variables from .env
from src.utils import load_env
load_env()


class DeepSeekClient:
    """Client for DeepSeek LLM API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key (if None, uses DEEPSEEK_API_KEY from env)
            base_url: API base URL (if None, uses DEEPSEEK_BASE_URL from env)
            model: Model name (if None, uses 'deepseek-chat')
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.base_url = base_url or os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
        self.model = model or os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
        
        if not self.api_key:
            raise ValueError("DeepSeek API key not configured")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """Generate response from LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for API call
        
        Returns:
            Generated text response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
    
    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        tool_choice: str = "auto"
    ) -> Dict[str, Any]:
        """Generate response with tool calling support.
        
        Args:
            messages: List of message dicts
            tools: List of tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tool_choice: Tool choice strategy ("auto", "none", or specific tool)
        
        Returns:
            Dict with 'content' (text response) and 'tool_calls' (list of tool calls)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            message = response.choices[0].message
            
            result = {
                "content": message.content or "",
                "tool_calls": []
            }
            
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            return result
        
        except Exception as e:
            raise Exception(f"LLM generation with tools failed: {str(e)}")


def create_simple_prompt(question: str) -> List[Dict[str, str]]:
    """Create a simple prompt for answering questions without search.
    
    Args:
        question: Question to answer
    
    Returns:
        List of message dicts
    """
    system_prompt = """You are a helpful assistant that answers questions accurately and concisely.

Important instructions:
1. Provide direct, factual answers
2. Be concise and specific
3. If you're not certain, say so
4. Format your final answer clearly"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\n\nProvide a clear and concise answer."}
    ]

