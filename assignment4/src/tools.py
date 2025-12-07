"""
Python code execution tool for the AIME solver agent.
"""

import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any


def execute_python_code(code: str) -> Dict[str, Any]:
    """
    Execute Python code in a controlled environment and return the results.
    
    Args:
        code: Python code string to execute
        
    Returns:
        Dictionary containing:
        - success: Boolean indicating if execution was successful
        - stdout: Standard output captured during execution
        - stderr: Standard error captured during execution  
        - return_value: The value of the last expression (if any)
        - error: Error message if execution failed
    """
    # Create string buffers to capture output
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    # Prepare the execution environment with safe modules
    exec_globals = {
        '__builtins__': __builtins__,
        'math': __import__('math'),
        'fractions': __import__('fractions'),
        'itertools': __import__('itertools'),
        'sympy': __import__('sympy'),
        'numpy': __import__('numpy'),
    }
    exec_locals = {}
    
    result = {
        'success': False,
        'stdout': '',
        'stderr': '',
        'return_value': None,
        'error': None
    }
    
    try:
        # Redirect stdout and stderr
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Try to evaluate as expression first (for return value)
            try:
                return_value = eval(code, exec_globals, exec_locals)
                result['return_value'] = str(return_value) if return_value is not None else None
                result['success'] = True
            except SyntaxError:
                # If it's not an expression, execute as statement
                exec(code, exec_globals, exec_locals)
                result['success'] = True
                
                # Try to get the last variable assigned or computed
                if exec_locals:
                    # Get the last non-private variable
                    non_private_vars = {k: v for k, v in exec_locals.items() if not k.startswith('_')}
                    if non_private_vars:
                        last_var = list(non_private_vars.values())[-1]
                        result['return_value'] = str(last_var)
                        
    except Exception as e:
        result['success'] = False
        result['error'] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
    # Capture output
    result['stdout'] = stdout_buffer.getvalue()
    result['stderr'] = stderr_buffer.getvalue()
    
    return result


def get_tool_definition() -> Dict[str, Any]:
    """
    Get the tool definition in OpenAI function calling format.
    
    Returns:
        Dictionary with tool definition
    """
    return {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": "Execute Python code to perform calculations and verify solutions. Available modules: math, fractions, itertools, sympy, numpy. Returns the output and any computed values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Can include calculations, imports from available modules, and variable assignments."
                    }
                },
                "required": ["code"]
            }
        }
    }
