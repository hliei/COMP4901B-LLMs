"""
Main script to run AIME problem solving with and without tools.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import AIMEAgent


def load_aime_data(data_path: str):
    """Load AIME problems from JSONL file."""
    problems = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems


def save_results(results, output_path: str):
    """Save results to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Run AIME problem solver')
    parser.add_argument('--mode', choices=['no_tool', 'with_tool'], required=True,
                       help='Mode: no_tool (baseline) or with_tool')
    parser.add_argument('--data', type=str, default='data/aime24.jsonl',
                       help='Path to AIME data file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for results')
    parser.add_argument('--num-rollouts', type=int, default=4,
                       help='Number of rollouts per problem')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Sampling temperature')
    parser.add_argument('--max-steps', type=int, default=20,
                       help='Maximum reasoning steps (only for with_tool mode)')
    parser.add_argument('--model', type=str, default='deepseek-chat',
                       help='Model name')
    parser.add_argument('--max-tokens', type=int, default=None,
                       help='Maximum tokens to generate (default: None for no limit)')
    
    args = parser.parse_args()
    
    # Load problems
    print(f"Loading problems from {args.data}...")
    problems = load_aime_data(args.data)
    print(f"Loaded {len(problems)} problems")
    
    # Initialize agent
    use_tools = (args.mode == 'with_tool')
    max_steps = args.max_steps if use_tools else 1
    
    agent = AIMEAgent(
        model=args.model,
        temperature=args.temperature,
        max_steps=max_steps,
        use_tools=use_tools,
        max_tokens=args.max_tokens
    )
    
    print(f"\nMode: {args.mode}")
    print(f"Temperature: {args.temperature}")
    print(f"Num rollouts: {args.num_rollouts}")
    print(f"Max steps: {max_steps}")
    print(f"Max tokens: {args.max_tokens if args.max_tokens else 'No limit'}")
    print(f"Use tools: {use_tools}")
    print()
    
    # Solve problems
    results = []
    total_iterations = len(problems) * args.num_rollouts
    
    with tqdm(total=total_iterations, desc="Solving problems") as pbar:
        for problem_data in problems:
            problem_id = problem_data['id']
            problem_text = problem_data['problem']
            gold_answer = problem_data['answer']
            
            # Run multiple rollouts
            for rollout_id in range(args.num_rollouts):
                try:
                    if use_tools:
                        # Use agent with tool calling
                        result = agent.solve_problem(problem_text)
                        llm_response = result['response']
                    else:
                        # Simple single-turn generation
                        llm_response = agent.solve_problem_simple(problem_text)
                    
                    # Store result
                    results.append({
                        'id': problem_id,
                        'rollout_id': rollout_id,
                        'problem': problem_text,
                        'answer': gold_answer,
                        'llm_response': llm_response,
                        'mode': args.mode,
                        'temperature': args.temperature,
                        'model': args.model
                    })
                    
                except Exception as e:
                    print(f"\nError on problem {problem_id}, rollout {rollout_id}: {str(e)}")
                    results.append({
                        'id': problem_id,
                        'rollout_id': rollout_id,
                        'problem': problem_text,
                        'answer': gold_answer,
                        'llm_response': f"Error: {str(e)}",
                        'mode': args.mode,
                        'temperature': args.temperature,
                        'model': args.model
                    })
                
                pbar.update(1)
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    save_results(results, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
