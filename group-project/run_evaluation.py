"""
Run evaluation on Natural Questions dataset.

This script generates predictions for the NQ test set using:
1. Baseline (no search)
2. Search-augmented agent

Output files:
- results/predictions_nosearch.jsonl
- results/predictions_search.jsonl  
- results/agent_trajectories.jsonl
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

from src.agent import SearchAgent, answer_without_search
from src.llm_client import DeepSeekClient


def load_nq_data(file_path: str) -> List[Dict[str, Any]]:
    """Load Natural Questions data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of question dicts
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_predictions(predictions: List[Dict[str, Any]], output_file: str):
    """Save predictions to JSONL file.
    
    Args:
        predictions: List of prediction dicts
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(predictions)} predictions to {output_file}")


def save_trajectories(trajectories: List[Dict[str, Any]], output_file: str):
    """Save agent trajectories to JSONL file.
    
    Args:
        trajectories: List of trajectory dicts
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for traj in trajectories:
            # Remove 'formatted_results' from steps to keep file clean
            clean_traj = {
                "id": traj["id"],
                "question": traj["question"],
                "ground_truths": traj["ground_truths"],
                "trajectory": {
                    "question": traj["trajectory"]["question"],
                    "steps": [
                        {k: v for k, v in step.items() if k != "formatted_results"}
                        for step in traj["trajectory"]["steps"]
                    ],
                    "final_answer": traj["trajectory"]["final_answer"],
                    "total_search_steps": traj["trajectory"]["total_search_steps"]
                }
            }
            f.write(json.dumps(clean_traj, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(trajectories)} trajectories to {output_file}")


def run_baseline(
    data: List[Dict[str, Any]],
    output_file: str,
    verbose: bool = False
):
    """Run baseline evaluation (no search).
    
    Args:
        data: List of NQ questions
        output_file: Output file path
        verbose: Whether to print progress
    """
    print("\n" + "="*60)
    print("Running BASELINE evaluation (no search)")
    print("="*60)
    
    llm_client = DeepSeekClient()
    predictions = []
    
    for item in tqdm(data, desc="Generating answers"):
        question = item["question"]
        
        try:
            answer = answer_without_search(question, llm_client, verbose=False)
            
            prediction = {
                "id": item["id"],
                "question": question,
                "answers": item["answers"],
                "llm_response": answer
            }
            
            predictions.append(prediction)
            
            if verbose:
                print(f"\nQ: {question}")
                print(f"A: {answer}\n")
        
        except Exception as e:
            print(f"\nError processing {item['id']}: {e}")
            # Add empty response on error
            predictions.append({
                "id": item["id"],
                "question": question,
                "answers": item["answers"],
                "llm_response": f"Error: {str(e)}"
            })
    
    save_predictions(predictions, output_file)
    
    print(f"\nBaseline evaluation complete!")
    print(f"Total questions: {len(predictions)}")


def run_search_agent(
    data: List[Dict[str, Any]],
    predictions_file: str,
    trajectories_file: str,
    max_search_steps: int = 3,
    num_search_results: int = 3,
    verbose: bool = False
):
    """Run search agent evaluation.
    
    Args:
        data: List of NQ questions
        predictions_file: Output file for predictions
        trajectories_file: Output file for trajectories
        max_search_steps: Maximum search iterations per question
        num_search_results: Number of results per search
        verbose: Whether to print progress
    """
    print("\n" + "="*60)
    print("Running SEARCH AGENT evaluation")
    print("="*60)
    print(f"Max search steps: {max_search_steps}")
    print(f"Search results per query: {num_search_results}")
    
    agent = SearchAgent(
        max_search_steps=max_search_steps,
        num_search_results=num_search_results,
        verbose=verbose
    )
    
    predictions = []
    trajectories = []
    
    for item in tqdm(data, desc="Agent answering questions"):
        question = item["question"]
        
        try:
            trajectory = agent.answer_question(question)
            
            # Save prediction
            prediction = {
                "id": item["id"],
                "question": question,
                "answers": item["answers"],
                "llm_response": trajectory["final_answer"]
            }
            predictions.append(prediction)
            
            # Save trajectory
            trajectory_record = {
                "id": item["id"],
                "question": question,
                "ground_truths": item["answers"],
                "trajectory": trajectory
            }
            trajectories.append(trajectory_record)
            
            if verbose:
                print(f"\nQ: {question}")
                print(f"Steps: {trajectory['total_search_steps']}")
                print(f"A: {trajectory['final_answer']}\n")
        
        except Exception as e:
            print(f"\nError processing {item['id']}: {e}")
            # Add empty response on error
            predictions.append({
                "id": item["id"],
                "question": question,
                "answers": item["answers"],
                "llm_response": f"Error: {str(e)}"
            })
            trajectories.append({
                "id": item["id"],
                "question": question,
                "ground_truths": item["answers"],
                "trajectory": {
                    "question": question,
                    "steps": [],
                    "final_answer": f"Error: {str(e)}",
                    "total_search_steps": 0
                }
            })
    
    save_predictions(predictions, predictions_file)
    save_trajectories(trajectories, trajectories_file)
    
    print(f"\nSearch agent evaluation complete!")
    print(f"Total questions: {len(predictions)}")
    
    # Print statistics
    total_searches = sum(t["trajectory"]["total_search_steps"] for t in trajectories)
    avg_searches = total_searches / len(trajectories) if trajectories else 0
    print(f"Total searches performed: {total_searches}")
    print(f"Average searches per question: {avg_searches:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on Natural Questions dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/nq_test_100.jsonl',
        help='Input JSONL file with questions'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['baseline', 'search', 'both'],
        default='both',
        help='Evaluation mode: baseline (no search), search (with agent), or both'
    )
    parser.add_argument(
        '--max_search_steps',
        type=int,
        default=3,
        help='Maximum number of search steps per question'
    )
    parser.add_argument(
        '--num_search_results',
        type=int,
        default=3,
        help='Number of search results per query'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    data = load_nq_data(args.input)
    print(f"Loaded {len(data)} questions")
    
    # Run evaluations
    if args.mode in ['baseline', 'both']:
        run_baseline(
            data,
            output_file='results/predictions_nosearch.jsonl',
            verbose=args.verbose
        )
    
    if args.mode in ['search', 'both']:
        run_search_agent(
            data,
            predictions_file='results/predictions_search.jsonl',
            trajectories_file='results/agent_trajectories.jsonl',
            max_search_steps=args.max_search_steps,
            num_search_results=args.num_search_results,
            verbose=args.verbose
        )
    
    print("\n" + "="*60)
    print("All evaluations complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run EM evaluation:")
    print("   python grade_with_em.py --input results/predictions_nosearch.jsonl --output grading_results_nosearch_em.json")
    print("   python grade_with_em.py --input results/predictions_search.jsonl --output grading_results_search_em.json")
    print("\n2. Run LLM judge evaluation:")
    print("   python grade_with_llm_judge.py --input results/predictions_nosearch.jsonl --output grading_results_nosearch_judge.json")
    print("   python grade_with_llm_judge.py --input results/predictions_search.jsonl --output grading_results_search_judge.json")


if __name__ == "__main__":
    main()
