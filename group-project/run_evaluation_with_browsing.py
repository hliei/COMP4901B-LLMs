"""
Run evaluation with search + browsing agent on Natural Questions dataset.

This script evaluates the BONUS browsing functionality.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from src.agent import SearchAgent
from src.llm_client import DeepSeekClient


def load_dataset(file_path: str):
    """Load JSONL dataset."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_predictions(predictions, output_path: str):
    """Save predictions to JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    print(f"\n✅ Predictions saved to: {output_path}")


def save_trajectories(trajectories, output_path: str):
    """Save agent trajectories to JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + '\n')
    print(f"✅ Trajectories saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate search + browsing agent')
    parser.add_argument('--data', type=str, default='data/nq_test_100.jsonl',
                       help='Path to test data')
    parser.add_argument('--output_pred', type=str, default='results/predictions_browsing.jsonl',
                       help='Path to save predictions')
    parser.add_argument('--output_traj', type=str, default='results/trajectories_browsing.jsonl',
                       help='Path to save trajectories')
    parser.add_argument('--max_search_steps', type=int, default=3,
                       help='Maximum search steps')
    parser.add_argument('--num_results', type=int, default=3,
                       help='Number of search results per query')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed debug information')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔍 Search + Browse Agent Evaluation")
    print("="*60)
    print(f"Dataset: {args.data}")
    print(f"Max search steps: {args.max_search_steps}")
    print(f"Results per search: {args.num_results}")
    print(f"Verbose: {args.verbose}")
    print("="*60)
    
    # Check if browsing is available
    try:
        from src.browsing_tool import browse_webpage
        print("✅ Browsing tool is available")
    except ImportError:
        print("❌ ERROR: Browsing tool not available!")
        print("Please install required packages: pip install beautifulsoup4 requests")
        return
    
    # Load dataset
    print("\n📂 Loading dataset...")
    dataset = load_dataset(args.data)
    print(f"Loaded {len(dataset)} questions")
    
    # Initialize agent
    print("\n🤖 Initializing agent...")
    llm_client = DeepSeekClient()
    agent = SearchAgent(
        llm_client=llm_client,
        max_search_steps=args.max_search_steps,
        num_search_results=args.num_results,
        verbose=args.verbose
    )
    
    # Run evaluation
    print("\n🚀 Running evaluation with search + browsing...\n")
    predictions = []
    trajectories = []
    
    for item in tqdm(dataset, desc="Processing"):
        question = item['question']
        ground_truths = item['answers']
        
        try:
            # Use browsing-enabled agent
            trajectory = agent.answer_question_with_browsing(question)
            answer = trajectory['final_answer']
            
            # Save prediction
            pred = {
                "id": item['id'],
                "question": question,
                "answers": ground_truths,
                "llm_response": answer
            }
            predictions.append(pred)
            
            # Save trajectory (with ground truth)
            trajectory_with_gt = {
                "id": item['id'],
                "question": question,
                "ground_truths": ground_truths,
                "trajectory": {
                    "question": trajectory["question"],
                    "steps": trajectory["steps"],
                    "final_answer": trajectory["final_answer"],
                    "total_search_steps": trajectory["total_search_steps"],
                    "total_browse_steps": trajectory["total_browse_steps"]
                }
            }
            trajectories.append(trajectory_with_gt)
            
        except Exception as e:
            print(f"\n❌ Error processing question '{question}': {e}")
            # Save empty prediction
            pred = {
                "id": item['id'],
                "question": question,
                "answers": ground_truths,
                "llm_response": f"Error: {str(e)}"
            }
            predictions.append(pred)
    
    # Save results
    print("\n" + "="*60)
    print("💾 Saving results...")
    save_predictions(predictions, args.output_pred)
    save_trajectories(trajectories, args.output_traj)
    
    # Print statistics
    print("\n" + "="*60)
    print("📊 Statistics:")
    total_searches = sum(t['trajectory']['total_search_steps'] for t in trajectories)
    total_browses = sum(t['trajectory']['total_browse_steps'] for t in trajectories)
    avg_searches = total_searches / len(trajectories) if trajectories else 0
    avg_browses = total_browses / len(trajectories) if trajectories else 0
    
    print(f"Total questions: {len(dataset)}")
    print(f"Total search steps: {total_searches}")
    print(f"Total browse steps: {total_browses}")
    print(f"Avg searches per question: {avg_searches:.2f}")
    print(f"Avg browses per question: {avg_browses:.2f}")
    print("="*60)
    
    print("\n✅ Evaluation complete!")
    print("\n📋 Next steps:")
    print(f"1. Run EM evaluation:")
    print(f"   PYTHONPATH=. python scripts/grade_with_em.py --input {args.output_pred} --output grading_results_browsing_em.json")
    print(f"\n2. Run LLM Judge evaluation:")
    print(f"   PYTHONPATH=. python scripts/grade_with_llm_judge.py --input {args.output_pred} --output grading_results_browsing_llm_judge.json")


if __name__ == "__main__":
    main()
