#!/usr/bin/env python
"""Script to validate all challenges in the challenges directory."""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bioagents.config.loader import load_and_validate_task


def validate_challenge(challenge_dir: Path) -> Dict[str, Any]:
    """
    Validate a single challenge directory.
    
    Args:
        challenge_dir: Path to challenge directory
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'name': challenge_dir.name,
        'path': str(challenge_dir),
        'valid': False,
        'errors': [],
        'warnings': [],
        'task_config': None
    }
    
    # Check if directory exists
    if not challenge_dir.exists():
        result['errors'].append(f"Directory does not exist: {challenge_dir}")
        return result
    
    # Check required files
    required_files = ['task.yaml', 'README.md']
    for file_name in required_files:
        file_path = challenge_dir / file_name
        if not file_path.exists():
            result['errors'].append(f"Missing required file: {file_name}")
    
    # Check required directories
    required_dirs = ['data/agent', 'data/eval', 'scripts']
    for dir_name in required_dirs:
        dir_path = challenge_dir / dir_name
        if not dir_path.exists():
            result['errors'].append(f"Missing required directory: {dir_name}")
    
    # Check evaluation script
    eval_script = challenge_dir / 'scripts' / 'evaluate.py'
    if not eval_script.exists():
        result['errors'].append("Missing evaluation script: scripts/evaluate.py")
    elif not eval_script.is_file():
        result['errors'].append("scripts/evaluate.py is not a file")
    
    # Try to load and validate task configuration
    try:
        task_config = load_and_validate_task(challenge_dir)
        result['task_config'] = {
            'name': task_config.name,
            'display_name': task_config.display_name,
            'version': task_config.version,
            'agent_data_files': len(task_config.available_data.agent_data),
            'eval_data_files': len(task_config.available_data.eval_data),
            'metrics': len(task_config.evaluation.metrics),
            'required_outputs': len(task_config.evaluation.required_outputs)
        }
        
        # Check if data files actually exist
        for data_file in task_config.available_data.agent_data:
            file_path = challenge_dir / data_file.path
            if not file_path.exists():
                result['errors'].append(f"Agent data file not found: {data_file.path}")
        
        for data_file in task_config.available_data.eval_data:
            file_path = challenge_dir / data_file.path
            if not file_path.exists():
                result['errors'].append(f"Eval data file not found: {data_file.path}")
        
        # Warnings for optional files
        optional_files = ['Dockerfile']
        for file_name in optional_files:
            file_path = challenge_dir / file_name
            if not file_path.exists():
                result['warnings'].append(f"Optional file missing: {file_name} (will use default)")
        
    except Exception as e:
        result['errors'].append(f"Task configuration validation failed: {str(e)}")
    
    # Mark as valid if no errors
    result['valid'] = len(result['errors']) == 0
    
    return result


def validate_all_challenges(challenges_dir: Path) -> List[Dict[str, Any]]:
    """
    Validate all challenges in the challenges directory.
    
    Args:
        challenges_dir: Path to challenges directory
        
    Returns:
        List of validation results for each challenge
    """
    results = []
    
    if not challenges_dir.exists():
        print(f"Challenges directory not found: {challenges_dir}")
        return results
    
    # Find all challenge directories (ignore files and hidden directories)
    challenge_dirs = [
        d for d in challenges_dir.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ]
    
    if not challenge_dirs:
        print(f"No challenge directories found in: {challenges_dir}")
        return results
    
    # Validate each challenge
    for challenge_dir in sorted(challenge_dirs):
        print(f"Validating challenge: {challenge_dir.name}")
        result = validate_challenge(challenge_dir)
        results.append(result)
    
    return results


def print_validation_summary(results: List[Dict[str, Any]]) -> None:
    """Print a summary of validation results."""
    total_challenges = len(results)
    valid_challenges = sum(1 for r in results if r['valid'])
    invalid_challenges = total_challenges - valid_challenges
    
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total challenges: {total_challenges}")
    print(f"Valid challenges: {valid_challenges}")
    print(f"Invalid challenges: {invalid_challenges}")
    
    if invalid_challenges > 0:
        print(f"\n{'='*60}")
        print(f"INVALID CHALLENGES")
        print(f"{'='*60}")
        
        for result in results:
            if not result['valid']:
                print(f"\n❌ {result['name']}")
                for error in result['errors']:
                    print(f"   Error: {error}")
                for warning in result['warnings']:
                    print(f"   Warning: {warning}")
    
    if valid_challenges > 0:
        print(f"\n{'='*60}")
        print(f"VALID CHALLENGES")
        print(f"{'='*60}")
        
        for result in results:
            if result['valid']:
                print(f"\n✅ {result['name']}")
                if result['task_config']:
                    tc = result['task_config']
                    print(f"   Display Name: {tc['display_name']}")
                    print(f"   Version: {tc['version']}")
                    print(f"   Agent Data Files: {tc['agent_data_files']}")
                    print(f"   Eval Data Files: {tc['eval_data_files']}")
                    print(f"   Metrics: {tc['metrics']}")
                
                if result['warnings']:
                    for warning in result['warnings']:
                        print(f"   Warning: {warning}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Validate BioAgents challenges')
    parser.add_argument('--challenges-dir', default='challenges', 
                       help='Path to challenges directory (default: challenges)')
    parser.add_argument('--challenge', help='Validate specific challenge only')
    parser.add_argument('--quiet', action='store_true', help='Only show summary')
    
    args = parser.parse_args()
    
    challenges_dir = Path(args.challenges_dir)
    
    if args.challenge:
        # Validate specific challenge
        challenge_dir = challenges_dir / args.challenge
        result = validate_challenge(challenge_dir)
        results = [result]
    else:
        # Validate all challenges
        results = validate_all_challenges(challenges_dir)
    
    if not args.quiet:
        print_validation_summary(results)
    
    # Exit with error code if any challenges are invalid
    invalid_count = sum(1 for r in results if not r['valid'])
    if invalid_count > 0:
        print(f"\n❌ {invalid_count} challenge(s) failed validation")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(results)} challenge(s) passed validation")


if __name__ == "__main__":
    main() 