#!/usr/bin/env python3
"""
run_all.py
----------
Master script to run the complete VisionXplain pipeline:
1. Run all tests
2. Train all models (CNN, ViT, Hybrid)
3. Evaluate all models
4. Run computational benchmarks
5. Generate explainability visualizations
6. Generate summary report

Usage:
    python run_all.py [--skip-tests] [--skip-training] [--skip-evaluation] [--skip-benchmarks] [--skip-explainability]
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime
import json

# Project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Color codes for output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_step(step_num, total, text):
    """Print a step header."""
    print(f"\n{Colors.OKCYAN}[{step_num}/{total}] {text}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'-'*70}{Colors.ENDC}")


def run_command(cmd, description, check=True):
    """Run a shell command and handle errors."""
    print(f"\n{Colors.OKBLUE}Running: {description}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}Command: {' '.join(cmd)}{Colors.ENDC}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            check=check,
            capture_output=False,
            text=True
        )
        if result.returncode == 0:
            print(f"\n{Colors.OKGREEN}✓ {description} completed successfully{Colors.ENDC}")
            return True
        else:
            print(f"\n{Colors.FAIL}✗ {description} failed with return code {result.returncode}{Colors.ENDC}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"\n{Colors.FAIL}✗ {description} failed: {e}{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"\n{Colors.FAIL}✗ {description} error: {e}{Colors.ENDC}")
        return False


def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"{Colors.OKGREEN}✓ {description} exists{Colors.ENDC}")
        return True
    else:
        print(f"{Colors.WARNING}⚠ {description} not found{Colors.ENDC}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run complete VisionXplain pipeline')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running tests')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip model evaluation')
    parser.add_argument('--skip-benchmarks', action='store_true', help='Skip computational benchmarks')
    parser.add_argument('--skip-explainability', action='store_true', help='Skip explainability testing')
    parser.add_argument('--data-dir', type=str, default='data/processed/chest_xray', 
                       help='Path to processed data directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    
    args = parser.parse_args()
    
    print_header("VisionXplain - Complete Pipeline Runner")
    print(f"{Colors.BOLD}Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")
    
    results = {
        'start_time': datetime.now().isoformat(),
        'steps': {}
    }
    
    step_num = 1
    total_steps = 5
    
    # Step 1: Run Tests
    if not args.skip_tests:
        print_step(step_num, total_steps, "Running Tests")
        success = run_command(
            ['python', '-m', 'pytest', 'tests/', '-v', '--tb=short'],
            'Running all tests'
        )
        results['steps']['tests'] = {'success': success}
        step_num += 1
    else:
        print(f"\n{Colors.WARNING}⚠ Skipping tests{Colors.ENDC}")
        results['steps']['tests'] = {'success': None, 'skipped': True}
    
    # Step 2: Train Models
    if not args.skip_training:
        print_step(step_num, total_steps, "Training Models")
        
        models_to_train = ['cnn', 'vit', 'hybrid']
        training_results = {}
        
        for model in models_to_train:
            print(f"\n{Colors.OKBLUE}Training {model.upper()} model...{Colors.ENDC}")
            
            # Adjust batch size for ViT and Hybrid (they need smaller batches)
            batch_size = args.batch_size if model == 'cnn' else args.batch_size // 2
            
            # Adjust learning rate
            lr = 1e-4 if model in ['cnn', 'hybrid'] else 3e-5
            
            cmd = [
                'python', 'src/training/train.py',
                '--model', model,
                '--epochs', str(args.epochs),
                '--batch', str(batch_size),
                '--img_size', str(args.img_size),
                '--lr', str(lr),
                '--data_dir', args.data_dir
            ]
            
            success = run_command(cmd, f'Training {model.upper()} model', check=False)
            training_results[model] = success
            
            # Check if model file was created
            model_path = project_root / 'outputs' / 'models' / f'best_{model}_model.pt'
            check_file_exists(model_path, f'{model.upper()} model checkpoint')
        
        results['steps']['training'] = training_results
        step_num += 1
    else:
        print(f"\n{Colors.WARNING}⚠ Skipping training{Colors.ENDC}")
        results['steps']['training'] = {'skipped': True}
    
    # Step 3: Evaluate Models
    if not args.skip_evaluation:
        print_step(step_num, total_steps, "Evaluating Models")
        
        models_to_evaluate = ['cnn', 'vit', 'hybrid']
        evaluation_results = {}
        
        for model in models_to_evaluate:
            print(f"\n{Colors.OKBLUE}Evaluating {model.upper()} model...{Colors.ENDC}")
            
            cmd = [
                'python', 'scripts/evaluation/evaluate_model.py',
                '--model', model,
                '--data_dir', args.data_dir,
                '--img_size', str(args.img_size)
            ]
            
            success = run_command(cmd, f'Evaluating {model.upper()} model', check=False)
            evaluation_results[model] = success
            
            # Check if evaluation results exist
            metrics_path = project_root / 'outputs' / 'evaluation' / f'{model}_test' / 'metrics.json'
            check_file_exists(metrics_path, f'{model.upper()} evaluation metrics')
        
        results['steps']['evaluation'] = evaluation_results
        step_num += 1
    else:
        print(f"\n{Colors.WARNING}⚠ Skipping evaluation{Colors.ENDC}")
        results['steps']['evaluation'] = {'skipped': True}
    
    # Step 4: Run Benchmarks
    if not args.skip_benchmarks:
        print_step(step_num, total_steps, "Computational Benchmarks")
        
        cmd = [
            'python', 'scripts/evaluation/run_benchmarks.py',
            '--output', 'outputs/benchmarks/computational_efficiency.json'
        ]
        
        success = run_command(cmd, 'Running computational benchmarks', check=False)
        results['steps']['benchmarks'] = {'success': success}
        
        # Check if benchmark results exist
        benchmark_path = project_root / 'outputs' / 'benchmarks' / 'computational_efficiency.json'
        check_file_exists(benchmark_path, 'Benchmark results')
        
        step_num += 1
    else:
        print(f"\n{Colors.WARNING}⚠ Skipping benchmarks{Colors.ENDC}")
        results['steps']['benchmarks'] = {'skipped': True}
    
    # Step 5: Explainability Testing
    if not args.skip_explainability:
        print_step(step_num, total_steps, "Explainability Testing")
        
        cmd = [
            'python', 'scripts/explainability/test_explainability.py',
            '--num_samples', '6',
            '--output_dir', 'outputs/explanations/gallery'
        ]
        
        success = run_command(cmd, 'Generating explainability visualizations', check=False)
        results['steps']['explainability'] = {'success': success}
        
        # Check if explanations exist
        gallery_dir = project_root / 'outputs' / 'explanations' / 'gallery'
        if gallery_dir.exists():
            num_files = len(list(gallery_dir.glob('*.png')))
            print(f"{Colors.OKGREEN}✓ Found {num_files} explanation images{Colors.ENDC}")
        
        step_num += 1
    else:
        print(f"\n{Colors.WARNING}⚠ Skipping explainability{Colors.ENDC}")
        results['steps']['explainability'] = {'skipped': True}
    
    # Summary
    results['end_time'] = datetime.now().isoformat()
    results['duration'] = str(datetime.fromisoformat(results['end_time']) - 
                              datetime.fromisoformat(results['start_time']))
    
    print_header("Pipeline Summary")
    
    # Print results summary
    for step_name, step_result in results['steps'].items():
        if isinstance(step_result, dict):
            if step_result.get('skipped'):
                print(f"{Colors.WARNING}⚠ {step_name.upper()}: Skipped{Colors.ENDC}")
            elif 'success' in step_result:
                status = f"{Colors.OKGREEN}✓ PASSED{Colors.ENDC}" if step_result['success'] else f"{Colors.FAIL}✗ FAILED{Colors.ENDC}"
                print(f"{step_name.upper()}: {status}")
            elif isinstance(step_result, dict) and any('success' in v for v in step_result.values() if isinstance(v, dict)):
                # For nested results (like training)
                passed = sum(1 for v in step_result.values() if isinstance(v, dict) and v.get('success'))
                total = len([v for v in step_result.values() if isinstance(v, dict)])
                print(f"{step_name.upper()}: {passed}/{total} completed")
    
    # Save results
    results_file = project_root / 'outputs' / 'pipeline_results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{Colors.OKBLUE}Results saved to: {results_file}{Colors.ENDC}")
    print(f"\n{Colors.BOLD}Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.BOLD}Total duration: {results['duration']}{Colors.ENDC}\n")
    
    # Exit with appropriate code
    all_passed = all(
        (isinstance(r, dict) and r.get('success', False)) or 
        (isinstance(r, dict) and 'skipped' in r) or
        (isinstance(r, dict) and all(v.get('success', False) for v in r.values() if isinstance(v, dict)))
        for r in results['steps'].values()
    )
    
    if all_passed:
        print(f"{Colors.OKGREEN}{Colors.BOLD}✓ All steps completed successfully!{Colors.ENDC}\n")
        return 0
    else:
        print(f"{Colors.WARNING}{Colors.BOLD}⚠ Some steps had issues. Check the output above.{Colors.ENDC}\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())

