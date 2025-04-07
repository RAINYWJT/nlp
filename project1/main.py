#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for Chinese Text Correction task.
This script provides a framework for analyzing and correcting errors in Chinese text.
"""

import json
import argparse
from typing import Dict, List, Any
from tqdm import tqdm
import numpy as np

# Import modules
from data_analysis import analyze_data, visualize_error_distribution
from rule_based import RuleBasedCorrector
from statistical import StatisticalMLCorrector,StatisticalNgramCorrector
from ensemble import OnlineEnsembleCorrector, MultiEnsembleCorrector
from nn import NNCorrector
from evaluation import evaluate_performance, print_detailed_metrics
from sklearn.model_selection import train_test_split
from itertools import product


def grid_search_optimize(train_data):
    train_subset, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)
    lambda_1_values = [0.1, 0.5, 1, 2]
    lambda_2_values = [0.5, 1, 2, 3]
    lambda_3_values = [1, 2, 3, 4]
    lambda_4_values = [0.5, 1, 1.5, 2]

    best_score = -float('inf')
    best_params = None

    param_combinations = list(product(lambda_1_values, lambda_2_values, lambda_3_values, lambda_4_values))
    for lambda_1, lambda_2, lambda_3, lambda_4 in tqdm(param_combinations):
        corrector = StatisticalNgramCorrector(lambda_1, lambda_2, lambda_3, lambda_4)
        corrector.train(train_subset)
        correct_count = sum(
            1 for sample in valid_data if corrector.correct(sample['source']) == sample['target']
        )
        score = correct_count / len(valid_data)
        if score > best_score:
            best_score = score
            best_params = (lambda_1, lambda_2, lambda_3, lambda_4)

    return {"best_params": best_params, "best_score": best_score}


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from jsonl file.

    Args:
        file_path: Path to the jsonl file.

    Returns:
        List of dictionaries containing the data.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def main():
    """
    Main function to run the Chinese text correction pipeline.
    """
    # 输入格式如下:
    # python3 main.py --method xxx --analyze 1 --statistical_method xxx
    parser = argparse.ArgumentParser(description='Chinese Text Correction')
    parser.add_argument('--train_file', type=str, default='data/train.jsonl', help='Path to training data')
    parser.add_argument('--test_file', type=str, default='data/test.jsonl', help='Path to test data')
    parser.add_argument(
        '--method',
        type=str,
        choices=['rule', 'statistical', 'ensemble', 'nn', 'ol'],
        default='statistical',
        help='Correction method to use',
    )
    parser.add_argument('--analyze', type=int, default=0, help='Perform data analysis')
    parser.add_argument('--statistical_method', type=str, default='ml', help='Statistical method to use')
    parser.add_argument('--statistical_optparam', type=int, default=0, help='Statistical girdsearch to use')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    train_data = load_data(args.train_file)
    test_data = load_data(args.test_file)

    # Data analysis
    # print(args.analyze)
    if args.analyze == 1:
        print("\nPerforming data analysis...")
        analysis_results = analyze_data(train_data)
        visualize_error_distribution(analysis_results)

    # Initialize corrector based on method
    if args.method == 'rule':
        print("\nInitializing rule-based corrector...")
        corrector = RuleBasedCorrector()
        corrector.train(train_data)
        # assert 0
    elif args.method == 'statistical':
        print("\nInitializing statistical corrector...")
        if args.statistical_method == 'ngram':
            if args.statistical_optparam == 1:
                print('ngram grid search!')
                best_result = grid_search_optimize(train_data)
                best_lambda_1, best_lambda_2, best_lambda_3, best_lambda_4 = best_result["best_params"]
                print('best param', best_lambda_1, best_lambda_2, best_lambda_3, best_lambda_4)
                corrector = StatisticalNgramCorrector(lambda_1=best_lambda_1, 
                                                    lambda_2=best_lambda_2, 
                                                    lambda_3=best_lambda_3, 
                                                    lambda_4=best_lambda_4)
                corrector.train(train_data)
            else:
                corrector = StatisticalNgramCorrector()
                corrector.train(train_data)
        elif args.statistical_method == 'ml':
            corrector = StatisticalMLCorrector()
            corrector.train(train_data)
        else:
            print('There is no method named ' + args.statistical_method)
    elif args.method == 'nn':
        corrector = NNCorrector(train_data=train_data)
        corrector.train_model(train_data)

    elif args.method == 'ensemble':
        print("\nInitializing ensemble corrector...")
        # TODO start
        # Implement ensemble method that combines rule-based and statistical methods

        # You can implement a simple voting mechanism or a more sophisticated ensemble method
        # For example, you could use rule-based method first, then apply statistical method on the results
        # Or you could use different methods for different types of errors
        # TODO end
        # 2个corrector
        # rule_corrector = RuleBasedCorrector()
        # stat_corrector = StatisticalNgramCorrector()
        # corrector = EnsembleCorrector(rule_corrector, stat_corrector)
        # corrector.train(train_data)
        # 3个corrector，理论上说，你可以集成无数个。至于为什么这里集成3个，因为3个效果好，问就是其他情况我都试过。
        corrector = MultiEnsembleCorrector([StatisticalMLCorrector() ,RuleBasedCorrector(), StatisticalNgramCorrector()])
        corrector.train(train_data)

    elif args.method == 'ol':
        print(f"\nRunning online learning with online learning ...")
        all_metrics = []
        
        for seed in range(10):
            print(f"\n--- Trial {seed+1}/{10} (seed={seed}) ---")

            rule_corr = RuleBasedCorrector()
            stat_corr = StatisticalNgramCorrector()
            # 我们不使用ml的学习器做在线集成，因为并不是所有的算法支持partial fit
            corrector = OnlineEnsembleCorrector(rule_corr, stat_corr, seed= seed)
            corrector.train(train_data)
            
            predictions = []
            for sample in tqdm(test_data, ncols=100):
                corrected = corrector.correct(sample)
                predictions.append({
                    'source': sample['source'],
                    'prediction': corrected,
                    'target': sample['target'],
                    'label': sample['label']
                })
            
            metrics = evaluate_performance(predictions)
            all_metrics.append(metrics['final_score'])
            print(f"Accuracy: {metrics['final_score']:.4f}")
        
        # 汇总统计结果
        print("\n=== Final Summary ===")
        avg_accuracy = np.mean(all_metrics)
        std_accuracy = np.std([m for m in all_metrics])
        print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        return 
    
    else:
        print('There is no method named ' + args.method)

    # Evaluate on test data
    print("\nEvaluating on test data...")
    predictions = []
    for sample in tqdm(test_data, ncols=100):
        source = sample['source']
        corrected = corrector.correct(source)

        # print(sample['label'])
        # print(sample['source'])
        # print(corrected)
        # print(sample['target'])
        # print(' ')
        predictions.append(
            {'source': source, 'prediction': corrected, 'target': sample['target'], 'label': sample['label']}
        )

    # Calculate evaluation metrics
    metrics = evaluate_performance(predictions)
    print_detailed_metrics(metrics)


if __name__ == "__main__":
    main()
