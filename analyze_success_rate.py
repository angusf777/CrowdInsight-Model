#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze success rates of Kickstarter projects across different dimensions.

This script analyzes historical Kickstarter project data to identify patterns
in success rates across various dimensions including categories, subcategories,
countries, funding goals, and campaign durations. It generates statistical analyses
and visualizations to help understand which factors correlate with project success.
Results provide insights for both model interpretation and project creators.

Example usage:
    python analyze_success_rate.py --input_json data/allProcessed.json

Author: Angus Fung
Date: April 2025
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict

def analyze_success_rate(data_path="data/allProcessed.json"):
    """
    Analyze the success rate of Kickstarter projects in the dataset
    
    Args:
        data_path: Path to the JSON data file
    """
    print(f"Analyzing success rate in: {data_path}")
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return
    
    # Load data
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} projects")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Count successful and failed projects
    successes = sum(1 for project in data if project.get('state', 0) == 1)
    failures = len(data) - successes
    success_rate = (successes / len(data)) * 100
    
    print("\n----- SUCCESS RATE ANALYSIS -----")
    print(f"Total projects: {len(data)}")
    print(f"Successful projects: {successes} ({success_rate:.2f}%)")
    print(f"Failed projects: {failures} ({100-success_rate:.2f}%)")
    
    # Additional analysis by categories
    analyze_by_category(data)
    analyze_by_numerical_features(data)
    
    # Create output directory for plots
    os.makedirs("analysis", exist_ok=True)
    
    # Plot success rate
    plt.figure(figsize=(10, 6))
    plt.bar(['Successful', 'Failed'], [successes, failures], color=['green', 'red'])
    plt.title('Kickstarter Project Outcomes')
    plt.ylabel('Number of Projects')
    plt.savefig('analysis/success_rate.png')
    print("\nSaved success rate chart to: analysis/success_rate.png")

def analyze_by_category(data):
    """Analyze success rate by project categories"""
    # This assumes there's some category information in the data
    # We might need to adapt this based on what's actually in the dataset
    
    try:
        # Check if category information is available based on embeddings presence
        if 'category_embedding' in data[0]:
            print("\n----- CATEGORY INFORMATION -----")
            print("Category data is present but represented as embeddings.")
            print("Statistical analysis by category would require mapping embeddings to category names.")
        else:
            print("\n----- CATEGORY INFORMATION -----")
            print("No category information found in the dataset.")
    except (IndexError, KeyError):
        print("\nUnable to analyze category information - no data or unexpected format")

def analyze_by_numerical_features(data):
    """Analyze how numerical features relate to success rate"""
    numerical_features = [
        'funding_goal', 
        'campaign_duration',
        'image_count',
        'video_count',
        'previous_projects_count',
        'previous_success_rate'
    ]
    
    print("\n----- NUMERICAL FEATURES ANALYSIS -----")
    
    for feature in numerical_features:
        try:
            # Skip if feature doesn't exist
            if feature not in data[0]:
                print(f"Feature '{feature}' not found in dataset")
                continue
                
            # Collect data for successful and failed projects
            success_values = [project[feature] for project in data if project.get('state', 0) == 1]
            failure_values = [project[feature] for project in data if project.get('state', 0) == 0]
            
            if not success_values or not failure_values:
                print(f"No valid data for feature: {feature}")
                continue
                
            # Calculate statistics
            success_avg = sum(success_values) / len(success_values)
            failure_avg = sum(failure_values) / len(failure_values)
            
            print(f"\n{feature.replace('_', ' ').title()}:")
            print(f"  Successful projects avg: {success_avg:.2f}")
            print(f"  Failed projects avg: {failure_avg:.2f}")
            print(f"  Difference: {(success_avg - failure_avg):.2f}")
            
            # Plot distributions
            plt.figure(figsize=(10, 6))
            plt.hist(success_values, alpha=0.5, label='Successful', color='green', bins=30)
            plt.hist(failure_values, alpha=0.5, label='Failed', color='red', bins=30)
            plt.title(f'Distribution of {feature.replace("_", " ").title()}')
            plt.xlabel(feature.replace('_', ' ').title())
            plt.ylabel('Number of Projects')
            plt.legend()
            plt.savefig(f'analysis/{feature}_distribution.png')
            print(f"  Saved distribution chart to: analysis/{feature}_distribution.png")
            
        except Exception as e:
            print(f"Error analyzing feature '{feature}': {e}")

if __name__ == "__main__":
    # Check for custom data path as argument
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/allProcessed.json"
    analyze_success_rate(data_path) 