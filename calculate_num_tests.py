#!/usr/bin/env python3
import os
import ast
import sys
from pathlib import Path
from collections import defaultdict

def count_functions(file_path):
    """Count the number of functions in a Python file using AST."""
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            tree = ast.parse(file.read())
        except SyntaxError as e:
            print(f"Syntax error parsing {file_path}: {e}")
            return 0

    # Count top-level functions
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    
    # Count methods in classes
    class_methods = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_methods.extend([n for n in node.body if isinstance(n, ast.FunctionDef)])
    
    return len(functions), len(class_methods)

def main():
    base_dir = Path("pset")
    
    # Get all directories in pset that don't start with underscore
    folders = [f for f in base_dir.iterdir() if f.is_dir() and not f.name.startswith('_')]
    
    total_functions = 0
    total_methods = 0
    breakdown = {}
    missing_files = []
    
    print(f"Analyzing {len(folders)} folders in {base_dir}...")
    
    for folder in sorted(folders):
        test_file = folder / "paper2code_test.py"
        
        if test_file.exists():
            num_functions, num_methods = count_functions(test_file)
            total_functions += num_functions
            total_methods += num_methods
            breakdown[folder.name] = (num_functions, num_methods, num_functions + num_methods)
            print(f"{folder.name:30} - Functions: {num_functions:3}, Methods: {num_methods:3}, Total: {num_functions + num_methods:3}")
        else:
            missing_files.append(folder.name)
            print(f"{folder.name:30} - No paper2code_test.py file found")
    
    print("\nSummary:")
    print(f"Total folders analyzed: {len(folders)}")
    print(f"Folders with test files: {len(breakdown)}")
    print(f"Folders without test files: {len(missing_files)}")
    if missing_files:
        print(f"Missing test files in: {', '.join(missing_files)}")
    
    print(f"\nTotal functions: {total_functions}")
    print(f"Total methods: {total_methods}")
    print(f"Total functions + methods: {total_functions + total_methods}")
    
    # Sort by total count for a ranked list
    if breakdown:
        print("\nRanked by total number of functions/methods:")
        for idx, (folder, (funcs, methods, total)) in enumerate(sorted(breakdown.items(), key=lambda x: x[1][2], reverse=True), 1):
            print(f"{idx:2}. {folder:30} - Functions: {funcs:3}, Methods: {methods:3}, Total: {total:3}")

if __name__ == "__main__":
    main()
