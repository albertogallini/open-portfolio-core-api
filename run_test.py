#!/usr/bin/env python3

import subprocess
import sys
import os

def run_tests():
    """Run all test files in the tests directory using the local virtual environment."""
    
    # Get the path to the Python executable in the virtual environment
    venv_python = os.path.join(os.getcwd(), 'venv/bin/python3')
    
    # Check if the virtual environment exists
    if not os.path.exists(venv_python):
        print("Virtual environment not found at: {}".format(venv_python))
        sys.exit(1)
    
    # Get the path to pytest in the virtual environment
    venv_pytest = os.path.join(os.getcwd(), 'venv/bin/pytest')
    
    if not os.path.exists(venv_pytest):
        print("Pytest not found in virtual environment at: {}".format(venv_pytest))
        sys.exit(1)
    
    # Get the list of test files in the tests directory
    tests_dir = os.path.join(os.getcwd(), 'tests')
    
    if not os.path.exists(tests_dir):
        print("Tests directory not found at: {}".format(tests_dir))
        sys.exit(1)
    
    # Get all Python test files in the tests directory (excluding directories and __pycache__)
    test_files = []
    for item in os.listdir(tests_dir):
        if (item.endswith('.py') or 
            (os.path.isdir(os.path.join(tests_dir, item)) and not item.startswith('__'))):
            test_files.append(os.path.join(tests_dir, item))
    
    if not test_files:
        print("No test files found in the tests directory.")
        sys.exit(0)
    
    # Sort the test files for consistent ordering
    test_files.sort()
    
    print("Running {} test files in the tests directory...".format(len(test_files)))
    for f in test_files:
        print("  - {}".format(f))
    
    # Run pytest with all test files
    cmd = [venv_pytest] + test_files
    
    try:
        result = subprocess.run(cmd, cwd=os.getcwd(), check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nTest run interrupted by user.")
        sys.exit(130)

if __name__ == "__main__":
    run_tests()