#!/usr/bin/env python3
import unittest
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_all_tests():
    """Run all test cases in the tests directory"""
    # Discover all tests in the current directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__), pattern='test_*.py')
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return the appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run STAGED tests')
    parser.add_argument('--pattern', type=str, default='test_*.py',
                       help='Pattern to match test files (default: test_*.py)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Increase output verbosity')
    
    args = parser.parse_args()
    
    # Set verbosity
    verbosity = 2 if args.verbose else 1
    
    # Discover tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__), pattern=args.pattern)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1) 