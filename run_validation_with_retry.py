#!/usr/bin/env python
"""
Wrapper script to run validation with proper error handling and retry capability.
This script helps handle keyboard interrupts and other errors gracefully.
"""

import sys
import os
import time
import signal
import subprocess
import argparse

def signal_handler(sig, frame):
    """Handle keyboard interrupt gracefully."""
    print("\n\nScript interrupted by user. Exiting gracefully...")
    sys.exit(0)

def run_command_with_retry(command, max_retries=3, retry_delay=30):
    """
    Run a command with retry capability.
    
    Args:
        command: The command to run as a list of strings
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        
    Returns:
        True if the command succeeded, False otherwise
    """
    retries = 0
    while retries <= max_retries:
        try:
            print(f"Running command: {' '.join(command)}")
            result = subprocess.run(command, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
            retries += 1
            if retries <= max_retries:
                print(f"Retrying in {retry_delay} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Maximum retries ({max_retries}) reached. Giving up.")
                return False
        except KeyboardInterrupt:
            print("\n\nCommand interrupted by user.")
            choice = input("Do you want to retry this command? (y/n): ")
            if choice.lower() == 'y':
                print("Retrying command...")
                continue
            else:
                print("Skipping command and continuing with the next step...")
                return False

def main():
    """Main function to parse arguments and run the validation script."""
    parser = argparse.ArgumentParser(description='Run validation with retry capability')
    parser.add_argument('--script', required=True, help='Validation script to run')
    parser.add_argument('--mode', default='extractor', help='Mode to run the script in')
    parser.add_argument('--prompts', required=True, help='Path to prompts file')
    parser.add_argument('--paper', required=True, help='Path to paper file')
    parser.add_argument('--checklist', default='Li-Paper', help='Checklist type')
    parser.add_argument('--retries', type=int, default=3, help='Maximum number of retries')
    parser.add_argument('--delay', type=int, default=30, help='Delay between retries in seconds')
    
    args = parser.parse_args()
    
    # Set up signal handler for keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)
    
    # Build the command
    command = [
        'python',
        args.script,
        '--mode', args.mode,
        '--prompts', args.prompts,
        '--paper', args.paper,
        '--checklist', args.checklist
    ]
    
    # Run the command with retry
    success = run_command_with_retry(command, args.retries, args.delay)
    
    if success:
        print(f"Successfully ran {args.script} for paper {os.path.basename(args.paper)}")
    else:
        print(f"Failed to run {args.script} for paper {os.path.basename(args.paper)}")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
