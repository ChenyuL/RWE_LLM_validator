# src/utils/logger.py
import os
import logging
import sys
from datetime import datetime
from typing import Optional

def setup_logger(name: str = "llm_validation", 
                 log_file: Optional[str] = None, 
                 level: int = logging.INFO,
                 format_str: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file
        level: Logging level
        format_str: Custom format string for log entries
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Default format if not specified
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_str)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_transaction_logger(base_dir: str, transaction_id: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a specific transaction with a unique log file.
    
    Args:
        base_dir: Base directory for log files
        transaction_id: Optional transaction ID (generated if not provided)
        
    Returns:
        Logger instance for the transaction
    """
    if transaction_id is None:
        # Generate transaction ID from timestamp if not provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transaction_id = f"transaction_{timestamp}"
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(base_dir, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create log file path
    log_file = os.path.join(logs_dir, f"{transaction_id}.log")
    
    # Set up and return logger
    return setup_logger(
        name=f"llm_validation.{transaction_id}",
        log_file=log_file,
        level=logging.INFO
    )

class APICallLogger:
    """
    Logger for tracking API calls to LLM providers.
    Logs detailed information about each call for monitoring and debugging.
    """
    
    def __init__(self, base_logger: logging.Logger):
        """
        Initialize the API call logger.
        
        Args:
            base_logger: Base logger to use for logging
        """
        self.logger = base_logger
    
    def log_api_call(self, 
                    provider: str, 
                    model: str, 
                    prompt_tokens: int,
                    completion_tokens: int,
                    duration_ms: float,
                    status: str = "success",
                    error: Optional[str] = None) -> None:
        """
        Log an API call to an LLM provider.
        
        Args:
            provider: Name of the LLM provider (e.g., "openai", "anthropic")
            model: Model name (e.g., "gpt-4", "claude-3")
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            duration_ms: Duration of the call in milliseconds
            status: Status of the call ("success" or "error")
            error: Error message if status is "error"
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "duration_ms": duration_ms,
            "status": status
        }
        
        if error:
            log_entry["error"] = error
        
        # Log as INFO for successful calls, ERROR for failed calls
        if status == "success":
            self.logger.info(f"API Call: {log_entry}")
        else:
            self.logger.error(f"API Call Failed: {log_entry}")

# Example usage
if __name__ == "__main__":
    # Example of setting up a basic logger
    logger = setup_logger(log_file="example.log")
    logger.info("This is a test log entry")
    
    # Example of setting up a transaction logger
    txn_logger = get_transaction_logger("./logs")
    txn_logger.info("Starting transaction")
    
    # Example of using the API call logger
    api_logger = APICallLogger(txn_logger)
    api_logger.log_api_call(
        provider="openai",
        model="gpt-4",
        prompt_tokens=500,
        completion_tokens=200,
        duration_ms=1250.5
    )