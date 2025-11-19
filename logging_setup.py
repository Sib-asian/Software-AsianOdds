#!/usr/bin/env python3
"""
Logging Setup Module
====================

Centralized logging configuration for automation system.
Sets up rotating file handler and console output.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def init_logging():
    """
    Initialize logging with RotatingFileHandler and StreamHandler.
    
    Configuration:
    - LOG_DIR from AUTOMATION_LOG_DIR env var (default: './logs')
    - RotatingFileHandler: 10MB max size, 5 backup files
    - Format: timestamp - logger name - level - message
    
    Returns:
        logging.Logger: Configured root logger
    """
    # Get log directory from environment or use default
    log_dir = os.getenv('AUTOMATION_LOG_DIR', './logs')
    
    # Create log directory if it doesn't exist
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        # Log to console if directory creation fails
        print(f"Warning: Failed to create log directory '{log_dir}': {e}")
        print(f"Falling back to console-only logging")
        # Setup console-only logging as fallback
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        return logging.getLogger()
    
    # Setup log file path
    log_file = os.path.join(log_dir, 'automation_24h.log')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add StreamHandler for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add RotatingFileHandler for file output
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # Log warning to console if file handler setup fails
        root_logger.warning(
            f"Failed to setup file logging to '{log_file}': {e}. "
            f"Continuing with console-only logging."
        )
    
    return root_logger
