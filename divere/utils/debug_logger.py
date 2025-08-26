#!/usr/bin/env python3
"""
Debug Logger for DiVERE
Provides centralized logging functionality for debugging path resolution and config loading issues
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
import platform
import time
from datetime import datetime, timedelta


class DiVEREDebugLogger:
    """Centralized debug logger for DiVERE application"""
    
    _instance: Optional['DiVEREDebugLogger'] = None
    _logger: Optional[logging.Logger] = None
    _log_file: Optional[Path] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the debug logger"""
        # Check if debug logging is enabled
        self.debug_enabled = self._should_enable_debug()
        
        if not self.debug_enabled:
            return
        
        # Set up log file location
        self._log_file = self._get_log_file_path()
        self._ensure_log_directory()
        
        # Set up logger
        self._setup_logger()
        
        # Log initialization
        self.info("=" * 60)
        self.info(f"DiVERE Debug Logger initialized at {datetime.now()}")
        self.info(f"Platform: {platform.system()} {platform.release()}")
        self.info(f"Python: {sys.version}")
        self.info(f"Working directory: {Path.cwd()}")
        self.info(f"sys.argv[0]: {sys.argv[0]}")
        self.info(f"__file__: {__file__}")
        self.info("=" * 60)
    
    def _should_enable_debug(self) -> bool:
        """Check if debug logging should be enabled"""
        # Debug logging is always enabled
        return True
    
    def _get_user_config_dir(self) -> Path:
        """Get user configuration directory"""
        app_name = "DiVERE"
        system = platform.system()
        
        if system == "Darwin":  # macOS
            return Path.home() / "Library" / "Application Support" / app_name
        elif system == "Windows":
            return Path.home() / "AppData" / "Local" / app_name
        elif system == "Linux":
            return Path.home() / ".config" / app_name
        else:
            return Path.cwd() / "debug_logs"
    
    def _get_log_file_path(self) -> Path:
        """Get the path for the debug log file"""
        log_dir = self._get_user_config_dir() / "logs"
        
        # Use timestamp in filename for new sessions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"divere_debug_{timestamp}.log"
        
        return log_dir / log_filename
    
    def _ensure_log_directory(self):
        """Ensure the log directory exists and cleanup old logs"""
        if self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            # Clean up old log files after creating the directory
            self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """Remove log files older than 30 days"""
        if not self._log_file:
            return
        
        log_dir = self._log_file.parent
        cutoff_date = datetime.now() - timedelta(days=30)
        
        try:
            deleted_count = 0
            for log_file in log_dir.glob("divere_debug_*.log"):
                try:
                    # Get file modification time
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    
                    # Delete if older than 30 days
                    if file_mtime < cutoff_date:
                        log_file.unlink()
                        deleted_count += 1
                except (OSError, ValueError) as e:
                    # Skip files that can't be processed
                    continue
            
            if deleted_count > 0:
                print(f"Debug logger: Cleaned up {deleted_count} old log files (older than 30 days)")
                
        except Exception as e:
            # Silently fail if cleanup encounters issues
            print(f"Warning: Could not clean up old log files: {e}")
    
    def _setup_logger(self):
        """Set up the logger with file and console handlers"""
        self._logger = logging.getLogger('divere_debug')
        self._logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self._logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)8s] %(name)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        if self._log_file:
            try:
                file_handler = logging.FileHandler(self._log_file, encoding='utf-8')
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                self._logger.addHandler(file_handler)
            except Exception as e:
                print(f"Warning: Could not set up file logging: {e}")
        
        # Console handler (only for ERROR and above to avoid spam)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        
        # Prevent propagation to avoid duplicate messages
        self._logger.propagate = False
    
    def debug(self, message: str, module: str = None):
        """Log debug message"""
        if self.debug_enabled and self._logger:
            module_prefix = f"[{module}] " if module else ""
            self._logger.debug(f"{module_prefix}{message}")
    
    def info(self, message: str, module: str = None):
        """Log info message"""
        if self.debug_enabled and self._logger:
            module_prefix = f"[{module}] " if module else ""
            self._logger.info(f"{module_prefix}{message}")
    
    def warning(self, message: str, module: str = None):
        """Log warning message"""
        if self.debug_enabled and self._logger:
            module_prefix = f"[{module}] " if module else ""
            self._logger.warning(f"{module_prefix}{message}")
    
    def error(self, message: str, module: str = None):
        """Log error message"""
        if self.debug_enabled and self._logger:
            module_prefix = f"[{module}] " if module else ""
            self._logger.error(f"{module_prefix}{message}")
    
    def log_path_search(self, description: str, paths: list, found_path: Optional[str] = None, module: str = None):
        """Log path search details"""
        if not self.debug_enabled:
            return
        
        self.info(f"{description}", module)
        for i, path in enumerate(paths, 1):
            exists = Path(path).exists() if path else False
            status = "✓ EXISTS" if exists else "✗ missing"
            self.debug(f"  [{i}] {path} - {status}", module)
        
        if found_path:
            self.info(f"  → RESOLVED: {found_path}", module)
        else:
            self.warning(f"  → NO PATH FOUND", module)
    
    def log_file_operation(self, operation: str, file_path: str, success: bool = True, error: str = None, module: str = None):
        """Log file operation details"""
        if not self.debug_enabled:
            return
        
        status = "SUCCESS" if success else "FAILED"
        message = f"{operation}: {file_path} - {status}"
        
        if success:
            self.debug(message, module)
        else:
            self.error(f"{message} - {error}", module)
    
    def get_log_file_path(self) -> Optional[Path]:
        """Get the current log file path"""
        return self._log_file if self.debug_enabled else None
    
    def is_enabled(self) -> bool:
        """Check if debug logging is enabled"""
        return self.debug_enabled


# Global debug logger instance
debug_logger = DiVEREDebugLogger()

# Convenience functions
def debug(message: str, module: str = None):
    """Log debug message"""
    debug_logger.debug(message, module)

def info(message: str, module: str = None):
    """Log info message"""
    debug_logger.info(message, module)

def warning(message: str, module: str = None):
    """Log warning message"""
    debug_logger.warning(message, module)

def error(message: str, module: str = None):
    """Log error message"""
    debug_logger.error(message, module)

def log_path_search(description: str, paths: list, found_path: Optional[str] = None, module: str = None):
    """Log path search details"""
    debug_logger.log_path_search(description, paths, found_path, module)

def log_file_operation(operation: str, file_path: str, success: bool = True, error: str = None, module: str = None):
    """Log file operation details"""
    debug_logger.log_file_operation(operation, file_path, success, error, module)

def is_debug_enabled() -> bool:
    """Check if debug logging is enabled"""
    return debug_logger.is_enabled()

def get_log_file_path() -> Optional[Path]:
    """Get the current log file path"""
    return debug_logger.get_log_file_path()