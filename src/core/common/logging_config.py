"""
Logging configuration for Amgix services.
"""
import logging
import os


def configure_logging():
    """
    Configure logging for Amgix services.
    
    Reads AMGIX_LOG_LEVEL environment variable (default: INFO).
    Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    Falls back to INFO with a warning if invalid value provided.
    """
    log_level_str = os.getenv("AMGIX_LOG_LEVEL", "INFO").upper()
    
    # Try to get the level, fall back to INFO if invalid
    try:
        log_level = getattr(logging, log_level_str)
        if not isinstance(log_level, int):  # Verify it's actually a level
            raise AttributeError
    except AttributeError:
        log_level = logging.INFO
        invalid = True
    else:
        invalid = False
    
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s [%(name)s] %(message)s"
    )
    
    if invalid:
        logging.getLogger(__name__).warning(
            f"Invalid AMGIX_LOG_LEVEL='{log_level_str}'. Using INFO. "
            f"Valid: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )

