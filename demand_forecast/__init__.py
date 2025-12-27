"""Demand Forecast - Time series forecasting with Transformer neural networks."""

import logging
from importlib.metadata import version

__version__ = version("demand-forecast")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    # Enhanced format string
    format="%(asctime)s [%(levelname)s] %(name)s | %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure logging level for the package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger("demand_forecast").setLevel(numeric_level)
