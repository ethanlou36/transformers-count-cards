"""Support running from a checkout with PYTHONPATH=src."""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parent.parent))

from .blackjack import VERSION1_RULES, BlackjackEngine, RulesConfig  # noqa: E402

__version__ = "0.1.0"

__all__ = ["VERSION1_RULES", "BlackjackEngine", "RulesConfig", "__version__"]
