"""Support running from a checkout with PYTHONPATH=src."""

from pathlib import Path

# Let Python find blackjackgpt submodules directly under src in a checkout.
if __name__ == "blackjackgpt":
    __path__ = [str(Path(__file__).resolve().parent)]
    if __spec__ is not None:
        __spec__.submodule_search_locations = __path__

from blackjackgpt.blackjack import (
    VERSION1_RULES,
    BlackjackEngine,
    RulesConfig,
)

__version__ = "0.1.0"

__all__ = ["VERSION1_RULES", "BlackjackEngine", "RulesConfig", "__version__"]
