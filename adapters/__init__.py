"""Public adapter-factory API used by the main application."""

from adapters.registry import adapter_names, create_adapter

__all__ = ["adapter_names", "create_adapter"]
