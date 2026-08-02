"""Stub the apify SDK so the pure-logic modules can be tested without it."""
import sys
import types

if "apify" not in sys.modules:
    module = types.ModuleType("apify")

    class _Log:
        def __getattr__(self, name):
            return lambda *a, **k: None

    class _Actor:
        log = _Log()

    module.Actor = _Actor
    sys.modules["apify"] = module
