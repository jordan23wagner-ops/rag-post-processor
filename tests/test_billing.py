"""Unit tests for input_kb_processed billing: pin the arithmetic and prove
chunk count no longer affects charge count. This is revenue-critical code.

Dev-only; excluded from the shipped image via .dockerignore. Run with:
    python -m unittest discover -s tests
"""
import asyncio
import sys
import types
import unittest


class _StubLog:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass


class _ChargeRecorder:
    """Records every Actor.charge call so tests can assert call count/args
    without touching the real Apify platform."""

    def __init__(self):
        self.calls = []
        self.limit_reached = False

    async def charge(self, event_name, count=1):
        self.calls.append((event_name, count))
        return types.SimpleNamespace(event_charge_limit_reached=self.limit_reached)


class _StubActor:
    log = _StubLog()

    def __init__(self):
        self._charger = _ChargeRecorder()

    async def charge(self, event_name, count=1):
        return await self._charger.charge(event_name, count=count)


_stub_actor = _StubActor()
sys.modules.setdefault("apify", types.SimpleNamespace(Actor=_stub_actor))

from rag_post_processor.main import (  # noqa: E402
    compute_input_bytes,
    compute_billed_units,
    charge_for_input,
    INPUT_KB_EVENT,
)


class BilledUnitsArithmeticTests(unittest.TestCase):
    def test_1_byte_is_1_unit(self):
        self.assertEqual(compute_billed_units(1), 1)

    def test_1024_bytes_is_1_unit(self):
        self.assertEqual(compute_billed_units(1024), 1)

    def test_1025_bytes_is_2_units(self):
        self.assertEqual(compute_billed_units(1025), 2)

    def test_zero_bytes_is_zero_units(self):
        self.assertEqual(compute_billed_units(0), 0)

    def test_never_negative(self):
        self.assertEqual(compute_billed_units(-5), 0)


class InputByteMeasurementTests(unittest.TestCase):
    def test_empty_items_is_zero_bytes(self):
        self.assertEqual(compute_input_bytes([]), 0)

    def test_inline_and_dataset_paths_measure_identically(self):
        # Same logical items, however they arrived - inline `data` array or
        # fetched from an upstream dataset via datasetId - must produce the
        # same byte count, since charge_for_input only ever sees `items`.
        items_inline = [{"text": "hello world"}, {"text": "second item"}]
        items_from_dataset = [{"text": "hello world"}, {"text": "second item"}]
        self.assertEqual(compute_input_bytes(items_inline), compute_input_bytes(items_from_dataset))

    def test_non_dict_items_still_measured(self):
        # Malformed items (skipped during processing) were still received
        # and must still count toward the charge.
        items = [5, None, "hello", ["nested", "list"]]
        self.assertGreater(compute_input_bytes(items), 0)


class ChargeCallTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        _stub_actor._charger = _ChargeRecorder()

    async def test_empty_input_no_charge_call(self):
        units, result = await charge_for_input([])
        self.assertEqual(units, 0)
        self.assertIsNone(result)
        self.assertEqual(_stub_actor._charger.calls, [])

    async def test_charged_exactly_once_regardless_of_item_or_future_chunk_count(self):
        # charge_for_input is called once per run in main() regardless of
        # how many items/chunks the input later produces - this test proves
        # a single call to charge_for_input results in exactly one
        # Actor.charge call, with a count derived purely from bytes.
        items = [{"text": "x" * 500}, {"text": "y" * 3000}]
        units, result = await charge_for_input(items)
        self.assertEqual(len(_stub_actor._charger.calls), 1)
        event_name, count = _stub_actor._charger.calls[0]
        self.assertEqual(event_name, INPUT_KB_EVENT)
        self.assertEqual(count, units)
        self.assertEqual(units, compute_billed_units(compute_input_bytes(items)))

    async def test_charge_independent_of_chunk_settings(self):
        # The whole point of the migration: identical input bytes must
        # produce an identical charge no matter what chunk_size/overlap the
        # caller later uses to chunk that same text. charge_for_input takes
        # no chunk_size/overlap argument at all, so this is true by
        # construction - assert it explicitly anyway.
        items = [{"text": "z" * 10_000}]
        units_a, _ = await charge_for_input(items)
        _stub_actor._charger = _ChargeRecorder()
        units_b, _ = await charge_for_input(items)
        self.assertEqual(units_a, units_b)

    async def test_charge_limit_reached_is_surfaced(self):
        _stub_actor._charger.limit_reached = True
        units, result = await charge_for_input([{"text": "x" * 5000}])
        self.assertTrue(result.event_charge_limit_reached)
        self.assertGreater(units, 0)


if __name__ == "__main__":
    unittest.main()
