#!/usr/bin/env python3
"""
Basic test to verify the test structure works without external dependencies.
"""

import unittest
import sys
import os

# Add the project root to the path to import knexus
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

try:
    import knexus
except ImportError:
    print("Warning: knexus module not found. Tests will be skipped.")
    knexus = None


def get_first_runtime_with_device():
    """Helper function to get the first runtime that has at least one device."""
    runtimes = knexus.get_runtimes()
    for runtime in runtimes:
        devices = runtime.get_devices()
        if len(devices) > 0:
            return runtime, devices[0]
    return None, None


@unittest.skipIf(knexus is None, "knexus module not available")
class TestBasic(unittest.TestCase):
    """Basic test cases that don't require numpy."""

    def test_knexus_import(self):
        """Test that knexus module can be imported."""
        self.assertIsNotNone(knexus)
        self.assertTrue(hasattr(knexus, 'get_runtimes'))

    def test_runtime_discovery(self):
        """Test runtime discovery."""
        runtimes = knexus.get_runtimes()
        self.assertIsInstance(runtimes, list)
        # Should have at least one runtime (CPU fallback)
        self.assertGreater(len(runtimes), 0)

    def test_device_discovery(self):
        """Test device discovery."""
        runtimes = knexus.get_runtimes()
        
        total_devices = 0
        for runtime in runtimes:
            devices = runtime.get_devices()
            total_devices += len(devices)
        
        # Should have at least one device (CPU fallback)
        self.assertGreater(total_devices, 0)

    def test_buffer_creation(self):
        """Test basic buffer creation."""
        runtime, device = get_first_runtime_with_device()
        if device is not None:
            buffer = device.create_buffer(1024)
            self.assertIsNotNone(buffer)
            self.assertEqual(buffer.get_size(), 1024)


class TestFramework(unittest.TestCase):
    """Test cases that don't require knexus module."""

    def test_unittest_framework(self):
        """Test that unittest framework works."""
        self.assertTrue(True)
        self.assertEqual(1 + 1, 2)

    def test_import_structure(self):
        """Test that import structure is correct."""
        # Test that we can import the test modules
        # Note: This will fail if numpy is not available, which is expected
        try:
            import test_system
            import test_runtime
            import test_buffer
            import test_schedule
            import test_properties
            import test_integration
            self.assertTrue(True)  # All imports succeeded
        except ImportError as e:
            # This is expected if dependencies are missing
            error_msg = str(e)
            self.assertTrue(
                "numpy" in error_msg or "knexus" in error_msg,
                f"Unexpected import error: {error_msg}"
            )

    def test_file_structure(self):
        """Test that all test files exist."""
        test_files = [
            'test_system.py',
            'test_runtime.py',
            'test_buffer.py',
            'test_schedule.py',
            'test_properties.py',
            'test_integration.py',
            'test_basic.py',
            'run_tests.py',
            '__init__.py',
            'README.md'
        ]
        
        for test_file in test_files:
            self.assertTrue(
                os.path.exists(test_file),
                f"Test file {test_file} should exist"
            )


if __name__ == '__main__':
    unittest.main() 