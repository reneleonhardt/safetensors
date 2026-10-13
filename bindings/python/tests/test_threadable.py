import os
import subprocess
import sys
import sysconfig
import textwrap
import threading
import time
import unittest
from concurrent import futures

import numpy as np
import pytest
from safetensors.numpy import load_file

from safetensors import TensorSpec, serialize_file


class TestCase(unittest.TestCase):
    def test_serialize_file_releases_gil(self):
        """Test that serialize_file releases the GIL and can run concurrently."""
        # Create large numpy arrays to ensure serialization takes measurable time
        # Keep them alive throughout the test since we pass raw pointers
        tensor_a = np.random.randn(2000, 20000).astype(np.float32)
        tensor_b = np.random.randint(0, 128, (20000, 2000), dtype=np.int8)

        # Build the tensor dict with data pointers (as serialize_file expects)
        tensor_data = {
            "tensor_a": TensorSpec(
                dtype=tensor_a.dtype.name,
                shape=tensor_a.shape,
                data_ptr=tensor_a.ctypes.data,
                data_len=tensor_a.nbytes,
            ),
            "tensor_b": TensorSpec(
                dtype=tensor_b.dtype.name,
                shape=tensor_b.shape,
                data_ptr=tensor_b.ctypes.data,
                data_len=tensor_b.nbytes,
            ),
        }

        num_threads = 4
        results = {}
        barrier = threading.Barrier(num_threads)
        file_names = [f"tmp_thread_{i}.safetensors" for i in range(num_threads)]

        def saving_thread(thread_id):
            file_name = file_names[thread_id]
            # Wait for all threads to be ready
            barrier.wait()
            start_time = time.monotonic()
            serialize_file(tensor_data, file_name)
            end_time = time.monotonic()
            results[thread_id] = (start_time, end_time)

        try:
            # Run multiple serialize_file calls concurrently
            with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futs = [executor.submit(saving_thread, i) for i in range(num_threads)]
                for f in futs:
                    f.result()  # Raise any exceptions

            # Verify all threads completed
            self.assertEqual(len(results), num_threads)

            # Check that the threads actually ran concurrently by verifying
            # their execution windows overlap. If the GIL was held, threads
            # would run sequentially with no overlap.
            all_starts = [r[0] for r in results.values()]
            all_ends = [r[1] for r in results.values()]

            # The latest start should be before the earliest end if threads overlapped
            latest_start = max(all_starts)
            earliest_end = min(all_ends)

            # If GIL is released, threads run in parallel so latest_start < earliest_end
            # If GIL is NOT released, threads run sequentially so latest_start >= earliest_end
            self.assertLess(
                latest_start,
                earliest_end,
                f"Threads did not run concurrently - GIL may not be released. "
                f"Latest start: {latest_start}, Earliest end: {earliest_end}",
            )

            # Verify all output files are valid and contain correct data
            for file_name in file_names:
                loaded = load_file(file_name)
                np.testing.assert_array_equal(
                    loaded["tensor_a"],
                    tensor_a,
                    err_msg=f"tensor_a mismatch in {file_name}",
                )
                np.testing.assert_array_equal(
                    loaded["tensor_b"],
                    tensor_b,
                    err_msg=f"tensor_b mismatch in {file_name}",
                )
        finally:
            # Clean up all temporary files
            for file_name in file_names:
                if os.path.exists(file_name):
                    os.remove(file_name)


if __name__ == "__main__":
    unittest.main()


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="requires free-threaded Python",
)
def test_free_threaded_roundtrip():
    # A fresh interpreter catches imports that silently enable the GIL.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import sys
            assert not sys._is_gil_enabled()
            import numpy as np
            from safetensors import safe_open
            from safetensors.numpy import load, save, save_file
            from concurrent.futures import ThreadPoolExecutor
            from pathlib import Path
            from tempfile import TemporaryDirectory
            from threading import Barrier
            assert not sys._is_gil_enabled()
            barrier = Barrier(4)
            def roundtrip(worker):
                expected = np.arange(32, dtype=np.int32) + worker
                barrier.wait(timeout=10)
                with TemporaryDirectory() as directory:
                    path = str(Path(directory) / "tensor.safetensors")
                    for _ in range(10):
                        np.testing.assert_array_equal(load(save({"x": expected}))["x"], expected)
                        save_file({"x": expected}, path)
                        with safe_open(path, framework="np") as handle:
                            np.testing.assert_array_equal(handle.get_tensor("x"), expected)
                            np.testing.assert_array_equal(handle.get_slice("x")[2:7], expected[2:7])
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(roundtrip, range(4)))
            assert not sys._is_gil_enabled()
        """),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
