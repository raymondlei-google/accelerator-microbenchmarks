"""Utility functions for microbenchmarking."""

import os
from typing import Dict, Any
import glob

import jax
import jsonlines
import numpy as np
import random
import string
import pathlib
import gzip
import json
import re
from collections import defaultdict
import subprocess
import shutil
import time
from jax.experimental import multihost_utils


def simple_timeit(
    f, *args, matrix_dim=None, warmup_tries=10, tries=10, task=None, trace_dir=None
) -> list[float]:
    """Simple utility to time a function for multiple runs."""
    assert task is not None

    if trace_dir:
        return timeit_from_trace(
            f,
            *args,
            matrix_dim=matrix_dim,
            warmup_tries=warmup_tries,
            tries=tries,
            task=task,
            trace_dir=trace_dir,
        )

    is_multihost = jax.process_count() > 1

    # --- Warmup Loop ---
    print(f"Running warmup loop with {warmup_tries} tries...")
    for _ in range(warmup_tries):
        result = f(*args)
    # Block until the final warmup run is complete on the local host.
    jax.block_until_ready(result)
    print("Warmup complete.")
    # --- Measurement Loop ---
    outcomes_ms = []

    # Final barrier after warmup to ensure all hosts are ready to start measuring together.
    if is_multihost:
        multihost_utils.sync_global_devices(f"warmup_done_{task}")

    print(f"Running measurement loop with {tries} tries...")
    for i in range(tries):
        s_time = time.perf_counter()

        jax.block_until_ready(f(*args))

        # Synchronize (Multi-Host Only): Wait for ALL hosts to finish the operation.
        if is_multihost:
            multihost_utils.sync_global_devices(f"end_run_{i}_{task}")

        e_time = time.perf_counter()
        outcomes_ms.append(1000 * (e_time - s_time))

    return outcomes_ms


def get_trace(log_dir: str) -> dict[str, Any]:
    """Extract the trace object from the log directory.

    Returns:
      A trace object in JSON format.
    """
    # Navigate to the folder with the latest trace dump to find `trace.json.jz`
    trace_folders = (pathlib.Path(log_dir).absolute() / "plugins" / "profile").iterdir()
    latest_trace_folder = max(trace_folders, key=os.path.getmtime)
    trace_jsons = latest_trace_folder.glob("*.trace.json.gz")
    try:
        (trace_json,) = trace_jsons
    except ValueError as value_error:
        raise ValueError(
            f"Invalid trace folder: {latest_trace_folder}"
        ) from value_error

    with gzip.open(trace_json, "rb") as f:
        trace = json.load(f)

    return trace


def get_metrics_from_trace(trace: dict[str, Any], task: str) -> float:
    event_matcher = re.compile(task)
    if "traceEvents" not in trace:
        raise KeyError("Key 'traceEvents' not found in trace.")

    events = []
    for e in trace["traceEvents"]:
        if "name" in e and event_matcher.match(e["name"]):
            events.append(e)

    durations_ms = []
    try:
        # Duration is in us.
        for e in events:
            durations_ms.append(e["dur"] / 1e3)
    except KeyError:
        print("KeyError: Key 'dur' not found in the event object")
        raise
    return durations_ms


def is_local_directory_path(dir: str) -> bool:
    """
    Returns true if the path is a local path.
    """
    if not dir:  # Handle None or empty string
        return False

    # Heuristics for local paths
    return dir.startswith("/") or dir.startswith("./") or dir.startswith("../")


def timeit_from_trace(
    f, *args, matrix_dim=None, warmup_tries=10, tries=10, task=None, trace_dir=None
) -> list[float]:
    """
    Time a function with jax.profiler and get the run time from the trace.
    """
    LOCAL_TRACE_DIR = "/tmp/microbenchmarks_tmptrace"
    is_multihost = jax.process_count() > 1

    # warmup loop
    print(f"Running warmup loop with {warmup_tries} tries...")
    for _ in range(warmup_tries):
        data = f(*args)
    jax.block_until_ready(data)
    if is_multihost:
        multihost_utils.sync_global_devices(f"warmup_done_{task}")

    if matrix_dim is not None:
        trace_name = f"{task}_dim_{matrix_dim}"
    else:
        trace_name = f"t_{task}_" + "".join(
            random.choices(string.ascii_uppercase + string.digits, k=10)
        )

    trace_full_dir = f"{trace_dir}/{trace_name}"
    tmp_trace_dir = trace_full_dir
    # If the trace_dir isn't a local path, create one for dumping the trace for parsing and getting metrics.
    if trace_dir and not is_local_directory_path(trace_dir):
        tmp_trace_dir = f"{LOCAL_TRACE_DIR}/{trace_name}"
    with jax.profiler.trace(tmp_trace_dir):
        for i in range(tries):
            with jax.profiler.TraceAnnotation(task):
                jax.block_until_ready(f(*args))
            if is_multihost:
                multihost_utils.sync_global_devices(f"end_run_{i}_{task}")
    trace = get_trace(tmp_trace_dir)

    if trace_full_dir != tmp_trace_dir:
        # Upload the traces to desired location
        upload_to_storage(remote_dir=trace_full_dir, local_file=tmp_trace_dir)
    return get_metrics_from_trace(trace, task)


def maybe_write_metrics_file(
    metrics_dir,
    metrics,
    metadata,
    test_name,
    test_start_time,
    test_end_time,
    remote_dir=None,
):
    """Writes metrics to a JSONL file to be consumed by the XLML metrics pipeline."""

    local_devices = jax.local_devices()
    tpu_worker_id = int(os.getenv("TPU_WORKER_ID", "0"))
    is_multislice = hasattr(local_devices[0], "slice_index")

    # For multi-slice workload, the result is only written by the first host on the first slice (slice_index=0, tpu_worker_id=0).
    # For single-slice workload, the result is only written by the first host (tpu_worker_id=0).
    if is_multislice:
        if local_devices[0].slice_index != 0 or tpu_worker_id != 0:
            return
    else:
        if tpu_worker_id != 0:
            return

    jsonl_name = "metrics_report.jsonl"
    jsonl_path = metrics_dir + "/" + jsonl_name
    metadata.update(
        {
            "testsuite": "microbenchmark",
            "test_name": f"{test_name}",
            "test_start_timestamp": f"{test_start_time}",
            "test_end_timestamp": f"{test_end_time}",
        }
    )
    metrics_data = {
        "metrics": metrics,
        "dimensions": metadata,
    }
    # Make sure the metadata value is a string.
    for key, value in metadata.items():
        metadata[key] = str(value)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    print(f"Writing metrics to JSONL file: {jsonl_path}")
    with jsonlines.open(jsonl_path, mode="a") as writer:
        writer.write(metrics_data)

    if remote_dir:
        upload_to_storage(remote_dir=remote_dir, local_file=jsonl_path)


def upload_to_storage(remote_dir: str, local_file: str):
    """
    Uploads a local file to a specified storage location.
    """

    if remote_dir.startswith("gs://"):  # Google Cloud Storage (GCS)
        try:
            subprocess.run(
                ["gsutil", "cp", "-r", local_file, remote_dir],
                check=True,
                capture_output=True,
            )

        except subprocess.CalledProcessError as e:
            print(
                f"Failed to upload '{local_file}' to GCS: '{remote_dir}'. Error: {e.stderr.decode()}"
            )
    else:
        raise KeyError(f"{remote_dir} is not a valid GCS path.")


class MetricsStatistics:
    """
    Represents statistics for a list of metrics.
    """

    def __init__(self, metrics_list, metrics_name: str):
        self.metrics_list = metrics_list
        self.metrics_name = metrics_name
        self.statistics = self._calculate_statistics()

    def _calculate_statistics(self) -> Dict[str, float]:
        """Calculates the statistics of the metrics list."""
        if not self.metrics_list:
            return {}  # Return an empty dict if metrics_list is empty
        return {
            "p50": np.percentile(self.metrics_list, 50),
            "p90": np.percentile(self.metrics_list, 90),
            "p95": np.percentile(self.metrics_list, 95),
            "p99": np.percentile(self.metrics_list, 99),
            "avg": np.mean(self.metrics_list),
        }

    def __repr__(self):
        return (
            f"MetricsStatistics(metrics_name='{self.metrics_name}', "
            f"statistics={self.statistics})"
        )

    def serialize_statistics(self):
        serialized = {}
        for stat_name, stat_value in self.statistics.items():
            serialized[f"{self.metrics_name}_{stat_name}"] = stat_value
        return serialized


def rename_xla_dump(
    tmp_xla_dump_dir: str,
    dest_xla_dump_dir: str,
    benchmark_name: str,
    benchmark_param: Dict[str, Any],
):
    """
    Finds the latest XLA dump file matching '*jit_f*before_optimizations*.txt',
    then identifies all other files that share the same 'jit_f.[unique_id]' identifier
    and renames them to 'benchmark_name_serialized_params.original_suffix_with_extension'.
    """

    serialized_benchmark_param = "_".join(
        f"{key}_{value}" for key, value in benchmark_param.items()
    )
    anchor_pattern = os.path.join(tmp_xla_dump_dir, "*jit_f*before_optimizations*.txt")
    matching_anchor_files = glob.glob(anchor_pattern)

    if not matching_anchor_files:
        print(
            f"No files found for anchor pattern: '{anchor_pattern}'. No files will be renamed."
        )
        return

    # Sort anchor files by modification time (latest first)
    matching_anchor_files.sort(key=os.path.getmtime, reverse=True)
    latest_anchor_file = matching_anchor_files[0]

    # Example: 'module_0080.jit_f.cl_747713181.before_optimizations.txt'
    # This will extract 'module_0080.jit_f.cl_747713181'
    filename_base = os.path.basename(latest_anchor_file)
    jit_id_match = re.search(r"(module.*jit_f\.[^.]+)", filename_base)

    if not jit_id_match:
        print(
            f"Could not extract 'jit_f.[unique_id]' from '{filename_base}'. Cannot proceed with renaming."
        )
        return

    common_jit_id_prefix = jit_id_match.group(1)

    # Find all files in the directory that contain this specific common_jit_id_prefix
    all_related_files_pattern = os.path.join(
        tmp_xla_dump_dir, f"*{common_jit_id_prefix}*"
    )
    all_related_files = glob.glob(all_related_files_pattern)

    if not all_related_files:
        print(
            f"No files found containing '{common_jit_id_prefix}'. This is unexpected if an anchor was found."
        )
        return

    new_base_name = f"{benchmark_name}_{serialized_benchmark_param}"

    for original_filepath in all_related_files:
        original_filename = os.path.basename(original_filepath)

        # Find the specific suffix part *after* the common_jit_id_prefix.
        # This regex looks for the common_jit_id_prefix, then captures everything after it,
        # ensuring it starts with a dot if there's more.
        # Example: if original_filename is 'module_0080.jit_f.cl_747713181.after_codegen.txt'
        # and common_jit_id_prefix is 'jit_f.cl_747713181'
        # we want to capture '.after_codegen.txt'
        suffix_match = re.search(
            re.escape(common_jit_id_prefix) + r"(\..*)", original_filename
        )

        if suffix_match:
            original_suffix_with_extension = suffix_match.group(
                1
            )  # e.g., '.after_codegen.txt'

        new_filename = f"{new_base_name}{original_suffix_with_extension}"
        new_filepath = os.path.join(dest_xla_dump_dir, new_filename)

        if original_filepath == new_filepath:
            print(
                f"Skipping: '{original_filename}' already has the desired name or path."
            )
            continue

        # Copy the renamed files to desired location
        if is_local_directory_path(dest_xla_dump_dir):
            try:
                os.makedirs(dest_xla_dump_dir, exist_ok=True)
                shutil.copy(original_filepath, new_filepath)
            except Exception as e:
                print(
                    f"An unexpected error occurred while copy '{original_filepath}': {e}"
                )
        else:
            upload_to_storage(remote_dir=new_filepath, local_file=original_filepath)
    print(f"The XLA dump is stored in {dest_xla_dump_dir}")
