"""Add several, special (, handwritten) CI jobs."""

import re
from typing import Dict, Any
from typeguard import typechecked
from .clang_analysis import get_clang_debug_analysis_job, get_clang_asan_job
from .cuda import (
    get_nvcc_relocatable_device_code_job,
    get_nvcc_extended_lambda_off_job,
    get_cuda_only_job,
)


@typechecked
def get_special_jobs(
    container_version: str,
    image_check: bool,
    stage_name: str,
    job_filter: str,
) -> Dict[str, Any]:
    """Return Dict of special CI jobs.

    Args:
        container_version (str): Container version.
        image_check (bool): Check if configured image exist. If not, use fallback.
        stage_name (str): Stage name. If empty, do not create stage property.
        job_filter (str): Filter jobs by job name. If empty, do not filter.

    Returns:
        Dict[str, Any]: Dict of CI jobs.
    """
    special_jobs: Dict[str, Any] = {}

    if stage_name:
        special_jobs["stages"] = [stage_name]

    special_jobs |= get_clang_debug_analysis_job(
        clang_version="14",
        cmake_version="3.25.3",
        container_version=container_version,
        stage_name=stage_name,
        image_check=image_check,
    )
    special_jobs |= get_clang_asan_job(
        clang_version="16",
        cmake_version="3.25.3",
        container_version=container_version,
        stage_name=stage_name,
        image_check=image_check,
    )
    special_jobs |= get_nvcc_relocatable_device_code_job(
        nvcc_version="12.0",
        gcc_version="11",
        cmake_version="3.26.5",
        container_version=container_version,
        stage_name=stage_name,
        image_check=image_check,
    )
    special_jobs |= get_nvcc_extended_lambda_off_job(
        nvcc_version="12.0",
        gcc_version="11",
        cmake_version="3.27.1",
        container_version=container_version,
        stage_name=stage_name,
        image_check=image_check,
    )
    special_jobs |= get_cuda_only_job(
        nvcc_version="12.5",
        gcc_version="13",
        cmake_version="3.30.3",
        container_version=container_version,
        stage_name=stage_name,
        image_check=image_check,
    )

    if job_filter:
        compiled_regex = re.compile(job_filter)
        special_jobs = {
            job_name: job_body
            for job_name, job_body in special_jobs.items()
            if compiled_regex.match(job_name) or job_name == "stages"
        }

    return special_jobs
