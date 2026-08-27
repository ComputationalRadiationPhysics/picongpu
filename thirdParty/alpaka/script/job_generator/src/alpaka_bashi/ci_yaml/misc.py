"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Different GitLab CI yaml code snippets, which are not in an extra file.
"""

from typing import Dict, Any
from typeguard import typechecked


@typechecked
def set_misc_job_properties(job_body: Dict[str, Any]):
    """Set different GitLab CI job yaml properties.

    Args:
        job_body (Dict[str, Any]): _description_
    """
    job_body["interruptible"] = True


@typechecked
def get_dummy_job() -> Dict:
    """Return GitLab CI job, which simply prints a message. Can be used, if no job is generated for
    a CI pipeline."""
    return {
        "dummy-job": {
            "image": "alpine:latest",
            "interruptible": True,
            "script": ['echo "This is a dummy job so that the CI does not fail."'],
        }
    }
