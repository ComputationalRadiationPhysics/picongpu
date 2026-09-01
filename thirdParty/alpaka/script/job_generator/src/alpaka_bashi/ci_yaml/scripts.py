"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Set the script section of the GitLab CI test job yaml.
"""

from typing import Dict, Any
from typeguard import typechecked


@typechecked
def set_script(job_body: Dict[str, Any]):
    """Set the job section of a job. Overwrite an existing job section."""
    job_body["script"] = [
        "source ./script/set_default_env_vars.sh",
        "source ./script/gitlabci/print_env.sh",
        "source ./script/gitlab_ci_run.sh",
    ]
