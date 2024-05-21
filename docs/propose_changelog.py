#!/usr/bin/env python
#
# Copyright 2024 Julian J. Lenz
#
# License: GPLv3+
#
# requirements:
#   PyGithub
#   pyyaml
#
"""
propose_changelog.py

This little tool queries the Github API for merged pull requests corresponding to the given milestone and labelled by the label "changelog".
The obtained list is categorised and printed to stdout. Suggested usage is:

```bash
$ MILESTONE="0.8.0 / Next stable"  # or whatever version you're interested in
$ GH_PTA="<your Github personal access token>"
$ python propose_changelog.py "$MILESTONE" $GH_PTA > changelog.txt
# edit `changelog.txt` according to your needs
```

For a typical end user running this once or twice a day, the environment variable "GH_PTA" can be empty.
If you are running this frequently, e.g., during a debug session, you might want to
[acquire a personal acces token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
and store it in this environment variable. The most restricted one with public repository read access is sufficient.

For adjustments to the categorisation, you can simply change the global variable `CATEGORIES`.
"""

from github import Github
import sys
import yaml


def make_lambda(main_condition, component=None):
    """Factory for lambdas encoding the categorisation conditions. Only needed to capture the current iteration by value."""
    if component:
        return lambda pr: contains_label(pr, f"component: {component}") and main_condition(pr)

    # Okay, admittedly this is just a lazy way to sneak the 'other' category into this interface:
    # A careful review would probably request to split this function up and give both of them better names.
    return lambda pr: all(
        not contains_label(pr, f"component: {component}") for component in COMPONENTS.values()
    ) and main_condition(pr)


# map the changelog naming to the tag naming, e.g., "component: core", etc.
COMPONENTS = {"PIC": "core", "PMacc": "PMacc", "plugins": "plugin", "tools": "tools"}

# describe how to detect the main categories
MAIN_CATEGORIES = {
    "Features": lambda pr: not contains_label(pr, "bug") and not contains_label(pr, "refactoring"),
    "Bug Fixes": lambda pr: contains_label(pr, "bug") and contains_label(pr, "affects latest release"),
    "Refactoring": lambda pr: not contains_label(pr, "bug") and contains_label(pr, "refactoring"),
    "Documentation": lambda pr: contains_label(pr, "documentation"),
}

# This is the main configuration point: The changelog will have the same tree structure as this nested dict. The leaves
# however will be replaced with something roughly equivalent to `list(filter(func, PRs))` if `func` is the corresponding
# function constituting a leave of `CATEGORIES`. In order to save some typing, the bulk of the categorisation is
# formulated as a cartesion product of `MAIN_CATEGORIES` and `COMPONENTS` but this is only for convenience. Feel free
# to change this or amend this by hand afterwards.
CATEGORIES = {
    "User Input Changes": (lambda pr: contains_label(pr, "component: user input")),
} | {
    main_cat: {
        # This is important: If you create the lambda in-place, variables are captured by reference and the
        # corresponding value is only fetched upon a call. This leads to all lambdas using the last value of the
        # iteration.
        name: make_lambda(main_condition, component)
        for name, component in COMPONENTS.items()
    }
    | {"other": make_lambda(main_condition)}
    for main_cat, main_condition in MAIN_CATEGORIES.items()
}


def contains_label(issue, label):
    """Helper function to check if an issue is labelled by label."""
    return label in map(lambda lab: lab.name, issue.labels)


def categorise(prs, categories_or_condition):
    """Recursively run over the given categories and filter the PRs according to their conditions."""
    if not isinstance(categories_or_condition, dict):
        return list(filter(categories_or_condition, prs))
    return {key: categorise(prs, val) for key, val in categories_or_condition.items()}


def apply_to_leaves(func, mapping):
    """Helper function to recursively apply to the leaves of a nested dictionary (applying to values of a list individually)."""
    if isinstance(mapping, dict):
        return {key: apply_to_leaves(func, val) for key, val in mapping.items()}
    if isinstance(mapping, list):
        return list(map(func, mapping))
    return func(mapping)


def to_string(categories):
    """Transform our nested dictionary of GH PR objects into something readable."""
    return yaml.dump(apply_to_leaves(lambda pr: f"{pr.title} #{pr.number}", categories))


def pull_requests(gh_key, version):
    """Query the Github API for the kind of PRs we need."""
    return Github(gh_key).search_issues(
        f'repo:ComputationalRadiationPhysics/picongpu type:pr is:merged milestone:"{version}" label:changelog'
    )


def main(version, gh_key=None):
    """Main logic: Download, categorise, print."""
    print(to_string(categorise(pull_requests(gh_key, version), CATEGORIES)))


if __name__ == "__main__":
    main(*sys.argv[1:])
