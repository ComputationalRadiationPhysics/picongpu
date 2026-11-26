"""Sphinx extension: resolve repository-root-relative links in dual-rendered files.

Some files in this repository are both meant to be read on disk and included in
the manual. ``CONTRIBUTING.md`` is the main example: it lives at the repository
root, but is also rendered as ``dev/CONTRIBUTING.html`` through the symlink
``docs/source/dev/CONTRIBUTING.md``.

Links in such a file must make sense when clicked on disk, i.e. they are
relative to the repository root (for example ``docs/source/dev/styleguide.rst``).
When the same text is rendered from ``docs/source/dev/``, MyST resolves the
target relative to the *source* directory, so the repository-root-relative path
does not exist and the link would be broken.

This extension closes that gap at build time without touching the source text:
for every unresolved MyST link it tries to interpret the target as a path
relative to the repository root, looks the resulting file up in the set of
Sphinx documents, and rewrites the reference to point at that document. Sphinx
then emits the correct relative URL for the current builder.

The lookup is by real path, so symlinked doc sources (``docs/source/dev/docs/
COMMIT.md`` -> ``docs/COMMIT.md``) map back to the document they are included
as. External URLs and targets that do not map to a document are left untouched,
so they keep their original behaviour instead of breaking the build.
"""

import os
from pathlib import Path
from urllib.parse import urlsplit

from sphinx import addnodes

# ``docs/source/_ext/repo_relative_links.py`` -> repository root
REPO_ROOT = Path(__file__).resolve().parents[3]


def _source_suffixes(config):
    """Return the configured source suffixes as a plain list."""
    suffixes = config.source_suffix
    if isinstance(suffixes, dict):
        return list(suffixes)
    if isinstance(suffixes, str):
        return [suffixes]
    return list(suffixes)


def _docname_by_realpath(app):
    """Map each document's real source path to its Sphinx docname.

    Resolving both the document source and the link target through
    ``os.path.realpath`` makes symlinked sources transparent: a link written
    for ``docs/COMMIT.md`` resolves to the docname that includes that file.
    """
    srcdir = Path(app.srcdir)
    lookup = {}
    for docname in app.env.found_docs:
        for suffix in _source_suffixes(app.config):
            source = srcdir / (docname + suffix)
            if source.is_file():
                lookup[os.path.realpath(source)] = docname
                break
    return lookup


def _is_external(target):
    """Reject URLs, anchors and absolute filesystem paths."""
    if target.startswith(("#", "/", "mailto:")):
        return True
    parsed = urlsplit(target)
    return bool(parsed.scheme or parsed.netloc)


def _rewrite_repo_links(app, doctree):
    """Point repo-root-relative MyST links at the matching Sphinx document.

    Only unresolved MyST references are considered (``refdomain`` is ``None``).
    A link that MyST already resolved relative to the source document keeps its
    normal behaviour. This is exactly the set of links that does not exist from
    the source directory but may exist from the repository root.
    """
    lookup = _docname_by_realpath(app)
    for node in doctree.findall(addnodes.pending_xref):
        if node.get("reftype") != "myst" or node.get("refdomain") is not None:
            continue
        target = node.get("reftarget")
        if not target or _is_external(target):
            continue
        path_target, _, fragment = target.partition("#")
        candidate = REPO_ROOT / path_target
        if not candidate.is_file():
            continue
        docname = lookup.get(os.path.realpath(candidate))
        if docname is None:
            continue
        # Hand the reference back to Sphinx' document resolver, which builds the
        # correct relative URL for the active builder.
        node["refdomain"] = "doc"
        node["reftarget"] = docname
        node["reftargetid"] = fragment or None


def setup(app):
    app.connect("doctree-read", _rewrite_repo_links)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
