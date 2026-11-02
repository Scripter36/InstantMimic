"""Checkout-local assistant workflow registration and reference integrity."""

from __future__ import annotations

import json
import re
import shutil

from path_helpers import repo_root

PLUGIN = repo_root() / "plugins/instantmimic-workflows"
SKILLS = {
    "instantmimic",
    "train",
    "smoke-run",
    "play-best",
    "config-variant-check",
    "experiment-report",
    "nsys-profile",
    "ncu-profile",
    "tune-hyperparameters",
    "migrate-external-env",
    "review-external-env-migration",
    "optimize-runtime",
}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def test_workflow_bundle_has_the_documented_skill_surface():
    assert {p.parent.name for p in (PLUGIN / "skills").glob("*/SKILL.md")} == SKILLS
    assert not any(p.is_symlink() for p in PLUGIN.rglob("*"))
    assert (PLUGIN / "agents/migration-reviewer.md").is_file()


def test_workflow_marketplaces_and_manifests_resolve_to_this_checkout():
    versions = []
    for name in (".codex-plugin", ".claude-plugin"):
        manifest = json.loads((PLUGIN / name / "plugin.json").read_text())
        assert manifest["name"] == PLUGIN.name
        assert manifest["skills"] == "./skills/"
        versions.append(manifest["version"].split("+", 1)[0])
    assert versions[0] == versions[1]
    for name in (".agents/plugins/marketplace.json", ".claude-plugin/marketplace.json"):
        marketplace = PLUGIN.parent / name
        entries = json.loads(marketplace.read_text())["plugins"]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["name"] == PLUGIN.name
        source = entry["source"]
        path = source["path"] if isinstance(source, dict) else source
        assert (PLUGIN.parent / path).resolve() == PLUGIN.resolve()


def test_workflow_local_documentation_links_stay_inside_the_checkout():
    for markdown in PLUGIN.rglob("*.md"):
        for raw in MARKDOWN_LINK.findall(markdown.read_text()):
            target = raw.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith("#") or "://" in target:
                continue
            path = (markdown.parent / target.split("#", 1)[0]).resolve()
            assert path.is_relative_to(repo_root()), (markdown, target)
            assert path.is_file(), (markdown, target)


def test_workflow_cached_documentation_links_are_self_contained(tmp_path):
    cache = tmp_path / "plugin-cache"
    shutil.copytree(PLUGIN, cache, ignore=shutil.ignore_patterns("__pycache__"))
    for markdown in cache.rglob("*.md"):
        for raw in MARKDOWN_LINK.findall(markdown.read_text()):
            target = raw.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith("#") or "://" in target:
                continue
            path = (markdown.parent / target.split("#", 1)[0]).resolve()
            assert path.is_relative_to(cache), (markdown, target)
            assert path.is_file(), (markdown, target)
