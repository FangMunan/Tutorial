# Legacy MaskGIT / SNN_Transform Gate D7–D9 archive

Status: **archived mechanistic evidence; not active development**.

This directory contains exact Git-blob copies of the D7–D9 protocols, implementations/compatibility launchers, and task-specific GitHub Actions workflow sources that previously lived in the active tree.

## Start here

- Scientific conclusions: `MASKGIT_LEGACY_D7_D9_SUMMARY.md`
- File-level inventory and dependency decision: `provenance/INVENTORY.md`
- Exact source-to-archive Git blob mapping: `provenance/ARCHIVE_MANIFEST.json`
- Historical run/artifact references: `provenance/RUNS_AND_ARTIFACTS.md`

## Archive policy

The archived source files are retained for provenance and reproducibility of the old forensic/mechanism-audit line. They may import shared D3/D6/R6B modules that remain in the active tree; that does not make the old gates active again.

No large GitHub Actions binary artifacts are copied into the repository. Historical commits, Actions run history, artifacts, tags and remote history are untouched.

D9A.4 was not present on the audited branch and is therefore not represented here.
