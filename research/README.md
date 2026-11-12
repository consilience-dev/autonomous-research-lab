# Research record and preservation

Start with the [project overview](../START_HERE.md). The [machine-readable registry](index.json) gives every experiment a permanent ID, owner-side record, configuration checksum, timestamps, resource totals, and one or more SHA-256 manifests. The [timeline](../TIMELINE.md), [findings](../FINDINGS.md), and [decision log](../DECISIONS.md) provide the research narrative.

## Register and document the next experiment

Run from the repository root with Python 3.9 or later. Use a repository-relative configuration path that contains no credentials or private seed text. For protected evaluation evidence, use opaque local filenames and keep the protected contents outside the editing workspace.

```sh
python3 -B research/registry.py register --kind run \
  --title "Short descriptive title" --config owner/path/to/config.json \
  --artifact owner/path/to/frozen-prompt.txt
```

Use `--kind eval` for an evaluation and `--parent RUN-001` or the relevant upstream ID to link it to a run. The registry assigns the next ID for that kind, timestamps registration, hashes the initial files, and creates a record from the [template](TEMPLATE.md). Record actual timestamps as ISO 8601 values with a timezone offset when available. Missing times stay null in the index and are explained in the record. Register from a single shared branch or checkout so independently created branches do not allocate the same number.

When the run ends, retain completed and failed attempts, then seal it with its permanent artifact paths. For work already completed, registration also accepts the outcome status, the actual start and end, and the full initial set of artifacts.

```sh
python3 -B research/registry.py seal RUN-002 --status completed \
  --started "$actual_start" --ended "$actual_end" \
  --artifact owner/path/to/run-002 \
  --resources '{"lab_passes_completed":30}'
python3 -B research/registry.py verify
```

The verifier rehashes referenced artifacts, validates the manifest hashes and record relationships, and checks local Markdown link targets. It checks the captured files even when additional files are later added to an artifact root. The two historical inventories cover both original storage and existing owner snapshots. Local runtimes are rebuildable using the dependency locks. The project brief remains an owner-controlled chat attachment outside the repository.

For each new experiment, fill in the generated record and append a milestone to [TIMELINE.md](../TIMELINE.md). Append only warranted observations or dated corrections to [FINDINGS.md](../FINDINGS.md), and record methodological changes in [DECISIONS.md](../DECISIONS.md). A new intervention or evaluation gets a new ID. A rerun that only recovers failed samples can remain an explicitly identified attempt under the original evaluation ID. Preserve both raw logs. Amend interpretations with a dated note and evidence while retaining the prior conclusion. Keep manifests and frozen configuration snapshots intact.

## Git and backups

Git is for source code, the dependency locks, safe reusable configuration, documentation, and checksummed references. The repository originally had no commits or configured remote. The ignore rules exclude local runtimes, weights, agent work, owner logs, Petri seeds and runs, and common credential files. The Petri source, dependency lock, and reusable scoring rubric are allowlisted. The live Petri run configuration contains study-specific paths and stays with its owner archive. Inspect the staged file list before committing and leave cloud publication to an owner-approved private remote.

The evidence remains at its existing local paths. Git cloning restores the documentation and code. Artifact verification requires restoring the matching local evidence. For an independent backup, copy the manifest roots to an encrypted owner-controlled volume or private artifact store with the same repository-relative layout. Preserve the native logs and the content-addressed owner snapshot as well as the checkpoint and patch lineage. Use the manifests to verify the copy, record its date and destination in an owner-only backup log, and periodically test restoration into a separate checkout. An off-device backup was not created during this documentation pass.

Before reusing `agent/work/`, copy RUN-001's work tree to the owner backup and verify it against the recorded manifest. Keep the original path mapping for restoration. For future runs, give the owner archive its own stable path before sealing and keep later writable workspaces separate. Store any future untouched evaluation inputs under owner-only storage with opaque names. Give an editing agent only its task workspace and sanitized handoff. Keep `START_HERE.md`, the registry, and evidence links outside the editing workspace. The checked-in ignore rules govern Git inclusion, while the [Seatbelt profile](../owner/sandbox-flags.zsh) governs editing-agent access.
