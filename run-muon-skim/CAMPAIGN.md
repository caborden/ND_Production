# Unattended FSD Cube production from the completed pilot

`campaign.py` prepares a new campaign, then submits a Slurm dependency chain:

```text
one skim job + validation
    -> afterok -> ten GPU array tasks (each loops over a group of files)
    -> afterok -> five Flow array tasks (each loops over a group of files)
    -> afterok -> one selection and merge job -> summary.json
```

There are four submissions and 17 scheduled tasks, not one scheduled task per
HDF5 file. Validation is part of the stage jobs, so it needs no extra allocation.
An `afterok` dependency on an array requires every task to succeed. Invalid
dependencies are cancelled. Queue time is additional; an overnight completion
is not guaranteed. The login session may close after submission finishes.

The default is **five new original files**, excluding the pilot source. With the
measured 740 tracks per original file this gives about **4,440 tracks including
the pilot**. This is an expected yield, not a stopping rule or guaranteed count.
The program never automatically adds more source files to chase a target.

## Configuration and preservation

This launcher is deliberately configured for the NERSC layout established in
the pilot instructions. It uses the existing `skim.venv`, `larnd.venv`, pinned
`flow-fsdcube-916fb2e1`, and the shared `efield2_rcmanduj` analysis environment.
It invokes the same Python CLIs as the pilot, rather than the generic production
Flow wrapper (which uses other workflows). There is no notebook dependency.

- Copy the exact skim configuration from the completed pilot manifest.
- Check skim source hashes and geometry/configuration registry against the pilot.
- Check larnd-sim's Git commit against `muon-skim-jobs/larnd-sim.commit.txt`.
- Require Flow revision `916fb2e1f023f95206632f4a839ab923c6561751`.
- Snapshot skim and selection Python code; validate selector/helper hashes against
  every pilot per-file track output, including its strict >29 cm cut and FSD mode.
- Record and guard current larnd-sim/Flow Git HEAD and tracked diff, source-file
  identities, and external response/pedestal file size and mtime. This detects
  changes during the campaign, but does not retroactively prove every installed
  dependency or external calibration file is identical to the pilot.
- Reuse installed environments. Do not update those environments or checkouts
  during the campaign. The launcher does not provision new dependencies.
- Use the same skim seed. New larnd-sim seeds start after the pilot file count
  (41 for this pilot), avoiding reuse of its 1–40 seed sequence.
- Keep the pilot files unchanged. Merge its **per-input** track files with new
  per-input files; never feed an already merged file into the merger.
- Preserve the original 90-hit and track-quality cuts and prompt-hit selection.

No `--mem` is requested. CPU jobs request two CPU slots. GPU tasks request one
GPU and 32 CPU slots. Limits are two hours for skimming, six for each GPU worker,
three for each Flow worker, and three for selection/merging. These are limits,
not estimates. Worker counts are configurable up to ten GPUs and five Flow jobs.

Submission and worker lock files live in `<campaign outputs>/locks/` on scratch,
not in the CFS campaign directory. CFS access through DVS on compute nodes does
not support `flock`. Preparation creates the shared scratch lock directory;
submission/workers also create it if needed. Do not delete lock files while a
campaign is active. `HDF5_USE_FILE_LOCKING=FALSE` controls HDF5 only and does not
replace these launcher locks.

For a campaign prepared with the old CFS lock paths, pull the fix and prepare a
**new campaign name** (for example `FSDCubeMuons_4500_v2`). The prepared campaign
contains a frozen copy of the launcher: pulling the repository does not update
that copy. Preserve the old directory for diagnosis and make sure its jobs have
terminated before starting the replacement. The new campaign automatically
creates both its CFS metadata directory and scratch output/lock directories.

## Prepare and submit on NERSC

Copy `campaign.py` to `ND_Production-muon-skim/run-muon-skim/` if it has not
arrived via Git. Preparation does not submit jobs or modify original inputs.
Run from a fresh terminal:

```bash
source /global/cfs/cdirs/dune/users/cborden/muon-skim-jobs/environment.sh
export CAMPAIGN="$JOB_ROOT/FSDCubeMuons_4500_v1"

"$ND_PRODUCTION_MUON_SKIM_PYTHON" \
    "$PRODUCTION_REPO/run-muon-skim/campaign.py" prepare \
    --name FSDCubeMuons_4500_v1 --additional-sources 5
```

Preparation prints the five selected source paths and expected combined yield.
It scans the original sample directory directly; the pilot-only
`all_sources.txt` is not used or overwritten. Symlink aliases are deduplicated;
the skim gate also rejects identical source content, including the pilot.
The prepared `campaign.json` is protected by a checksum; don't edit it in place.
Use a new campaign name to change settings. A failed preparation may leave a
directory for inspection; use a new name after addressing the problem.

After preparation succeeds:

```bash
"$ND_PRODUCTION_MUON_SKIM_PYTHON" "$CAMPAIGN/campaign.py" \
    submit "$CAMPAIGN/campaign.json"
```

The first job is held until all dependencies are registered, then released.
Submission failures cancel jobs already registered by that attempt. IDs are
saved after each successful `sbatch` in `submission.json`. Duplicate submission
is refused. No email or other external message is sent.

## Validation and completion

Skim checks require complete manifests, matching configuration and event counts,
unique source fingerprints, and readable required HDF5 datasets. An entirely
empty new sample stops the chain. GPU checks require packets, truth tables, and
one MC association row per packet. Flow checks require event/hit/timing/truth
tables, key event references, and finite geometry bounds. These are structural
checks, not a full physics validation of charge calibration or truth alignment.

GPU and Flow outputs publish only after successful subprocess exit and
validation, with per-file completion receipts. Flow globally validates every
GPU output before processing. Selection globally validates all Flow outputs,
uses the existing resumable batch runner, checks the cut and source hashes, and
requires at least one new accepted track. The final merge verifies track and
source counts. A result below 4,500 is still complete: the requested campaign is
five new source files, with approximately 4,440 tracks expected including pilot.

```bash
squeue -u "$USER"
cat "$CAMPAIGN/submission.json"
ls -lt "$CAMPAIGN/logs"
cat "$CAMPAIGN/summary.json"   # created only after successful final validation
```

The default final output is:

```text
$SCRATCH/fsd-muon-production/campaigns/FSDCubeMuons_4500_v1/tracks/FSDCubeMuons_4500_v1.tracks.hdf5
```

The final summary records pilot/new/total track counts and the output path.
Campaign metadata and logs live on CFS; large intermediate/final HDF5 files live
on scratch and need the user's normal retention/archival procedure.

## Explicit retry after failure

Inspect the failed stage and per-file logs first. Downstream stages do not run
after a failed gate. There are no automatic retry loops. Once **all jobs from
the preceding attempt are terminal**, repeat:

```bash
"$ND_PRODUCTION_MUON_SKIM_PYTHON" "$CAMPAIGN/campaign.py" \
    submit "$CAMPAIGN/campaign.json" --retry
```

This creates fresh dependencies (a failed Slurm dependency cannot be revived by
merely requeuing its predecessor). Complete skims and receipted GPU/Flow outputs
are checked and reused. Incomplete skim directories/partial HDF5 files are
renamed with `.failed-<timestamp>` and regenerated. A final GPU/Flow file without
a receipt is preserved and regenerated, never accepted based on existence.
An existing mismatched receipt fails closed. Track selection and merging use
their existing `resume` checks. Prior successes remain intact.

Retries refuse to proceed while earlier campaign jobs are queued or running.
Never run individual worker commands concurrently with scheduler-managed work.
If a process is forcibly killed during submission, use the persisted job ledger
to inspect/cancel its held jobs before retrying. A new software configuration
requires a new campaign rather than bypassing the immutability checks.

## Local validation

```bash
python -m unittest discover -s run-muon-skim/tests -v
```

Campaign tests mock Slurm/GPU execution and exercise dependency registration,
failed submissions, balanced groups, preparation/snapshots, validation gates,
changed inputs, completion receipts, retries, and the real CPU skim. Real GPU
simulation and NERSC scheduler execution must be verified on NERSC.
