# FSD Cube memory diagnostics

These tools are independent of the frozen campaign. They do not modify its
simulator, configuration, receipts, or outputs. Install display dependencies in
a separate CPU environment; use the existing larnd environment for reproduction.

## Reproduce input 62

`reproduce_larnd.py --campaign CAMPAIGN_DIRECTORY --index 62 --output-dir NEW_DIRECTORY`
checks the frozen campaign checksum and guards, reads `files.json`, and runs the
installed simulator script with the campaign arguments (fsd_cube, seed index+1,
compression None, original input, and working directory). The output directory
must not exist. The Slurm wrapper accepts the absolute Python script path,
campaign directory, and new output directory as its three positional arguments.
Its default working root is Casey's NERSC directory; edit for other users.

The installed source is compiled with just one AST call replacement: the RHS
`signals = cp.zeros(...)` calls an observer which forwards its arguments unchanged
to CuPy. No source on disk is patched. The observer logs before, after allocation,
and on failure. It uses a weak reference to the previous tensor, avoiding both
strong array references and frame-local caches that could alter allocation life
cycles. It neither frees memory nor changes batching or physics. Instrumentation
adds small timing overhead; this is a diagnostic reproduction, not a bitwise
reproducibility guarantee.

`run.json` records the command, input, simulator and observer hashes, campaign
checksum, guarded repository states, CuPy/Python versions, device properties and
CUDA/Slurm environment. `allocations.jsonl` records byte counts, tensor dimensions,
event/module/pixel range, time span/sampling, previous live signals size, CuPy
live/cached memory and CUDA free/total memory. Pool limit 0 means no explicit pool
limit. CUDA free memory excludes pool reservations; cached bytes can potentially
be reused. Do not simply add requested bytes to pool total to infer the minimum
GPU capacity. Previous tensor overlap is a hypothesis to test, not proof that
freeing it alone suffices.

A failed run leaves its partial diagnostic output. A success creates
`SIMULATION_RETURNED_SUCCESSFULLY`, which is NOT a campaign validation receipt.
Never copy diagnostic HDF5 outputs into production. Compare the actual failing
event and allocation with the original failure before choosing a fix.

## Notebook

Open `inspect_events.ipynb` with the diagnostics directory as the notebook working
directory. `requirements.txt` supplies a separate kernel's dependencies. No GPU,
larnd-sim import, or original convert2h5 file is needed: the translated skim
contains segments and its own geometry/provenance. Optional truth trajectories
and vertices are extracted too, but the display draws deposit segment endpoints.
Edit FILE_INDEX and EVENT_ID, then run cells. Default target: file 62, event 17;
control: file 61, event 14. Toggle ACTIVE_ONLY and DETECTOR_VIEW and rerun the plot.

Units are cm in EDepSim coordinates (x drift, y vertical, z transverse), before
larnd-sim's x/z swap. The endpoint mask matches the geometry test only, not all
simulation filters. Full-event view reveals outside activity; detector view uses
fixed geometry limits for comparisons. Hover shows IDs, dE and times when present
(in native stored units); legend clicks toggle particle species. Provenance retains
original event/source/translation and ID maps; original and remapped trajectory
IDs should not be joined without those maps.

Export cells optionally save one-event HDF5 bundles (exclusive creation) or
self-contained interactive HTML. Set BUNDLE to view a bundle elsewhere and skip
the control cell if the campaign isn't accessible. This is truth visualization,
not Flow reconstructed-hit visualization.

## Local checks

From this directory: `python -m unittest -v test_diagnostics.py`.
Tests cover the observer's unchanged allocation arguments, old-tensor lifetime,
error propagation, event extraction, strict endpoint mask, optional datasets,
bundle round-trip/no overwrite, Plotly serialization, and notebook syntax/schema.
Actual GPU measurements and notebook rendering on NERSC remain runtime checks.
