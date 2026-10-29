# FSD Cube muon skim

This CPU stage reads **unoverlaid cosmic convert2h5 output**, finds muon tracks
with useful length inside the Cube after translation, and writes whole translated
events for larnd-sim. It reuses Geant4 deposits; it does not regenerate transport.
The importable `muon_skim` package contains all production logic. The inspection
notebook imports that package and is not required to run production.

## Selection contract

- Configuration schema 1 defaults to a strict **length >29 cm**, seed 20260527,
  and at most 30 selected events per file.
- Candidate shifts use the shower notebook's 2×10×2 grid and positive xyz steps
  `[47.1085, 29.96, 47.8318]` cm. The destination bounds come from the selected
  larnd-sim YAML, not the old notebook's destination window. Grid windows can
  overlap or leave gaps; selection samples this finite list, not all translations.
- The current `fsd_cube` layout gives edep-sim bounds x=[0.3175,48.0675],
  y=[119.04,148.8], z=[0,47.616] cm. This implementation supports one tile/TPC.
- Select segment groups with `abs(pdg_id)==13`, grouped by `file_traj_id` within
  each source event. Clip each segment geometrically to the target active box.
- For each group, compute the PCA axis of clipped segment endpoints. The distance
  between the endpoints at the two projection extremes is the **chord**. Also sum
  clipped segment lengths to obtain the **path**. Both must strictly exceed the
  configured cut. There is no charge/dE cut, primary-only requirement, continuity
  cut, or shower veto. Disconnected deposits and curved trajectories can still
  pass; this is a loose truth preselection.
- Choose uniformly among qualifying **translations**, regardless of how many
  muons each contains. Write one copy of the source event, retaining all particles,
  vertices, descendants, and out-of-volume deposits. The saved `muons` list records
  every qualifying muon at the chosen placement.
- The deterministic PCG64 seed derives from SHA256 of `[configured seed, source
  file SHA256, original event ID]`. Packing, source ordering, and file renaming do
  not change a choice. Changing file contents does. NumPy's version is recorded.
  Uniformity is over the configured candidates, not over positions or exposure.

The downstream `dune-nd-mip-calib` selector now defaults to >29 cm too and accepts
`--min-length-cm`. Its length is still the existing endpoint-defined reconstructed
length, not this PCA truth chord. The 90-hit and shape cuts remain. A 29.76 cm
vertical truth crossing is eligible here but is not guaranteed to reconstruct or
pass selection. For an old 55 cm analysis, pass `--min-length-cm 55` explicitly.

## Standalone command

Run from `ND_Production/run-muon-skim` using a Python environment with
`requirements.txt` installed. No CUDA, ROOT, h5flow, or notebook kernel is needed.

```bash
python -m muon_skim \
  --input-glob '/path/to/convert2h5/**/*.EDEPSIM.hdf5' \
  --larnd-repo /path/to/larnd-sim \
  --config configs/fsd_cube.json \
  --output-dir /path/to/new-muon-skim
```

Files/directories can also be positional arguments. Directories are recursively
searched for `*.hdf5`. `--input-list` accepts paths relative to the list's directory
or absolute paths, ignoring blank lines and `#` comment lines. Discovery resolves
symlinks, sorts paths, and rejects duplicates, missing files, and unmatched globs.
Use an input-only directory: arbitrary HDF5 files are not silently skipped.

Override the effective configuration with `--min-length-cm`, `--seed`, or
`--events-per-file`. The manifest records both effective settings and the original
configuration-file digest. `--prefix` changes output file basenames. The output directory
must be new; no existing run is overwritten.

The default output names are `muons.0000000.EDEPSIM.hdf5`, etc. Each contains
at most 30 events (configurable). A change in input compound schemas or optional
table availability starts a new output file. Inputs are loaded one file at a time;
event indexes avoid repeatedly scanning full tables. Memory still scales with
the size of one source file, so this is not yet a streaming large-file reader.

## Truth and provenance

Required tables are `segments`, `trajectories`, and `vertices`. `mc_stack` and
`mc_hdr` are optional; empty generator tables are omitted. Unknown root tables
are rejected to avoid silently losing truth. Input is expected to use the current
convert2h5 field names; incompatible schemas fail with an error.

Output event IDs are contiguous from zero within each output file. `file_traj_id`,
`segment_id`, and `vertex_id` are remapped uniquely within an output file. `traj_id` is
event-local and contiguous. Parents resolve using the original `(vertex_id,
traj_id)` pair, so repeated local IDs across vertices work. Unmatched negative
or unsigned-maximum generator trajectory IDs and parent sentinels are preserved.
Missing/ambiguous references and integer overflow fail rather than wrap.

Segment centers/endpoints, trajectory endpoints, and vertex coordinates in
`vertices`/`mc_hdr` are translated. Momenta, deposited energies, and time fields
remain unchanged. Use event-relative cosmic times; spill overlays are unsupported.
Geometrical clipping is only used for selection: written deposits are never
cropped or have their dE rescaled.

Each output contains:

- `provenance/events`: JSON records with source/output event IDs, the translation,
  number of qualifying placements, selected muon metrics, and old/new ID maps.
- `provenance` attribute `id_map_columns_json`: column meanings for all maps.
- `provenance/sources/<index>`: source file path/hash/size/mtime, original root
  attributes, and dataset attributes. These attributes are preserved here instead
  of choosing one source's attributes as authoritative for a mixed output.
- `provenance` attribute `run_json`: effective configuration, coordinate convention,
  actual bounds, candidate shifts, RNG policy, package-code hashes, geometry YAML
  hashes, and Python/library versions.

Source indices refer to the run manifest, and output IDs are scoped to each output file.
The trajectory maps plus source identifiers allow ancestry to be traced back to
the original files; source files remain the authority for original parent values.

## Publication and empty selections

`manifest.json` records scanned/selected counts, source fingerprints, outputs, and
completion/failure status. An output file is first written as `.partial` and exclusively
published with a hard link after completion; its final name cannot overwrite a file.
The partial and final file live in the same directory/filesystem. The filesystem
must support hard links. Failed runs retain diagnostics and any partial files.
There is no automatic resume in this first implementation; use a new output
directory after correcting a failure. Do not submit a run whose manifest failed.

Only successful runs write `larnd_inputs.txt`, one absolute output file path per line.
Zero selections is a successful outcome: the manifest reports zero, the list is
empty, and no empty HDF5 file is sent to larnd-sim.

## ND_Production integration

The stage wrapper follows the existing production-index conventions. For example,
from `run-muon-skim`, with the output/log roots and installation environment set:

```bash
export ND_PRODUCTION_RUNTIME=NONE
export ND_PRODUCTION_INDEX=0
export ND_PRODUCTION_OUT_NAME=FSDCubeMuons_v1
export ND_PRODUCTION_CONVERT2H5_NAME=FSD_CosmicRun4.convert2h5
export ND_PRODUCTION_MUON_SKIM_LARND_REPO=/path/to/larnd-sim
bash run_muon_skim.sh
```

`ND_PRODUCTION_MUON_SKIM_PYTHON` selects the interpreter, and
`ND_PRODUCTION_MUON_SKIM_CONFIG` selects the configuration. The configuration seed
stays fixed across jobs. `ND_PRODUCTION_MUON_SKIM_INPUT_FILE` optionally overrides
the inferred source filename. The wrapper creates a unique output directory:

```text
run-muon-skim/<OUT_NAME>/EDEPSIM_H5/<subDir>/<OUT_NAME>.<globalIdx>/
```

Each source job may produce several output files or none. After all skim jobs complete,
collect their `larnd_inputs.txt` entries into a deterministic job list. Submit one
larnd-sim job per entry with **a unique downstream production index**:

```bash
export ND_PRODUCTION_LARNDSIM_CONFIG=fsd_cube
export ND_PRODUCTION_LARNDSIM_INPUT_FILE=/absolute/path/to/selected-output-file.EDEPSIM.hdf5
# Set a unique ND_PRODUCTION_INDEX and the desired ND_PRODUCTION_OUT_NAME.
# Then, from run-larnd-sim:
bash run_larnd_sim.sh
```

The larnd-sim wrapper retains its normal convert2h5 input convention when the new
explicit input variable is absent. This patch does not submit jobs, alter Flow
workflows, or choose the production electronics/reset settings. Pin those settings
to the intended data comparison and validate the Cube Flow setup in a pilot.

## Inspection and validation

`inspect_muon_skim.ipynb` loads the same configuration, geometry, candidate finder,
and seeded chooser as the CLI. Set a source file to preview one event, or set a
completed skim directory to inspect provenance. It does not write production data.
Matplotlib is an additional dependency for its optional preview plot.

```bash
python -m unittest discover -s tests -v
```

The synthetic tests cover clipping, strict length cuts, muon identity, seeded
placement, truth remapping/sentinels, optional tables, packing, empty runs,
provenance, errors, discovery, and CLI overrides. They do not establish physical
acceptance or detector-simulation throughput. Validate a real pilot through
larnd-sim, Flow, and track selection before scaling up.

Translations preserve Geant4 history, including original material interactions;
they do not reproduce transport through a new Cube geometry. Absolute rates and
independence also require accounting for any reuse already present upstream.
The current larnd-sim active-volume selector requires at least one segment endpoint
inside the volume, whereas this skim computes exact intersections. Very long
segments crossing with both endpoints outside can qualify here but be discarded
there; normal short Geant4 steps and pilot truth-to-hit checks should be verified.
