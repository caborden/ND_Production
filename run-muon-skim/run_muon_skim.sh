#!/usr/bin/env bash
# One convert2h5 source per production index; each job owns its output directory.
set -eo pipefail
# Shared environment helpers intentionally allow unset optional variables.
export ND_PRODUCTION_RUNTIME=${ND_PRODUCTION_RUNTIME:-NONE}
source ../util/reload_in_container.inc.sh
: "${ND_PRODUCTION_INDEX:?Set the source production index}"
: "${ND_PRODUCTION_OUT_NAME:?Set the skim production name}"
source ../util/init.inc.sh
set -u

: "${ND_PRODUCTION_MUON_SKIM_LARND_REPO:?Set the larnd-sim checkout used for geometry}"
if [[ -n "${ND_PRODUCTION_MUON_SKIM_INPUT_FILE:-}" ]]; then
    input=$ND_PRODUCTION_MUON_SKIM_INPUT_FILE
else
    : "${ND_PRODUCTION_CONVERT2H5_NAME:?Set the source convert2h5 production name}"
    input=$ND_PRODUCTION_OUTDIR_BASE/run-convert2h5/$ND_PRODUCTION_CONVERT2H5_NAME/EDEPSIM_H5/$subDir/$ND_PRODUCTION_CONVERT2H5_NAME.$globalIdx.EDEPSIM.hdf5
fi
destination=$outDir/EDEPSIM_H5/$subDir/$outName
config=${ND_PRODUCTION_MUON_SKIM_CONFIG:-$PWD/configs/fsd_cube.json}
python=${ND_PRODUCTION_MUON_SKIM_PYTHON:-python3}

# Keep the configuration seed fixed across jobs: event identity provides separation.
run "$python" -m muon_skim "$input" --config "$config" \
    --larnd-repo "$ND_PRODUCTION_MUON_SKIM_LARND_REPO" \
    --output-dir "$destination" --prefix "$outName"
