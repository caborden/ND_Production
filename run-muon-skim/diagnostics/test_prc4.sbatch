#!/bin/bash
#SBATCH --job-name=prc4-file62
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
set -eo pipefail
work=/global/cfs/cdirs/dune/users/cborden
campaign=$1
run=$2
diag="$work/ND_Production-muon-skim/run-muon-skim/diagnostics"
module load python/3.11
module load cudatoolkit/12.4
source "$work/larnd-wFFE.venv/bin/activate"
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
start=$SECONDS
env -u SLURM_TRES_PER_TASK srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python "$diag/reproduce_larnd.py" --campaign "$work/muon-skim-jobs/FSDCubeMuons_4500_v2" \
    --settings-campaign "$campaign" --index 62 --output-dir "$run/output"
echo "Full diagnostic elapsed seconds: $((SECONDS-start))"
python - "$campaign" "$run" <<'PY'
import json, sys
from pathlib import Path
import h5py
import yaml
campaign, run = map(Path, sys.argv[1:])
sys.path.insert(0, str(campaign))
from campaign import larnd_check, sha
output = run / 'output'
meta = json.loads((output / 'run.json').read_text())
assert meta['settings_campaign_sha256'] == sha(campaign / 'campaign.json')
assert 'A100' in str(meta['device_properties']['name'])
assert 38 < meta['device_properties']['totalGlobalMem'] / 2**30 < 42
cmd = meta['command']
sim = yaml.safe_load(Path(cmd[cmd.index('--simulation_properties')+1]).read_text())
det = yaml.safe_load(Path(cmd[cmd.index('--detector_properties')+1]).read_text())
assert (sim['segment_batch_size'], sim['farfield_enabled'], det['periodic_reset_cycles']) == (10000, True, 256)
assert meta['input']['index'] == 62 and cmd[cmd.index('--rand_seed')+1] == '63'
assert (output / 'SIMULATION_RETURNED_SUCCESSFULLY').exists()
larnd_check(output / 'simulation.LARNDSIM.hdf5')
with h5py.File(output / 'simulation.LARNDSIM.hdf5') as f:
    assert len(f['packets']) > 0
rows = [json.loads(line) for line in (output / 'allocations.jsonl').read_text().splitlines()]
assert any(r['phase'] == 'allocated' and r['ievd'] == 17 for r in rows)
assert not any(r['phase'] == 'error' for r in rows)
(run / 'PASS').write_text(meta['settings_campaign_sha256'] + '\n')
print('PASS: file 62, PRC4, FFE, batch size 10000, 40 GB A100')
PY
