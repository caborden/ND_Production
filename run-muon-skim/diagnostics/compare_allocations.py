"""Compare file-62 diagnostic directories in the order supplied."""
import argparse
from datetime import datetime
import json
from pathlib import Path


def report(root):
    meta = json.loads((root / 'run.json').read_text())
    rows = [json.loads(line) for line in (root / 'allocations.jsonl').read_text().splitlines()]
    starts = [r for r in rows if r['phase'] == 'before']
    errors = [r for r in rows if r['phase'] == 'error']
    complete = (root / 'SIMULATION_RETURNED_SUCCESSFULLY').is_file()
    print(f"\n{root}\nGPU {meta['device_properties']['name']} | simulator {meta['simulator_sha256']}")
    print('Outcome:', 'completed' if complete else errors[-1]['error'] if errors else 'see Slurm log')
    event = [r for r in starts if r['ievd'] == 17]
    for label, group in [('whole file attempted', starts), ('event 17', event)]:
        if not group:
            continue
        peak = max(group, key=lambda r: r['requested_bytes'])
        overlaps = sum(r['previous_signals_bytes'] > 0 for r in group)
        print(f"{label}: {len(group)} requests, {overlaps} with a live previous signals array")
        print(f"  largest request {peak['requested_bytes']/2**30:.2f} GiB; "
              f"event {peak['ievd']}, shape {peak['shape']}, pixels [{peak['start_pix']}, {peak['stop_pix']})")
        for key in ('previous_signals_bytes', 'pool_used_bytes', 'pool_cached_bytes',
                    'cuda_free_bytes', 'cuda_total_bytes'):
            print(f"  {key} before largest request: {peak[key]/2**30:.2f} GiB")
    if event:
        elapsed = datetime.fromisoformat(event[0]['utc']) - datetime.fromisoformat(starts[0]['utc'])
        print(f"First allocation to first event-17 allocation: {elapsed.total_seconds():.1f} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    for directory in parser.parse_args().directories:
        report(directory)
