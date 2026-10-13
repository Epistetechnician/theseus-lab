"""Aggregate-only inventory of recorded Theseus runs; never opens raw payloads.

State slices: theseus-lab-sandbox-v1, theseus-all-runs-audit-v1.
The inventory establishes availability and hashes, not experimental validity.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent
BASES = (Path('/Volumes/PrimaryED/ResearchArtifacts'), Path('/Users/shaanp/.codex/custody'))
SKIP = re.compile(r'weight|model|cache|toolchain|runtime|capture|fixture|prediction|outcome|frozen.packet|raw|trace|transcript|score', re.I)
RECEIPT = re.compile(r'report|result|terminal|abort|failure|reconciliation|stop|summary|primary\.json|replication.manifest|prefreeze|followup|execution.final|cleanup|verification', re.I)
STATUS_KEY = re.compile(r'^(status|outcome_status|assurance_level|reproducibility_status|replication_status|reviewer_status|passed|overall_pass|verdict|error_type|error_class|phase|run_status)$')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def statuses(value, prefix=''):
    found = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if STATUS_KEY.match(key) and isinstance(item, (str, bool, int, type(None))):
                found[prefix + key] = item
            elif isinstance(item, dict):
                found.update(statuses(item, prefix + key + '.'))
    return found


def main():
    # Every document is read and bound; extraction is only a navigation aid.
    sources = []
    for path in sorted([REPO / 'NOTES.md', REPO / 'README.md', *REPO.glob('docs/*.md'), *REPO.glob('docs/*.json')]):
        text = path.read_text()
        entry = {'path': str(path.relative_to(REPO)), 'sha256': digest(path), 'bytes': path.stat().st_size}
        if path.suffix == '.json':
            value = json.loads(text)
            entry['statuses'] = statuses(value)
            if isinstance(value, dict):
                entry['protocol_id'] = value.get('protocol_id')
                entry['packet_sha256'] = value.get('packet_sha256')
        else:
            entry['literal_status_tokens'] = sorted(set(re.findall(r'`([A-Z][A-Z0-9_]+)`', text)))
        sources.append(entry)
    custody = []
    for base in BASES:
        for root in sorted(base.glob('*theseus*')):
            if not root.is_dir() or re.search('toolchain|runtime', root.name):
                continue
            receipts = []
            pending = [(root, 0)]
            while pending:
                directory, depth = pending.pop()
                for path in sorted(directory.iterdir()):
                    if path.is_symlink():
                        continue
                    if path.is_dir() and depth < 2 and not SKIP.search(path.name):
                        pending.append((path, depth + 1))
                    elif path.is_file() and path.suffix == '.json' and RECEIPT.search(path.name):
                        if path.stat().st_size > 16 * 1024**2:
                            receipts.append({'name': str(path.relative_to(root)), 'inspection': 'NOT_OPENED_SIZE_CAP'})
                            continue
                        value = json.loads(path.read_text())
                        receipts.append({'name': str(path.relative_to(root)), 'sha256': digest(path), 'bytes': path.stat().st_size, 'statuses': statuses(value)})
            custody.append({'base': base.name, 'root_name': root.name, 'aggregate_receipts': receipts,
                            'availability': 'RECEIPTS_PRESENT' if receipts else 'NO_AGGREGATE_RECEIPT_WITHIN_DECLARED_SCAN'})
    result = {'state_slices': ['theseus-lab-sandbox-v1', 'theseus-all-runs-audit-v1'],
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
              'scope': 'all repository research documents and accessible bounded aggregate receipts through V31; excludes this audit and V32',
              'limitations': ['Not a rerun of old experiments', 'No raw payloads, weights, or row-level files opened', 'Missing receipts do not imply failure or success', 'Undocumented or inaccessible runs cannot be certified'],
              'sources': [x for x in sources if 'all-runs-audit' not in x['path'] and 'v32' not in x['path']],
              'custody': [x for x in custody if 'v32' not in x['root_name']]}
    output = REPO / 'docs/theseus-all-runs-audit-v1.json'
    if output.exists():
        raise FileExistsError('audit already exists; preserve the snapshot')
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'sources': len(result['sources']), 'custody_roots': len(result['custody']), 'receipts': sum(len(x['aggregate_receipts']) for x in result['custody']), 'inventory_sha256': digest(output)}))


if __name__ == '__main__':
    main()
