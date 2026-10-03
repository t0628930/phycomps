"""Integrity and release-readiness checks; never uploads files."""
from pathlib import Path
import argparse,csv,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]

def check(root):
    problems=[]
    manifest=root/'RELEASE_MANIFEST.csv'
    if not manifest.exists(): problems.append('Release manifest missing')
    else:
        with manifest.open(newline='',encoding='utf-8') as handle: entries=list(csv.DictReader(handle))
        for item in entries:
            path=(root/item['file']).resolve()
            if not path.is_relative_to(root.resolve()):
                problems.append('Manifest path outside repository'); continue
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                problems.append('File missing/changed: '+item['file'])
        tracked={x['file'] for x in entries}|{'RELEASE_MANIFEST.csv'}
        for path in root.rglob('*'):
            if not path.is_file() or '.git' in path.parts or '__pycache__' in path.parts: continue
            rel=path.relative_to(root).as_posix()
            if rel not in tracked: problems.append('Unexpected file: '+rel)
            if path.suffix.lower() in ('.pt','.pth','.npy','.npz','.zip','.pem','.key') or path.name.startswith('.env'):
                problems.append('Excluded binary or credential container: '+rel)
            if rel.startswith(('Frozen_Reproducibility/data/','validation_20261001/inputs/','validation_20261001/checkpoints/')) or path.name in ('strict_train.csv','strict_holdout.csv','strict_embargo.csv'):
                problems.append('Excluded data artifact: '+rel)
            if path.suffix.lower() in ('.json','.csv','.md','.txt','.cff','.patch'):
                text=path.read_text(encoding='utf-8-sig')
                if re.search(r'[A-Z]:[\\/]Users[\\/]|/content/drive/|/home/[^/]+/|/Users/[^/]+/',text):
                    problems.append('Personal path remains: '+rel)
                if re.search(r'gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',text):
                    problems.append('Potential secret: '+rel)
    state=json.loads((root/'RELEASE_STATUS.json').read_text(encoding='utf-8'))
    gates=[]
    if not state.get('software_license') or not state.get('license_text_approved') or not (root/'LICENSE').is_file(): gates.append('Software license/rights-holder approval pending')
    if not state.get('repository_url') or not state.get('public_access_verified'): gates.append('Public repository access not verified')
    if not (state.get('release_doi') or state.get('immutable_release_id')): gates.append('Immutable archival release not verified')
    if not state.get('author_affirmations_complete'): gates.append('Administrative author declarations not confirmed')
    if not state.get('published'): gates.append('Publication not performed')
    return {'technical_checks_pass':not problems,'problems':problems,'publication_ready':not problems and not gates,'open_release_gates':gates,'security_scope':'Targeted path/credential patterns and file allowlist; not an exhaustive security audit'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--require-publication-ready',action='store_true')
    args=parser.parse_args(); result=check(ROOT); print(json.dumps(result,indent=2))
    sys.exit(1 if result['problems'] else 2 if args.require_publication_ready and not result['publication_ready'] else 0)