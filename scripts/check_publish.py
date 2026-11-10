"""Pre-publish check: no media/meshes/data, no private strings, no secrets, no default remote URL."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAD_EXT = {'.png', '.jpg', '.jpeg', '.webp', '.glb', '.gltf', '.fbx', '.obj', '.blend', '.npz', '.npy', '.safetensors'}
TEXT = re.compile(r'next-genium|[A-Za-z]:\\\\(?!n)|ghp_[A-Za-z0-9]{20,}|github_pat_|sk-[A-Za-z0-9]{20,}|'
                  r'(password|passwd|api_key|token)\s*[:=]\s*["\'][^"\'\s]{6,}', re.I)


def files():
    out = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split()
    return [ROOT / f for f in out]


def main():
    errors = []
    for path in files():
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in BAD_EXT:
            errors.append(f'binary/data file: {rel}')
        if path.suffix.lower() == '.json' and not rel.startswith('examples/'):
            errors.append(f'json outside examples/: {rel}')
        if path.suffix.lower() in {'.py', '.js', '.json', '.yaml', '.yml', '.toml', '.md', '.txt'}:
            for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                if TEXT.search(line) and rel != 'scripts/check_publish.py':
                    errors.append(f'{rel}:{i}: {line.strip()[:100]}')
    for cfg in (ROOT / 'config/3da_server.yaml', ROOT / 'config.yaml'):
        if re.search(r'^\s*url:\s*\S*https?://', cfg.read_text(encoding='utf-8'), re.M):
            errors.append(f'{cfg.name}: remote URL must be empty by default')
    print('\n'.join(errors) or 'check_publish: OK')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
