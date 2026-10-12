"""Contained source freshness and a separate, per-platform mutant cache.

Never clean a target. Source touching is content-neutral and limited to workspace
members; a new checkout must not inherit another checkout's embedded manifest path.
"""
import argparse
import os
from pathlib import Path
import stat

PROJECT = Path('E:/AI/Projects/OpenMinds/Minions/Minion-Agent')


def checked(path, boundary):
    path, boundary = Path(path), Path(boundary)
    if not path.is_absolute() or '..' in path.parts or path == boundary or not path.is_relative_to(boundary):
        raise RuntimeError(f'outside cache/source sandbox: {path}')
    for entry in [*reversed(path.parents), path]:
        try:
            metadata = entry.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, 'st_file_attributes', 0) & 1024:
            raise RuntimeError(f'link/reparse in cache/source target: {entry}')
    if not path.resolve(strict=False).is_relative_to(boundary.resolve(strict=True)):
        raise RuntimeError(f'outside real cache/source target: {path}')
    return path


def refresh(tree):
    boundary = PROJECT if os.name == 'nt' else Path('/tmp')
    tree = checked(tree, boundary)
    count = 0
    # All workspace-member Rust sources, manifests and build scripts. Targets
    # are external and never walked. Inspect every directory before descending.
    for directory, dirs, files in os.walk(tree, followlinks=False):
        for name in list(dirs):
            if name in ('.git', 'target'):
                dirs.remove(name)
            else:
                checked(Path(directory) / name, boundary)
        for name in files:
            if name.endswith('.rs') or name in ('Cargo.toml', 'Cargo.lock'):
                os.utime(checked(Path(directory) / name, boundary), None)
                count += 1
    print(f'FRESH WORKSPACE SOURCES {count}: {tree}', flush=True)
    return count


def control_target():
    expected = PROJECT / '.tmp/cache/cargo-target/windows-controls' if os.name == 'nt' else Path('/control-target')
    supplied = Path(os.environ.get('MINION_CONTROL_CARGO_TARGET', ''))
    if supplied != expected:
        raise RuntimeError(f'MINION_CONTROL_CARGO_TARGET must be {expected}; no shared gate target for mutants')
    if os.name == 'nt':
        checked(expected, PROJECT)
        if not expected.exists():
            checked(expected, PROJECT).mkdir()
    else:
        # The launcher binds this exact directory from the contained E: cache.
        metadata = expected.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError('dedicated /control-target bind required')
    return str(expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', type=Path, required=True)
    args = parser.parse_args()
    refresh(args.tree)
