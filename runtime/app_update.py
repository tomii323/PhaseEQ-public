"""Public-release updates: fetch, verify, stage, then apply while apps are stopped.

This module uses only the standard library so launchers can run it before repairing
Python dependencies. User data and Git checkouts are never update destinations.
"""
from __future__ import annotations

import argparse
import atexit
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
import urllib.parse
import urllib.request
import uuid
import zipfile

REPOSITORY = 'tomii323/PhaseEQ-public'
API_URL = f'https://api.github.com/repos/{REPOSITORY}/releases/latest'
MAX_ARCHIVE = 256 * 1024 * 1024
MAX_EXPANDED = 512 * 1024 * 1024
ROOT_FILES = {'VERSION', 'README.md', 'CHANGELOG.md', 'LICENSE.md', 'pyproject.toml',
              'requirements.txt', 'phaseeq.py', 'composite_streamlit_app.py',
              'run_PhaseEQ_mac_linux.sh', 'run_PhaseEQ_windows.bat',
              '.gitignore', '.gitattributes', '.streamlit/config.toml', 'PUBLIC_FILES.sha256'}
DIRECTORIES = {'src', 'ui', 'utils', 'runtime', 'resources', 'constraints'}


def version(value):
    match = re.fullmatch(r'v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)', value.strip())
    if not match:
        raise ValueError('Invalid release version')
    return tuple(int(part) for part in match.groups())


def update_directory(root):
    return Path(root) / '.phaseeq-updates'


def writable_installation(root):
    root = Path(root)
    return (not (root / '.git').exists() and os.access(root, os.W_OK)
            and (root / 'PUBLIC_FILES.sha256').is_file())


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _ReleaseRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if (parsed.scheme != 'https' or parsed.username or parsed.password
                or parsed.hostname not in {'github.com', 'release-assets.githubusercontent.com',
                    'objects.githubusercontent.com', 'github-releases.githubusercontent.com'}):
            raise ValueError('Unexpected download redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _read_url(url, *, limit, timeout=4):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or parsed.username or parsed.password:
        raise ValueError('Invalid release URL')
    if url != API_URL and not url.startswith(f'https://github.com/{REPOSITORY}/releases/download/'):
        raise ValueError('Unexpected release source')
    request = urllib.request.Request(url, headers={'User-Agent': 'PhaseEQ-Updater',
                                                 'Accept': 'application/vnd.github+json'})
    opener = urllib.request.build_opener(_ReleaseRedirects())
    with opener.open(request, timeout=timeout) as response:
        final = urllib.parse.urlsplit(response.url)
        if final.scheme != 'https' or final.hostname not in {
            'api.github.com', 'github.com', 'release-assets.githubusercontent.com',
            'objects.githubusercontent.com', 'github-releases.githubusercontent.com',
        }:
            raise ValueError('Unexpected download redirect')
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Release download exceeds size limit')
    return data


def latest_release(current, *, fetch=_read_url):
    payload = json.loads(fetch(API_URL, limit=1024 * 1024))
    tag = str(payload.get('tag_name', ''))
    if payload.get('draft') or payload.get('prerelease') or version(tag) <= version(current):
        return None
    name = f'PhaseEQ_{tag}.zip'
    assets = {item['name']: item['browser_download_url'] for item in payload.get('assets', [])}
    if name not in assets or name + '.sha256' not in assets:
        raise ValueError('Release ZIP or checksum is missing')
    expected = f'https://github.com/{REPOSITORY}/releases/download/{tag}/'
    if assets[name] != expected + name or assets[name + '.sha256'] != expected + name + '.sha256':
        raise ValueError('Release asset URL does not match its version')
    return {'version': tag.lstrip('v'), 'zip_url': assets[name],
            'checksum_url': assets[name + '.sha256'],
            'release_url': f'https://github.com/{REPOSITORY}/releases/tag/{tag}'}


def _safe_path(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or str(path) != name or '..' in path.parts
            or '\\' in name or ':' in name or any(part.endswith((' ', '.')) for part in path.parts)):
        raise ValueError('Unsafe release path: ' + name)
    reserved = {'con', 'prn', 'aux', 'nul', *(f'com{i}' for i in range(1, 10)), *(f'lpt{i}' for i in range(1, 10))}
    if any(part.split('.')[0].casefold() in reserved for part in path.parts):
        raise ValueError('Reserved release path: ' + name)
    if name in ROOT_FILES or path.parts[0] in DIRECTORIES or name.startswith('docs/distribution/'):
        return name
    if name == '.github/workflows/release.yml':
        return name
    raise ValueError('Protected or unknown release path: ' + name)


def _inventory(data):
    result = {}
    for line in data.decode('utf-8').splitlines():
        digest, separator, name = line.partition('  ')
        if not separator or not re.fullmatch('[0-9a-f]{64}', digest) or name in result:
            raise ValueError('Invalid public file inventory')
        result[_safe_path(name)] = digest
    if not result:
        raise ValueError('Empty public file inventory')
    return result


def verified_files(data, expected_version):
    files = {}
    folded = set()
    total = 0
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        prefix = f'PhaseEQ_v{expected_version}/'
        for entry in archive.infolist():
            if not entry.filename.startswith(prefix):
                raise ValueError('Release ZIP directory does not match version')
            if entry.is_dir():
                continue
            name = _safe_path(entry.filename[len(prefix):])
            mode = entry.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise ValueError('Release ZIP contains a non-regular file')
            if name.casefold() in folded:
                raise ValueError('Release ZIP contains duplicate paths')
            folded.add(name.casefold())
            total += entry.file_size
            if total > MAX_EXPANDED:
                raise ValueError('Expanded release exceeds size limit')
            files[name] = (archive.read(entry), mode & 0o777)
    hashes = _inventory(files['PUBLIC_FILES.sha256'][0])
    if set(files) != set(hashes) | {'PUBLIC_FILES.sha256'}:
        raise ValueError('Release ZIP does not match its public inventory')
    for name, digest in hashes.items():
        if hashlib.sha256(files[name][0]).hexdigest() != digest:
            raise ValueError('Public file checksum mismatch: ' + name)
    if files['VERSION'][0].decode().strip() != expected_version:
        raise ValueError('Release VERSION mismatch')
    for required in ('phaseeq.py', 'composite_streamlit_app.py', 'runtime/app_update.py',
                     'requirements.txt', 'run_PhaseEQ_mac_linux.sh', 'run_PhaseEQ_windows.bat'):
        if required not in files:
            raise ValueError('Required application file is missing: ' + required)
    return {name: value for name, value in files.items() if not name.startswith('.github/')}


@contextmanager
def _update_lock(root):
    directory = update_directory(root)
    if directory.is_symlink():
        raise ValueError('Update directory cannot be a symlink')
    directory.mkdir(exist_ok=True)
    lock = directory / 'update.lock'
    if lock.is_symlink():
        raise ValueError('Update lock cannot be a symlink')
    # Kernel locks are released even after a crash, allowing journal recovery.
    with lock.open('a+b') as handle:
        if lock.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('Another update is active; retry after it finishes') from exc
        try:
            yield directory
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def stage_release(root, release, *, fetch=_read_url):
    root = Path(root)
    if not writable_installation(root):
        raise ValueError('Git checkouts or read-only installations cannot be updated')
    if version(release['version']) <= version((root / 'VERSION').read_text()):
        raise ValueError('Release is not newer than this installation')
    with _update_lock(root) as directory:
        if (directory / 'pending.json').exists() or (directory / 'journal.json').exists():
            raise ValueError('An update is already pending')
        data = fetch(release['zip_url'], limit=MAX_ARCHIVE, timeout=30)
        checksum = fetch(release['checksum_url'], limit=4096, timeout=10).decode('ascii').strip()
        digest, separator, filename = checksum.partition('  ')
        if (not separator or filename != f"PhaseEQ_v{release['version']}.zip"
                or hashlib.sha256(data).hexdigest() != digest):
            raise ValueError('Release SHA-256 mismatch')
        files = verified_files(data, release['version'])
        identifier = uuid.uuid4().hex
        staged = directory / identifier
        staged.mkdir()
        for name, (content, mode) in files.items():
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path.chmod(mode or 0o644)
        _atomic_json(directory / 'pending.json', {'version': release['version'], 'id': identifier,
                     'files': {name: hashlib.sha256(content).hexdigest()
                               for name, (content, _) in files.items()}})
    return release['version']


def cancel_pending(root):
    with _update_lock(root) as directory:
        if (directory / 'journal.json').exists():
            raise ValueError('Recover the interrupted update before cancelling')
        pending = directory / 'pending.json'
        if pending.exists():
            os.replace(pending, directory / f'cancelled-{time.time_ns()}.json')


def _pid_alive(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            # Access denied must never be mistaken for a stopped elevated app.
            return ctypes.get_last_error() != 87
        code = wintypes.DWORD()
        try:
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def register_running_app(root):
    """One marker per process, covering both PhaseEQ and Multiway."""
    directory = update_directory(root) / 'running'
    marker = directory / f'{os.getpid()}.json'
    if marker.exists():
        return
    with _update_lock(root):
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_json(marker, {'pid': os.getpid()})
        atexit.register(lambda: marker.unlink(missing_ok=True))


def _destination(root, name):
    _safe_path(name)
    path = root / name
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('Update destination contains a symlink: ' + name)
    return path


def _rollback(root, directory, journal):
    if not re.fullmatch(r'backup-\d+', journal['backup']):
        raise ValueError('Invalid update backup identifier')
    backup = directory / journal['backup']
    for name in reversed(journal['files']):
        destination = _destination(root, name)
        original = backup / name
        if original.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
        elif name in journal['new'] and destination.exists():
            quarantine = backup / 'new-files' / name
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            os.replace(destination, quarantine)
    (directory / 'journal.json').unlink()


def _previous_backups(directory):
    """Select only backups recorded by successful updates, never scan folders."""
    result_path = directory / 'last-result.json'
    if not result_path.exists():
        return []
    result = json.loads(result_path.read_text())
    names = [result['backup'], *result.get('cleanup_pending', [])]
    if any(not isinstance(name, str) or not re.fullmatch(r'backup-\d+', name) for name in names):
        raise ValueError('Invalid recorded update backup identifier')
    return list(dict.fromkeys(names))


def _finish_committed_update(directory, journal):
    """Publish success before pruning; cleanup failures cannot trigger rollback."""
    result = journal.get('result')
    if result is not None:
        # Persist the deletion plan before touching backups so crash recovery
        # can repeat cleanup without losing the previous successful identity.
        _atomic_json(directory / 'last-result.json', result)
        remaining = []
        for name in result.get('cleanup_pending', []):
            try:
                if not re.fullmatch(r'backup-\d+', name):
                    raise ValueError('Invalid backup cleanup target')
                path = directory / name
                if path.is_symlink():
                    raise ValueError('Backup cleanup target cannot be a symlink')
                if path.exists():
                    shutil.rmtree(path)
            except (OSError, ValueError) as exc:
                remaining.append(name)
                print(f'PhaseEQ更新は成功しましたが、旧バックアップ {name} の削除は未完了です: {exc}')
        result = {**result, 'cleanup_pending': remaining}
        _atomic_json(directory / 'last-result.json', result)
    (directory / 'pending.json').unlink(missing_ok=True)
    (directory / 'journal.json').unlink()


def apply_pending(root):
    root = Path(root).resolve()
    directory = update_directory(root)
    if not (directory / 'pending.json').exists() and not (directory / 'journal.json').exists():
        return False
    if not writable_installation(root):
        raise ValueError('Git checkouts or read-only installations cannot be updated')
    with _update_lock(root):
        for marker in (directory / 'running').glob('*.json'):
            if _pid_alive(int(marker.stem)):
                raise ValueError('Close both PhaseEQ and Multiway before updating')
        journal_path = directory / 'journal.json'
        if journal_path.exists():
            journal = json.loads(journal_path.read_text())
            if journal.get('committed'):
                _finish_committed_update(directory, journal)
                return True
            _rollback(root, directory, journal)
        pending = json.loads((directory / 'pending.json').read_text())
        if not re.fullmatch('[0-9a-f]{32}', pending['id']):
            raise ValueError('Invalid staged update identifier')
        if version(pending['version']) <= version((root / 'VERSION').read_text()):
            # A stale reservation must not block startup or overwrite the
            # current application. Keep its record and staged files recoverable.
            retired = directory / f'cancelled-{time.time_ns()}.json'
            os.replace(directory / 'pending.json', retired)
            print('PhaseEQ stale update reservation cancelled; application files unchanged:', retired)
            return False
        staged = directory / pending['id']
        incoming = pending['files']
        previous_path = root / 'PUBLIC_FILES.sha256'
        if not previous_path.is_file():
            raise ValueError('This installation lacks PUBLIC_FILES.sha256; use the official public ZIP first')
        previous = _inventory(previous_path.read_bytes())
        retired = set(previous) - set(incoming) - {'.github/workflows/release.yml'}
        names = sorted(set(incoming) | retired)
        for name, digest in incoming.items():
            _destination(root, name)
            source = _destination(staged, name)
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError('Staged file checksum mismatch: ' + name)
        modified_metadata = []
        for name, digest in previous.items():
            if name.startswith('.github/'):
                continue
            destination = _destination(root, name)
            if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                if name in {'.gitattributes', '.gitignore'}:
                    modified_metadata.append(name)
                    continue
                raise ValueError('Locally modified application file: ' + name)
        previous_backups = _previous_backups(directory)
        metadata_recovery = None
        if modified_metadata:
            # Git metadata is not runtime configuration. Preserve local edits
            # separately from successful-update backups, which are pruned.
            metadata_recovery = 'metadata-recovery-' + uuid.uuid4().hex
            recovery = directory / metadata_recovery
            recovery.mkdir()
            for name in modified_metadata:
                shutil.copy2(_destination(root, name), recovery / name)
            print('Local Git metadata preserved:', recovery)
        backup_name = 'backup-' + str(time.time_ns())
        backup = directory / backup_name
        backup.mkdir()
        new = []
        for name in names:
            destination = _destination(root, name)
            if destination.exists():
                saved = backup / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, saved)
            else:
                new.append(name)
        journal = {'backup': backup_name, 'files': names, 'new': new,
                   'result': {'version': pending['version'], 'backup': backup_name,
                              'metadata_recovery': metadata_recovery,
                              'cleanup_pending': [*previous_backups, backup_name]}}
        _atomic_json(journal_path, journal)
        try:
            for name in names:
                destination = _destination(root, name)
                if name in retired:
                    if destination.exists():
                        retired_path = backup / 'retired' / name
                        retired_path.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(destination, retired_path)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.phaseeq-update-tmp')
                try:
                    shutil.copy2(staged / name, temporary)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            journal['committed'] = True
            _atomic_json(journal_path, journal)
        except Exception:
            _rollback(root, directory, journal)
            raise
        _finish_committed_update(directory, journal)
    return True


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--cancel', action='store_true')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        if args.cancel:
            cancel_pending(args.root)
        changed = apply_pending(args.root) if args.apply else False
    except (OSError, ValueError, KeyError) as exc:
        print('PhaseEQ update stopped:', exc)
        return 1
    if changed:
        print('PhaseEQ update applied. Restarting launcher.')
        return 10
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
