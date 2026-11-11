#!/usr/bin/env python3
"""Build the UCM 0.3.0 CUDA store in the active Python 3.10 Conda environment."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    prefix = Path(sys.prefix)
    if sys.version_info[:2] != (3, 10) or not (prefix / 'conda-meta').is_dir():
        raise SystemExit('Activate the Python 3.10 ucm Conda environment first.')
    nvcc = prefix / 'bin/nvcc'
    if not nvcc.is_file() or 'release 12.6,' not in subprocess.check_output(
            [str(nvcc), '--version'], text=True):
        raise SystemExit('Install cuda-nvcc=12.6 in this Conda environment first.')
    cache = Path(__file__).resolve().parents[1] / '.cache/env-build'
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ucm-native-', dir=cache) as temp:
        build = Path(temp)
        source, cuda = build / 'source', build / 'cuda'
        subprocess.run([
            'git', 'clone', '--depth', '1', '--branch', 'v0.3.0',
            'https://github.com/ModelEngine-Group/unified-cache-management.git',
            str(source),
        ], check=True)
        # NVIDIA Conda packages use targets/...; upstream expects a toolkit root.
        cuda.mkdir()
        for name, target in {
            'bin': prefix / 'bin', 'nvvm': prefix / 'nvvm',
            'include': prefix / 'targets/x86_64-linux/include',
            'lib64': prefix / 'targets/x86_64-linux/lib',
            'targets': prefix / 'targets',
        }.items():
            if not target.is_dir():
                raise SystemExit(f'Missing CUDA toolkit directory: {target}')
            (cuda / name).symlink_to(target, target_is_directory=True)
        for relative in ('ucm/shared/trans/cuda', 'ucm/store/nfsstore/device/cuda'):
            path = source / relative / 'CMakeLists.txt'
            text = path.read_text()
            old = 'set(CUDA_ROOT "/usr/local/cuda/" CACHE PATH "Path to CUDA root directory")'
            if old not in text:
                raise SystemExit(f'Unexpected upstream CUDA configuration: {path}')
            path.write_text(text.replace(old, old.replace('/usr/local/cuda/', '$ENV{CUDA_HOME}')))
        # Upstream includes sm_80 (A100), sm_89 (L20), sm_90 (H20).
        # AdaTree uses this checkout's sparse implementation; only build its store.
        env = dict(os.environ, PLATFORM='cuda', ENABLE_SPARSE='false',
                   CUDA_HOME=str(cuda), CUDACXX=str(nvcc),
                   CC=str(prefix / 'bin/x86_64-conda-linux-gnu-cc'),
                   CXX=str(prefix / 'bin/x86_64-conda-linux-gnu-c++'),
                   PATH=str(prefix / 'bin') + os.pathsep + os.environ['PATH'])
        # GCC 12 emits this warning in pybind11's LTO build. Keep the warning
        # visible instead of allowing upstream's global -Werror to abort linking.
        env['CXXFLAGS'] = env.get('CXXFLAGS', '') + ' -Wno-error=stringop-overread'
        # Link against the NVML development stub; runtime uses the host driver.
        # Never add stubs to LD_LIBRARY_PATH.
        env['LIBRARY_PATH'] = str(cuda / 'lib64/stubs') + (
            os.pathsep + env['LIBRARY_PATH'] if env.get('LIBRARY_PATH') else '')
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--no-build-isolation',
            '--no-deps', '--force-reinstall', str(source),
        ], env=env, check=True)


if __name__ == '__main__':
    main()
