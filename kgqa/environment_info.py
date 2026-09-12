from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
import os

import psutil
import torch

from .common import ROOT, utc_now, write_json


def collect():
    processor = platform.processor()
    if os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                processor = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    packages = ['torch', 'transformers', 'pykeen', 'scikit-learn', 'sacrebleu', 'rouge-score', 'numpy', 'pandas']
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], capture_output=True, text=True, check=False).stdout.strip()
    info = {'collected_at': utc_now(), 'project_directory': str(ROOT), 'python_executable': sys.executable,
            'python_version': sys.version, 'operating_system': platform.platform(), 'processor': processor,
            'logical_cpu_count': psutil.cpu_count(), 'memory_gb': round(psutil.virtual_memory().total / 2**30, 2),
            'gpu': gpu, 'cuda_available': torch.cuda.is_available(), 'torch_cuda': torch.version.cuda,
            'packages': {name: importlib.metadata.version(name) for name in packages}}
    write_json(ROOT / 'output' / 'environment.json', info); print(json.dumps(info, indent=2)); return info


if __name__ == '__main__': collect()
