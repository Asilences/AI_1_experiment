from __future__ import annotations

import json
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw, ImageFont

from .common import ROOT, read_json

WIDTH, HEIGHT = 1600, 900
BG, TOP, FG, MUTED, GREEN = '#0c0c0c', '#202020', '#eeeeee', '#b8b8b8', '#7fdc7f'


def font(size=24, bold=False):
    candidates = [Path(r'C:\Windows\Fonts\consolab.ttf' if bold else r'C:\Windows\Fonts\consola.ttf'), Path(r'C:\Windows\Fonts\lucon.ttf')]
    for path in candidates:
        if path.exists(): return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def wrap_lines(lines, width=104):
    result=[]
    for line in lines:
        pieces=textwrap.wrap(str(line), width=width, subsequent_indent='  ', replace_whitespace=False) or ['']
        result.extend(pieces)
    return result


def render(lines, output):
    image=Image.new('RGB',(WIDTH,HEIGHT),BG); draw=ImageDraw.Draw(image)
    draw.rectangle((0,0,WIDTH,46),fill=TOP); draw.ellipse((18,15,34,31),fill='#e5534b'); draw.ellipse((44,15,60,31),fill='#d9b329'); draw.ellipse((70,15,86,31),fill='#45a65a')
    draw.text((105,10),'Windows PowerShell',font=font(20,bold=True),fill=FG)
    y=65; body=font(23); prompt=font(23,bold=True)
    for line in wrap_lines(lines):
        color=GREEN if line.startswith('PS H:') else (MUTED if line.startswith('#') else FG)
        draw.text((28,y),line,font=prompt if line.startswith('PS H:') else body,fill=color)
        y += 31
        if y > HEIGHT-35: break
    output.parent.mkdir(parents=True,exist_ok=True); image.save(output)


def build():
    out=ROOT/'output'/'evidence'; out.mkdir(parents=True,exist_ok=True)
    env=read_json(ROOT/'output'/'environment.json')
    env_lines=[
        r'PS H:\AI1experiment> . .\scripts\environment.ps1',
        r'PS H:\AI1experiment> .\.venv\Scripts\python.exe -m kgqa.environment_info',
        f"project_directory: {env['project_directory']}", f"python_executable: {env['python_executable']}",
        f"python_version: {env['python_version'].split('|')[0].strip()}", f"operating_system: {env['operating_system']}",
        f"processor: {env['processor']}", f"logical_cpu_count: {env['logical_cpu_count']}", f"memory_gb: {env['memory_gb']}", f"gpu: {env['gpu']}",
        f"cuda_available: {env['cuda_available']}", f"torch_cuda: {env['torch_cuda']}",
        'packages: ' + ', '.join(f'{k}={v}' for k,v in env['packages'].items()),
        r'PS H:\AI1experiment> Get-Location', r'Path', r'----', r'H:\AI1experiment']
    render(env_lines,out/'environment.png')

    config=read_json(ROOT/'runs'/'B0'/'config.json'); history=read_json(ROOT/'runs'/'B0'/'history.json')
    train_lines=[r'PS H:\AI1experiment> .\.venv\Scripts\python.exe -m kgqa.neural --run-id B0 --learning-rate 1e-4',
                 f"# device={config['device']} batch_size={config['batch_size']} effective_batch={config['effective_batch']} epochs={config['epochs']} seed={config['seed']}"]
    for row in history:
        train_lines.append(f"run=B0 epoch={row['epoch']} train_loss={row['train_loss']:.5f} valid_loss={row['validation_loss']:.5f} valid_f1={row['entity_macro_f1']:.3f} seconds={row['epoch_seconds']:.1f}")
    sel=read_json(ROOT/'runs'/'B0'/'selection.json')
    train_lines += [f"selected checkpoint: epoch={sel['best_epoch']} validation_f1={sel['best_valid_f1']:.3f}", f"training_seconds={sel['training_seconds']:.1f}", r'PS H:\AI1experiment>']
    render(train_lines,out/'training_b0.png')

    eval_lines=[r'PS H:\AI1experiment> .\.venv\Scripts\python.exe -m kgqa.plots', '# Frozen test set evaluation']
    for run in ('A0','B0','B1','B2','B3','C0'):
        m=read_json(ROOT/'runs'/run/'metrics_test.json')
        eval_lines.append(f"{run}: exact={m['entity_exact']:.2f} f1={m['entity_macro_f1']:.2f} BLEU={m['bleu']:.2f} ROUGE-L={m['rouge_l']:.2f} mean_ms={m.get('latency_mean_ms',0):.2f} p95_ms={m.get('latency_p95_ms',0):.2f}")
    selected=read_json(ROOT/'output'/'selected_model.json')
    eval_lines += [f"validation-selected neural system: {selected['selected_run']} ({selected['validation_f1']:.3f})", r'PS H:\AI1experiment>']
    render(eval_lines,out/'evaluation.png')


if __name__=='__main__': build()
