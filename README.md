# Wikidata Movie Knowledge Graph Question Answering

This project implements the Artificial Intelligence I experiment with two comparable systems:

- A0: entity linking, TF-IDF relation classification, graph lookup, and template generation.
- B0-B3: FLAN-T5-small conditioned on graph facts and frozen TransE structural prefixes.
- C0: the same generator without TransE prefixes.

All caches, environments, models, and temporary files are kept under this project on drive H.

## Setup

The virtual environment is stored in the project directory to avoid using drive C:

```powershell
cd H:\AI1experiment
python -m venv .venv
. .\scripts\environment.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Interactive QA demo

The trained models and processed data are already present. From PowerShell, enter the project directory and launch the comparison demo:

```powershell
cd H:\AI1experiment
.\scripts\run_demo.cmd
```

The `.cmd` launcher temporarily bypasses the PowerShell script policy only for this launch and does not modify the system policy.

Enter an English movie question at `Question>` and type `exit` to quit. For example:

```text
Who directed Shutter Island?
What genre is Boss Level?
```

The default `both` mode shows A0 and the validation-selected neural model side by side. The current selected model is B2. Each result displays the linked movie, predicted relation when applicable, graph evidence, final answer, and latency.

Run only one system when needed:

```powershell
.\scripts\run_demo.cmd -System a0
.\scripts\run_demo.cmd -System best
```

Run one question without entering interactive mode:

```powershell
.\scripts\run_demo.cmd -Question "Who directed Shutter Island?" -System both
```

Supported single-hop relations are director, genre, country of origin, and original language. Questions should be in English and refer to movies in the frozen local knowledge base. When the movie cannot be linked or the fact is unavailable, the system returns an explicit refusal.

## Smoke experiment

```powershell
.\scripts\run_experiment.ps1 -Mode smoke -Target 50
```

## Full experiment

```powershell
.\scripts\run_experiment.ps1 -Mode full -Target 1000
```

The Wikidata fetcher caches every successful response and resumes from cached data. The full run downloads a model and performs five independent fine-tuning runs, so it can take several hours. Results are written to `runs/` and figures and evaluation forms to `output/`.

The generated `human_evaluation.csv` is intentionally blank in its score columns. A human evaluator must fill those scores; automated model output is not presented as human judgment.


