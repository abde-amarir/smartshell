# SmartShell 
### A Local, ML-Powered Personal Command Predictor for Linux

SmartShell monitors your terminal usage, learns your command patterns, and predicts what you'll type next — before you type it. It acts like autocomplete for your shell, but trained entirely on *your* behavior, running entirely on *your* machine.

The goal is to build a fully offline, personal shell assistant that improves productivity while serving as a real-world machine learning system.

---

## The Problem

Advanced Linux users execute the same command chains repeatedly:

```bash
git pull → cd src → python manage.py runserver
sudo apt update → sudo apt full-upgrade → sudo apt autoremove
docker ps → docker logs <container> → docker exec -it <container> bash
```

SmartShell learns these patterns and suggests the next command automatically — saving keystrokes, reducing errors, and speeding up repetitive workflows.

---

## How It Works

```
Your terminal
     │
     ▼
[Collector]  — captures command, timestamp, directory, exit code
     │
     ▼
[Processor]  — normalizes, tokenizes, engineers features
     │
     ▼
[Model]      — predicts next command from sequence context
     │
     ▼
[Engine]     — displays suggestion, waits for confirmation
     │
     ▼
You press Enter (or ESC to ignore)
```

No data leaves your machine. No cloud API. Fully local.

---

## Architecture

### 1. Data Collection Layer (`collector/`)
Hooks into your shell via `PROMPT_COMMAND` to capture:
- Command text
- Timestamp
- Working directory (`pwd`)
- Exit code
- Stores into a local SQLite database

### 2. Data Processing Layer (`processor/`)
Transforms raw logs into structured training data:
- Command normalization (strips dynamic arguments, canonicalizes paths)
- Feature engineering (time-of-day buckets, directory context, sequence windows)
- Builds `[last N commands] → next command` training pairs

### 3. Modeling Layer (`models/`)
Three progressively powerful models — built and compared in phases:

| Phase | Model | Purpose |
|---|---|---|
| 1 | Markov Chain | Baseline — pure transition probabilities |
| 2 | Logistic Regression / Random Forest | Classical ML with engineered features |
| 3 | LSTM / Transformer | Deep learning — full sequence context |

### 4. Suggestion Engine (`engine/`)
A CLI tool that:
- Reads the last command from the database
- Queries the trained model
- Displays the prediction inline using `prompt_toolkit`
- Accepts Enter to run or ESC to ignore

### 5. Shell Integration (`integration/`)
Two integration modes:
- **`PROMPT_COMMAND` hook** — triggers prediction before each new prompt (recommended)
- **`smartsh` wrapper** — custom shell entry point

---

## Project Structure

```
smartshell/
├── collector/
│   ├── __init__.py
│   └── logger.py          # Shell hook, SQLite writer
├── processor/
│   ├── __init__.py
│   ├── pipeline.py        # End-to-end data pipeline
│   └── normalizer.py      # Command normalization logic
├── models/
│   ├── __init__.py
│   ├── markov.py          # Phase 1 — Markov Chain
│   ├── sklearn_model.py   # Phase 2 — Classical ML
│   └── lstm_model.py      # Phase 3 — Deep Learning
├── engine/
│   ├── __init__.py
│   ├── predictor.py       # Model query interface
│   └── cli.py             # User-facing suggestion UI
├── data/
│   ├── logs/              # Raw SQLite database (gitignored)
│   └── processed/         # Cleaned training data (gitignored)
├── tests/
├── config.py              # Central configuration
├── main.py                # Entry point
├── requirements.txt
└── README.md
```

---

## Roadmap

- [x] Phase 0 — Project structure and environment setup
- [x] Phase 1 — Shell logger + SQLite collector
- [x] Phase 2 — Data processing pipeline and normalization
- [x] Phase 3 — Markov Chain model (baseline)
- [ ] Phase 4 — CLI suggestion engine with `prompt_toolkit`
- [ ] Phase 5 — scikit-learn classifier (Logistic Regression + Random Forest)
- [ ] Phase 6 — LSTM sequence model with PyTorch
- [ ] Phase 7 — Shell integration via `PROMPT_COMMAND`
- [ ] Phase 8 — Packaging, systemd service, Docker (optional)

---

## Security

SmartShell is designed to **suggest, never execute.**

- Suggestions are always confirmed manually by the user
- A hardcoded blacklist prevents dangerous commands from ever being suggested:
  - `rm`, `dd`, `mkfs`, `shutdown`, `reboot`, `:(){ :|:& };:`, and others
- Any command beginning with `sudo` requires explicit confirmation
- No `eval`, no `bash -c`, no automatic execution under any condition
- All data is stored locally — no telemetry, no external APIs

---

## Tech Stack

| Layer | Tools |
|---|---|
| Shell integration | Bash, `PROMPT_COMMAND` |
| Storage | SQLite, `sqlite-utils` |
| Data processing | Python, `pandas`, `regex` |
| Classical ML | `scikit-learn` |
| Deep Learning | `PyTorch` |
| CLI interface | `prompt_toolkit`, `rich` |
| Packaging (optional) | Docker, `systemd` |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/abde-amarir/smartshell.git
cd smartshell

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Status

> **Active development — Phase 0 complete.**
> This project is being built incrementally as a structured ML learning project.
> Follow the commits to see each phase take shape.

---

## License

MIT License — see `LICENSE` for details.