# Contributing to Narrative Consistency Engine

Thank you for your interest in contributing. This document covers everything you need to get started, from setting up the project locally to opening a pull request.

---

## Core Invariants

Every contribution must preserve the following three properties. These are not style preferences; they are architectural constraints that keep the system reliable.

**1. Conservative bias**
When evidence is ambiguous or missing, the system rejects rather than accepts. Do not loosen gate thresholds without demonstrating improvement on the training set via `optimize_thresholds.py`.

**2. Determinism after the LLM stage**
The aggregation layer (`reasoning/aggregation.py`) must remain free of stochastic calls. All LLM interactions belong upstream in normalization or debate. The gate logic must produce the same output given the same inputs, every time.

**3. Config driven behavior**
Thresholds, taxonomy tiers, LLM provider settings, and retrieval parameters all live in `config.yaml`. Do not hardcode values that belong in configuration.

---

## Getting Started

Fork the repository, clone your fork, and create a branch:

```bash
git clone https://github.com/Roshansingh9/Narrative-Consistency-Engine.git
cd Narrative-Consistency-Engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.yaml.example config.yaml
```

Edit `config.yaml` with your API key, then verify the sample runs cleanly before making any changes:

```bash
python pathway_pipeline/index.py        # Terminal A: start vector memory server
python run_inference.py                 # Terminal B: run inference on sample data
```

Check `results.csv`. If all 5 sample rows produce the expected predictions, your environment is ready.

---

## Branch Naming

| Contribution type | Branch pattern |
|---|---|
| New feature | `feature/short description` |
| Bug fix | `fix/short description` |
| Prompt or taxonomy tuning | `tune/what changed` |
| Documentation | `docs/short description` |

Use lowercase words separated by spaces within the branch name slug.

---

## Pull Request Checklist

Before opening a pull request, confirm the following:

1. `run_inference.py` completes without errors on the sample data in `data/test.csv`
2. If your change affects predictions, include a before and after comparison of relevant rows from `results.csv`
3. If you modified thresholds or gate logic, run `optimize_thresholds.py` and report the accuracy result in the PR description
4. All new config keys are documented in `config.yaml.example`
5. No values that belong in `config.yaml` are hardcoded in source files

Your PR description should answer: what changed, why it was needed, and how you verified it works.

---

## High Value Contribution Areas

The following areas have the most room for improvement and the clearest path to measurable impact.

**Prompt hardening**
The normalization and judge prompts are the most fragile part of the pipeline. The system relies on the LLM returning valid JSON on the first attempt. Improvements here include structured output enforcement, few shot examples, and more robust JSON extraction fallbacks.

**Async inference**
The main loop in `run_inference.py` processes claims one at a time. A batched or async version with rate limit handling would reduce wall clock time significantly for large input sets.

**Evaluation metrics**
The threshold optimizer reports overall accuracy. Precision and recall broken down by risk tier would reveal whether the system is over rejecting Low risk claims or under rejecting High risk ones. This would make the grid search results much more interpretable.

**New LLM backends**
The `llm/wrapper.py` module wraps an OpenAI compatible client. Adding native support for Anthropic, Mistral, or local Ollama endpoints would make the system deployable without an external API dependency.

**Evidence quality scoring**
Currently all retrieved chunks are treated equally. A scoring layer that ranks evidence by relevance before passing it to the debate stage could reduce noise in the prosecutor and advocate arguments.

---

## Code Style

Follow the existing patterns in each module. Specifically:

1. Keep functions small and focused on a single responsibility
2. Prefer config driven changes over constants in source files
3. Add a comment only when the reason behind a decision is not obvious from the code itself
4. Do not add error handling for scenarios that cannot occur at runtime

---

## Questions

Open an issue with the `question` label if anything about the architecture, design decisions, or contribution process is unclear. The design rationale section in `README.md` is a good starting point before asking.
