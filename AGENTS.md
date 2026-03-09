# PaperBanana Agent Notes

## Project Summary
- PaperBanana is a multi-agent academic illustration system built around `Retriever -> Planner -> Stylist -> Visualizer -> Critic`.
- Primary entry points:
  - `demo.py`: Streamlit GUI for candidate generation and image refinement.
  - `main.py`: batch CLI runner.
  - `visualize/`: result viewers.

## Active Priorities
1. Fix deterministic pipeline bugs before adding new product features.
2. Keep CLI, GUI, and visualizers behaviorally consistent.
3. Reduce hidden state and hard-coded stage assumptions.
4. Preserve backward compatibility for existing result JSON files when possible.

## Current Risk Areas
- Provider/config state is still fragmented across `utils/config.py`, `utils/generation_utils.py`, `demo.py`, and CLI parsing.
- Critic round handling was historically hard-coded in multiple places.
- Plot support exists in the codebase, but the GUI product surface is still diagram-first.

## Working Conventions
- Prefer minimal, reversible changes that improve real user flows first.
- When adding new result metadata, keep old keys readable.
- When changing pipeline semantics, update viewers in the same batch.
- Record architectural decisions and progress in `docs/agent-memory.md`.
- Record prioritized follow-up work in `docs/optimization-backlog.md`.
- Prefer the user's existing global/shared Python and `uv` environment for this repo.
- Do not create a project-local `.venv` unless the user explicitly asks for a new virtual environment.

## Validation
- Use `python -m compileall` for a fast syntax pass after multi-file edits.
- Prefer focused validation over broad speculative refactors.
