# Agent Memory

## Architecture Snapshot
- `agents/` contains the business stages of the pipeline.
- `utils/paperviz_processor.py` orchestrates stage execution and evaluation.
- `utils/generation_utils.py` currently owns most provider/runtime plumbing.
- `demo.py` is the main user-facing product surface.
- `visualize/` must stay compatible with historical result files.

## Current Decisions
- First optimization wave focuses on deterministic bugs and consistency gaps:
  - retriever error-context crash
  - critic parse-failure semantics
  - missing defaults in vanilla Gemini image path
  - critic round hard-coding across generation, export, and visualization
  - CLI parameter parity improvements
- We are not attempting a full provider abstraction rewrite in the first wave.
- Environment policy for this workspace:
  - Prefer the user's existing global/shared Python and `uv` environment.
  - Do not create a new project-local `.venv` unless the user explicitly requests it.
  - A `.venv` was mistakenly created on 2026-03-09 during dataset setup and was immediately removed after the user corrected the preference.

## Compatibility Rules
- Result keys like `target_*_critic_descN` remain the source of truth for historical runs.
- New metadata may be added, but old files should still render.
- GUI remains Streamlit for now.

## Validation Status
- 2026-03-09 Wave 1 syntax check passed via `python -m compileall AGENTS.md docs main.py demo.py agents utils visualize`.
- 2026-03-09 Completed in Wave 1:
  - fixed retriever `candidate_id` propagation for auto retrieval
  - fixed critic parse failure semantics to stop cleanly instead of pretending `No changes needed`
  - added safe defaults for vanilla Gemini image generation
  - removed hard-coded critic round assumptions from generation/display/export paths
  - improved CLI parameter parity for provider/image model/concurrency
  - fixed referenced-eval prompt hot reload import and latest-critic selection
  - bounded refine retries with attempt/time caps to avoid endless hangs
  - decoupled refine provider/api/model settings from the generation tab
  - surfaced refine per-variant failures in the UI and result metadata
- 2026-03-09 Completed in Wave 2:
  - added `utils/config_loader.py` as the shared entry for env/local-secret/YAML loading
  - added `configs/local/*.txt` support for local API key storage, with gitignore coverage
  - aligned `ExpConfig` with provider-specific default model resolution so CLI and GUI no longer disagree by default
  - updated config template and README to document secret-file precedence and Evolink defaults
  - added `utils/demo_task_utils.py` to centralize demo task metadata, sample input construction, and result-stage key resolution
  - productized `plot` in the main Streamlit demo with task-aware inputs, plot code display, and plot ZIP export support
  - added unit tests for demo task helpers and provider-specific `ExpConfig` defaults
  - added `utils/run_report.py` and wired CLI output summaries/failure manifests into `main.py`
  - added `scripts/live_smoke_test.py` for cheap live validation against configured providers
  - fixed Gemini image-generation compatibility with the current `google-genai` SDK by removing deprecated `ImageConfig` usage and moving render hints into the prompt
  - reduced noisy Evolink missing-key logs during Gemini-only runs
  - validated live smoke on 2026-03-09 with:
    - `diagram` via `gemini-3.1-flash-lite-preview` + `gemini-3.1-flash-image-preview`: passed
    - `plot` via `gemini-3.1-flash-lite-preview`: passed
    - `diagram` via `evolink` + `nano-banana-2-lite`: blocked by missing local Evolink API key
- 2026-03-09 Completed in Wave 3:
  - added `utils/runtime_settings.py` to centralize provider defaults, local-key resolution, and provider runtime initialization for demo/CLI/tests
  - added `utils/pipeline_state.py` to centralize pipeline key naming, final-stage resolution, critic-round discovery, and render options
  - added `utils/result_paths.py` so viewers share one GT path-resolution strategy
  - refactored `demo.py` and `scripts/live_smoke_test.py` to use shared runtime-settings helpers instead of hand-rolled provider initialization
  - refactored `VisualizerAgent` and `VanillaAgent` to use `BaseAgent.call_text_api` / `call_image_api`, removing duplicated provider branching
  - fixed the remaining Gemini image-generation regression in `BaseAgent.call_image_api` and `VanillaAgent` by removing old `ImageConfig` assumptions
  - added structured plot execution diagnostics in `utils/plot_executor.py`, and surfaced them to `CriticAgent` when plot rendering fails
  - added cached reference metadata/image loading in `PlannerAgent`
  - added cached style-guide loading in `PolishAgent`, and short-circuited polish when the model says `No changes needed`
  - fixed `PolishAgent` to actually use its dedicated polish system prompt during image generation
  - updated both visualizers to consume shared final-stage and GT-path helpers
  - added focused tests for runtime settings, pipeline state helpers, base image API routing, and plot execution diagnostics
  - revalidated live smoke on 2026-03-09 after the refactor:
    - `diagram` via `gemini-3.1-flash-lite-preview` + `gemini-3.1-flash-image-preview`: passed (`results/smoke/diagram/20260309_210722_gemini_diagram.json`)
    - `plot` via `gemini-3.1-flash-lite-preview`: passed (`results/smoke/plot/20260309_210752_gemini_plot.json`)
    - `diagram` via `evolink` + `nano-banana-2-lite`: still blocked by missing local Evolink API key
- 2026-03-09 Completed in Wave 4:
  - added `utils/pipeline_registry.py` so supported `exp_mode` values are declared in one registry instead of hard-coded across call sites
  - refactored `PaperVizProcessor.process_single_query()` to execute registry-driven stage specs instead of one large `if/elif` block
  - added `utils/concurrency.py` and replaced the old no-op `auto` concurrency logic with workload-aware heuristics
  - fixed `main.py --exp_mode` default from invalid `dev` to valid `dev_full`, and wired CLI choices to the registry
  - added shared task-type detection for viewers via `utils.pipeline_state.detect_task_type_from_result`
  - revalidated the post-registry refactor on 2026-03-09 with:
    - `diagram` via `gemini-3.1-flash-lite-preview` + `gemini-3.1-flash-image-preview`: passed (`results/smoke/diagram/20260309_211635_gemini_diagram.json`)
    - `plot` via `gemini-3.1-flash-lite-preview`: passed (`results/smoke/plot/20260309_211703_gemini_plot.json`)
  - added focused tests for pipeline registry, processor registry execution, and auto concurrency heuristics
- 2026-03-09 Completed in Wave 5:
  - upgraded the refine tab to run jobs in a background thread so the page is no longer blocked by synchronous `asyncio.run(...)`
  - added user-visible stop support for refine jobs via cooperative cancel events and retry-loop cancellation checks
  - preserved refine progress, recent status lines, provider/model metadata, and finished outputs through session state
  - added unit tests for background refine job completion and cancellation without requiring real provider calls
  - pointed `requirements.txt` at editable package install so `pyproject.toml` becomes the dependency source of truth for normal installs

- 2026-03-09 Deferred detail:
  - refine cancellation is cooperative: it can stop future retries and pending variants, but it cannot interrupt a single provider request already in flight.
  - dependency versions are still not locked; environment reproduction is improved, but not yet fully pinned.
