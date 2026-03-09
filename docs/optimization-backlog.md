# Optimization Backlog

## In Progress
- [x] Wave 1: pipeline correctness and round-consistency fixes
- [x] Wave 1: CLI parameter parity improvements
- [x] Wave 2: bounded and cancellable refine workflow
- [x] Wave 2: shared runtime settings/provider defaults
- [x] Wave 2: typed pipeline state and shared result-path helpers

## Next Up
- [x] Add portable result-path resolution across viewers
- [x] Add richer batch summaries and failure manifests for CLI/demo exports
- [x] Add true user-visible cancel control for refine tasks via background execution
- [x] Unify dataset-aware asset and reference path resolution across CLI, demo, evaluation, and viewers

## Later
- [x] Replace monolithic mode branching with a pipeline registry
- [x] Add structured plot execution diagnostics
- [x] Add cache layers for reference metadata and style guides
- [x] Make viewers and smoke artifacts share one result-file schema/loader
- [x] Add a run manifest/bundle format for portable result sharing across custom datasets
- [ ] Unify dependency management around a single locked source

## 2026-03-10 Roadmap
- [x] Phase 8: Result stability and run identity
  - [x] assign a stable `input_index` / fallback `candidate_id` for every input sample
  - [x] make `PaperVizProcessor` yield/save results in deterministic input order even under concurrent execution
  - [x] make demo display / download / ZIP export use stable candidate identifiers instead of grid indexes
  - [x] make run filenames more collision-resistant and more self-describing
- [x] Phase 9: Contract and registry consistency
  - [x] attach richer pipeline metadata (`pipeline_spec`, final-stage hints) to produced results
  - [x] remove hard-coded `demo_full` / planner-vs-stylist assumptions from final-stage resolution and stage timelines
  - [x] make viewers render stage labels and auto mode selection from registry metadata instead of guesswork
- [x] Phase 10: Runtime isolation
  - [x] replace module-level provider/client/hook globals with per-run runtime context objects
  - [x] isolate generation runtime from refine runtime so concurrent sessions do not stomp each other
  - [x] centralize agent/provider shutdown and resource cleanup
- [ ] Phase 11: Product-loop closure
  - [ ] move candidate generation onto background jobs with cancel / status / resume semantics
  - [ ] add candidate-to-refine and plot-code-to-rerender entry points in the demo
  - [ ] add history/replay for saved manifests and bundles in the demo
- [ ] Phase 12: Experience parity and scale
  - [ ] align GUI and CLI on `manual` retrieval and `max_critic_rounds=0`
  - [ ] add plot input parsing / validation / preview before sending content to the planner
  - [ ] make plot `manual` retrieval explicit or disable it cleanly until implemented
  - [ ] replace full-prompt retrieval selection with prefilter + rerank for large datasets
  - [ ] expand viewer/demo integration coverage and refresh README usage docs
