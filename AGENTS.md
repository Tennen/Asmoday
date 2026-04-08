# AGENTS

## Purpose

This repository implements a standalone Python Vision Service that integrates with Gateway through the contract defined in `vision-service-contract.md`.

The service is responsible for:

- exposing the entity catalog endpoint
- accepting desired-state rule sync from Gateway
- consuming RTSP streams directly
- detecting configured entities with Ultralytics YOLO
- tracking entities with ByteTrack
- determining whether an entity stays inside a configured zone long enough to cross the dwell threshold
- reporting status, rule-level events, and screenshot evidence back to Gateway

## Non-Negotiable Constraints

### 1. Avoid excessive fallback behavior

- Do not build broad silent fallback paths that mask integration or runtime errors.
- Prefer explicit failure with structured logs and degraded service status.
- If a required dependency is unavailable, surface the problem clearly instead of inventing alternate behavior.
- Retries are allowed for outbound callbacks, but they must be bounded, observable, and intentional.

### 2. Split files before they become large

- Any source file that approaches or exceeds 500 lines must be split by responsibility.
- Preferred split axes:
  - API routes
  - config models
  - runtime orchestration
  - detection/tracking pipeline
  - callback transport
  - tests
- Do not keep adding helpers to a large file when a module boundary is already obvious.

### 3. Small-step delivery

- Implement in small, reviewable increments.
- When a completed task can form a coherent commit, commit it immediately.
- After each meaningful commit, push it to the remote branch if a remote is available.
- Avoid bundling unrelated refactors into the same commit.
- Do not switch branches by default in this repository.
- Work directly on the current branch unless the user explicitly requests branch creation or branch switching.

## Expected Stack

- Python 3.13+
- `uv` for environment and dependency management
- FastAPI for HTTP endpoints
- Pydantic for contract models
- Ultralytics YOLO for detection
- ByteTrack for tracking
- supervision for video/geometry utilities
- OpenCV for frame access and image encoding
- pytest for tests

## Service Design Rules

### Contract fidelity

- Treat `vision-service-contract.md` as the source of truth for the Gateway integration.
- Keep payload field names stable and aligned with the contract.
- Reject incompatible schema versions rather than guessing.
- Treat config sync as full desired-state reconciliation, not patch semantics.

### Rule semantics

- Runtime behavior is rule-centric, not raw per-frame detection-centric.
- Emit `threshold_met` exactly once per dwell episode for each rule.
- Emit `cleared` once when an active threshold-met episode is no longer active.
- Do not spam repeated `threshold_met` events for the same active episode.

### Runtime model

- Gateway owns desired state; the Vision Service owns execution state.
- Each enabled rule may map to an internal pipeline worker, but shared camera ingestion is allowed if it remains clear and maintainable.
- `recognition_enabled=false` must stop recognition work cleanly.
- Missing rules in a new sync payload must be removed from runtime.

### Evidence handling

- Evidence screenshots should be associated with emitted events using stable `event_id` values.
- Capture phases are `start`, `middle`, and `end`.
- If evidence upload fails, keep logs and retry according to the configured callback retry policy.

## Code Quality

- Use type hints throughout the Python codebase.
- Keep side effects isolated behind interfaces or dedicated modules.
- Prefer deterministic units for dwell-state logic so tests do not require live video.
- Keep logging structured and include `rule_id`, `camera_device_id`, and `event_id` where relevant.
- Do not hardcode Gateway callback paths outside the synced config.

## Suggested Layout

- `pyproject.toml`
- `src/vision_service/app.py`
- `src/vision_service/api/`
- `src/vision_service/contracts/`
- `src/vision_service/runtime/`
- `src/vision_service/vision/`
- `src/vision_service/gateway/`
- `src/vision_service/settings.py`
- `tests/`

## Validation Expectations

Before considering a task complete:

- run targeted tests for the touched area
- run a broader test command when the dependency graph allows it
- ensure the service can start locally with documented configuration
- verify payload serialization for catalog, sync, status, events, and evidence callbacks

## Commit Policy

Preferred commit boundaries in this repository:

- repository policy and scaffolding
- API and contract models
- runtime orchestration
- vision pipeline and dwell-state logic
- tests and documentation

Every commit should leave the repository in a coherent state that can be reviewed independently.
