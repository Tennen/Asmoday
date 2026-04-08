# Vision Service

Python Vision Service for Gateway's `vision_entity_stay_zone` capability.

## Scope

The service:

- exposes the entity catalog endpoint used by Gateway
- accepts desired-state sync payloads
- reads RTSP streams directly
- detects entities with Ultralytics YOLO
- tracks them with ByteTrack through supervision
- emits rule-level dwell events and evidence back to Gateway

## Quick Start

```bash
uv sync
uv run vision-service
```

Environment variables are loaded with the `VISION_SERVICE_` prefix.
