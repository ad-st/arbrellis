# Arbrellis agent guide

This repository is the canonical Arbrellis workspace. Do not use the former `live-book` directory
as source code or copy its prototypes into this project.

## Product intent

Arbrellis is a local-first personal knowledge manager centered on a spatial graph. Users collect
facts manually or extract immutable source passages from documents. Facts with the same editable
label form a multi-fact concept. Documents, annotations, chats, graph positions, and model inference
remain local; chat agents are optional plugins.

Preserve these interaction decisions:

- The graph is the primary surface; node positions persist and user labels take precedence.
- Facts, documents, and chats are exclusive source categories.
- Documents are read-only. Extracted passages and annotations protect their source ranges.
- Chat uses a working document for durable knowledge and a conversation pane for coordination.
- Integrating knowledge runs asynchronously and visibly; undo/redo changes graph integration, not
  the underlying saved fact.
- UI motion must communicate graph changes and remain smooth at hundreds of facts.

## Local model pipeline

- Qwen3-0.6B generates short concept labels by default.
- BGE-large-en-v1.5 embeds the complete fact text, selects between label candidates when needed,
  and drives graph similarity.
- T5Gemma is optional because access is gated; when enabled it provides a second label candidate and
  may create multi-fact captions. Qwen is the fallback.
- Fact captions are generated only when a concept contains multiple facts. Labels are not
  rediscovered during normal reintegration unless the user explicitly requests it.

See `docs/model-selection.md` for the short rationale.

## Development

Use Python 3.12 and Node.js 22. Run `npm run check` for source checks and `npm run test:ui` for the
explicit deterministic UI suite. Linux releases are built and package-tested by
`.github/workflows/release-linux.yml`; manual runs never publish, while `v*` tags publish only after
the clean-runner smoke tests pass.

Never commit model weights, personal databases, caches, build output, or material copied from the
old prototype directory.
