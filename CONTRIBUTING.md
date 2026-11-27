# Contributing to Arbrellis

Arbrellis is an early-stage project. Open an issue before substantial behavior or design changes,
and never commit user documents, databases, credentials, model caches, or build output.

By contributing, you agree that your contribution is licensed under Apache-2.0.

## Requirements

- Node.js 22.12 or newer
- Python 3.12
- An authenticated Codex CLI only when testing chat

## Development setup

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements/runtime.txt
source .venv/bin/activate
npm install
npm run dev
```

On Windows, activate `.venv\Scripts\activate` instead. Set
`ARBRELLIS_PYTHON=/absolute/path/to/python` to override interpreter discovery.

Qwen3-0.6B and BGE-large-en-v1.5 download on the first integration. Model weights and personal
databases do not belong in this repository. To test T5Gemma after accepting its Hugging Face terms
and authenticating locally, launch with `ARBRELLIS_ENABLE_T5GEMMA=1`.

## Verification

Run the source checks before submitting a change:

```sh
npm run check
```

For interface changes, install Playwright's browser once and run the deterministic UI suite:

```sh
npx playwright install chromium
npm run test:ui
npm run test:ui:headed
```

The suite uses a disposable fixture database, downloads no models, and does not touch personal
Arbrellis data. Failed tests retain screenshots, video, and a Playwright trace under
`test-results/`. Include a short recording or before/after screenshots with visible interface
changes.

## Linux packages

```sh
.venv/bin/python -m pip install -r requirements/build.txt
npm run build:server
npm run dist:linux
```

The build produces an AppImage, Debian package, portable archive, and `SHA256SUMS`. Installers and
frozen runtimes are release artifacts and must not be committed.

The [Linux release workflow](./.github/workflows/release-linux.yml) builds on Ubuntu, verifies
checksums on a clean runner, tests the frozen backend, installs the `.deb`, and opens the installed
application under Xvfb. A manual workflow run builds and tests without publishing. A `v*` tag
publishes only after the smoke job succeeds.

## Pull requests

1. Explain the user-visible behavior or design decision.
2. Keep the change focused and preserve unrelated local work.
3. Run `npm run check` and any relevant UI tests.
4. Include a short recording or before/after screenshots for interface changes.
