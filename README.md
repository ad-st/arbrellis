# Arbrellis

Arbrellis is a local-first personal knowledge workspace. Facts extracted from documents and working
conversations become a spatial graph that remains under the user's control.

> **Status:** early preview. Linux x86-64 is the first supported release target. macOS is next;
> Windows remains experimental.

## What stays local

Documents, facts, annotations, graph positions, embeddings, and model inference remain on the local
computer. Chat is an optional plugin: the Codex plugin sends only the conversation and working
document context needed for the selected chat.

## Local models

The default pipeline uses Qwen3-0.6B for labels and BGE-large-en-v1.5 for embeddings. Model weights
are downloaded on first use and require approximately 3 GB of additional disk space.

T5Gemma is optional because its files require accepting Google's Gemma terms on Hugging Face. After
accepting those terms and authenticating locally, enable it with:

```sh
ARBRELLIS_ENABLE_T5GEMMA=1 npm run dev
```

Without T5Gemma, labels and multi-fact captions continue to work using Qwen and BGE.

## Development

Requirements: Node.js 22.12+, Python 3.12+, and an authenticated Codex CLI only when testing chat.

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements/runtime.txt
source .venv/bin/activate
npm install
npm run dev
```

On Windows, activate `.venv\Scripts\python.exe` instead. Override interpreter discovery with
`ARBRELLIS_PYTHON=/absolute/path/to/python`.

The first integration downloads Qwen3-0.6B and BGE-large-en-v1.5 into the user's model cache.
No model weights or personal databases are stored in this repository.

## Install on Linux

Download a Linux package from the latest GitHub release:

- `arbrellis_*_amd64.deb` integrates with Debian and Ubuntu package management.
- `Arbrellis-*.AppImage` is the single-file application. Make it executable and run it.
- `arbrellis-*-linux-x64.tar.gz` is a FUSE-free archive. Extract it and run `arbrellis`.

Arbrellis currently targets x86-64 Linux. The application is an early preview, so keep backups of
important knowledge databases.

## Optional chat and T5Gemma

Chat requires a separately installed and authenticated Codex CLI. Arbrellis detects it at runtime;
the rest of the application does not depend on it.

T5Gemma is not redistributed. To opt in, accept the model's Gemma terms on Hugging Face, authenticate
on the machine running Arbrellis, and set `ARBRELLIS_ENABLE_T5GEMMA=1` before launching it. Qwen is
used for both labels and fact captions when this option is off.

## Verification

```sh
npm run check
```

Run the deterministic browser UI suite explicitly during interface development:

```sh
npx playwright install chromium   # once per development machine
npm run test:ui
npm run test:ui:headed            # watch the interactions
```

UI tests use a disposable fixture database and do not download models or touch the user's Arbrellis
data. Failed tests retain a screenshot, video, and Playwright trace under `test-results/`.

## Linux package

```sh
.venv/bin/python -m pip install -r requirements/build.txt
npm run build:server
npm run dist:linux
```

The Linux build produces an AppImage, a Debian package, a FUSE-free archive, and `SHA256SUMS`.
Tag-triggered releases are published only after a second clean GitHub runner verifies checksums,
tests the frozen backend, installs the `.deb`, and opens the installed desktop under Xvfb. Running
the release workflow manually performs the same build and smoke tests without publishing a release.

Installers and frozen model runtimes are release artifacts and are never committed to Git.

## License

Arbrellis source code is licensed under the [Apache License 2.0](LICENSE). Downloaded models retain
their own licenses and usage terms.
