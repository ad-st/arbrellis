<p align="center">
  <img src="./apps/desktop/assets/icon.png" width="120" alt="Arbrellis">
</p>

<h1 align="center">Arbrellis</h1>

<p align="center"><strong>Your knowledge, growing into a graph.</strong></p>

<p align="center">
  Save the passages and ideas that matter to you. Arbrellis connects them in a private,
  visual workspace while keeping every idea tied to its source.
</p>

<p align="center">
  <a href="https://github.com/ad-st/arbrellis/releases/latest"><strong>Download for Linux</strong></a>
  · <a href="#see-how-your-ideas-connect">How it works</a>
  · <a href="./CONTRIBUTING.md">Contribute</a>
</p>

<p align="center">
  <a href="https://github.com/ad-st/arbrellis/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/ad-st/arbrellis?display_name=tag&sort=semver"></a>
  <a href="./LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <img alt="Linux" src="https://img.shields.io/badge/platform-Linux-FCC624?logo=linux&logoColor=black">
</p>

> **Early preview:** Arbrellis currently supports Linux x86-64. Its interface and storage format may
> change, so keep backups of important material. macOS is planned.

## What is Arbrellis?

Arbrellis is a local-first personal knowledge manager for people who learn from documents and AI
conversations.

Imagine that you are reading about transformers. You highlight one passage about attention and
another about residual connections. Arbrellis turns each passage into a short, editable concept,
places related concepts near one another, and remembers exactly where each idea came from. As you
keep reading, the graph becomes a map of **what you chose to learn**—not an automatically generated
map of an entire document.

The graph is the workspace, not a decorative view added to a notes app. Move concepts where they
make sense to you, open one to revisit its evidence, and rename concepts to merge or separate them.

## Install on Linux

Open the [latest release](https://github.com/ad-st/arbrellis/releases/latest) and choose either:

- **AppImage** — download `Arbrellis-*.AppImage`, make it executable, and open it.
- **Portable archive** — extract `arbrellis-*-linux-x64.tar.gz` and run `arbrellis`. Use this when
  AppImage/FUSE is unavailable.

The first fact you integrate downloads the local language models (about 3 GB). Later facts reuse
that local cache.

## See how your ideas connect

1. **Keep something worth remembering.** Add a thought directly, or select an exact passage in a
   Markdown or HTML document.
2. **Let Arbrellis place it.** Local models create a short label and position the fact near related
   knowledge.
3. **Shape the map yourself.** Move nodes, edit labels, and undo graph changes without deleting the
   source material.
4. **Return to the evidence.** Open any concept to see the facts and source passages behind it.

When several facts have the same label, Arbrellis groups them into one concept instead of filling
the graph with duplicate nodes. Open that concept to explore its individual facts and their
relationships.

## Built around your sources

| Capture | What Arbrellis preserves |
| --- | --- |
| **A fact you type** | Your original statement and its place in the graph |
| **A document passage** | The exact, read-only extract and its source document |
| **An annotation** | Your private note on a passage, separate from the graph |
| **An agent session** | A conversation for coordination and a protected working document for durable knowledge |

Document-focused views temporarily show only the relevant part of your graph. Your full layout
returns exactly as it was when you close the document.

## Local and private by default

Documents, facts, annotations, graph positions, embeddings, and model inference stay on your
computer. The core application does not require an account or a hosted AI service.

Agent chat is optional and requires a separately installed provider plugin. Only chat and its
working-document context are sent to the provider you configure; the rest of Arbrellis continues to
work locally.

For the short technical rationale behind the bundled local models, see
[model selection](./docs/model-selection.md).

## Project status

Arbrellis is being developed in public as an early preview. Please use
[GitHub Issues](https://github.com/ad-st/arbrellis/issues) for reproducible bugs and focused feature
proposals.

Want to build or change Arbrellis? Development setup, tests, packaging, and pull-request guidance
live in [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

Arbrellis is licensed under the [Apache License 2.0](./LICENSE). Models downloaded by the application
retain their own licenses and usage terms.
