<p align="center">
  <img src="./apps/desktop/assets/icon.png" width="128" alt="Arbrellis icon">
</p>

<h1 align="center">Arbrellis</h1>

<p align="center">
  Turn the ideas you choose to remember into a private, visual knowledge graph.
</p>

<p align="center">
  <a href="https://github.com/ad-st/arbrellis/releases/latest">Download</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="./CONTRIBUTING.md">Contributing</a>
</p>

> **Early preview:** Linux x86-64 is currently supported. macOS is planned; Windows remains
> experimental. Keep backups of important data.

## Why Arbrellis?

Documents and AI chats contain more information than you want to keep. Arbrellis lets you select the
small pieces that matter and shows how they relate to knowledge you saved earlier.

The graph is not a predefined topic tree and the application does not decide what you must learn. It
grows from your own selections. Labels remain editable, nodes remain movable, and every graph item
keeps a connection to its source.

## How it works

1. **Capture** an idea directly, or highlight an exact passage in a Markdown or HTML document.
2. **Integrate** it locally. Arbrellis creates a short label and an embedding for the complete fact.
3. **Explore** the graph. Related facts move closer together while your manual node positions persist.
4. **Refine** it. Rename labels to merge or split concepts, annotate source passages, and undo graph
   integration without deleting the saved fact.

Facts sharing a label form a multi-fact concept. Opening it reveals the individual statements and
their relationships, so repeated ideas do not become a row of duplicate nodes.

## Features

- A full-window spatial graph built from your own knowledge
- Standalone facts and exact, read-only document extracts
- Persistent node and multi-fact layouts
- Editable labels with user decisions taking precedence over generated ones
- Document annotations that do not affect the graph
- Document-focused graph views and source highlighting
- Undo and redo for graph integration
- Optional agent chat with a protected working document

## Private by default

Facts, documents, annotations, graph positions, embeddings, and model inference stay on your
computer. Arbrellis uses Qwen3-0.6B for short labels and BGE-large-en-v1.5 for semantic relationships.
Their weights download when first needed and require approximately 3 GB of additional disk space.

Chat is optional. The Codex chat plugin sends only the selected conversation and working-document
context to its configured provider. Documents, facts, annotations, and the graph work without chat or
a Codex account.

## Install on Linux

Download a package from the [latest release](https://github.com/ad-st/arbrellis/releases/latest):

- **AppImage:** download `Arbrellis-*.AppImage`, make it executable, and run it.
- **Portable archive:** extract `arbrellis-*-linux-x64.tar.gz`, then run `arbrellis` inside the
  extracted directory. This option does not require FUSE.

## Your first graph

1. Double-click empty graph space and enter a useful statement.
2. Choose **Integrate** to generate its label and graph position.
3. Open **Docs**, import a document, select a passage, and choose **Extract fact**.
4. Click a node to inspect the source; click a multi-fact concept to see its individual facts.

The first integration may take longer while local models download. Later integrations reuse the
local cache and run in the background.

## Optional integrations

Agent chat requires a separately installed and authenticated Codex CLI. Arbrellis detects it at
runtime; no chat setup is needed for its other features.

T5Gemma may be enabled as an optional second label candidate after accepting Google's Gemma terms on
Hugging Face. It is not distributed with Arbrellis, and the default Qwen and BGE pipeline remains
fully functional without it.

## Project status

Arbrellis is an early personal-knowledge experiment, not yet a stable data archive. The interface and
storage format may change between preview releases. Please report reproducible problems through
[GitHub Issues](https://github.com/ad-st/arbrellis/issues).

Development setup, testing, packaging, and release instructions live in
[CONTRIBUTING.md](./CONTRIBUTING.md).

## License

Arbrellis source code is licensed under the [Apache License 2.0](./LICENSE). Downloaded models retain
their own licenses and usage terms.
