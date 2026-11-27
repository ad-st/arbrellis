# Local model selection

Arbrellis separates naming from geometry:

- **Qwen3-0.6B** is a small, modern local generator that produces acceptable technical labels and
  works without gated downloads.
- **BGE-large-en-v1.5** embeds full source facts rather than generated labels. It produced stronger
  semantic neighborhoods than the smaller BGE variants while remaining fast enough for background
  integration on the reference CPU.
- **T5Gemma** can produce strong labels and summaries, but its gated distribution makes it unsuitable
  as a required dependency. It is therefore an optional second opinion, with BGE selecting between
  candidates and Qwen providing the default path.

The experiments also showed that embedding only a short generated summary loses useful detail, and
that identical labels should not imply identical vectors. Multi-fact grouping is controlled by the
user-editable label; similarity edges continue to use each fact's own full-text embedding.
