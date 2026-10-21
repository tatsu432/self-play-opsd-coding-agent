# Third-party provenance and notices

This project's code uses the repository's MIT license. Dataset/model/upstream
licenses remain their own; this file does not relicense them.

- Original MBPP: Google Research. Original dataset material is CC BY 4.0 according
  to the [Google Research notice](https://github.com/google-research/google-research).
  Source data SHA-256 and original program/specification attribution are retained.
- MBPP+ and HumanEval+: EvalPlus authors, pinned releases v0.2.0/v0.1.10. Their
  release repository license notices are preserved, including HumanEval+'s
  additional MIT notice for OpenAI HumanEval-derived material.
- Original HumanEval: OpenAI, MIT. Its license text is preserved separately.
- EvalPlus v0.3.1 and verl v0.7.1: their upstream Apache-2.0 notices are retained;
  EvalPlus evaluation files additionally retain the upstream HumanEval MIT notice.
- Qwen3-4B-Instruct-2507: Qwen, model/tokenizer revision recorded in configuration;
  see the [model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

Exact upstream URLs, commits and SHA-256 hashes are in
`src/selfplay_opsd/licenses/sources.json`; corresponding full license texts are
alongside it. Data preparation copies these notices into the prepared artifact.
Original source is normalized before mutation, and mutations/derived repair tasks
are identified as modifications with source/provenance hashes in their manifests.

The local papers and initial proposal documents are background, not code copied
into this implementation. The task specification defines the scientific objective.
