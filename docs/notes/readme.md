# Notes

Point-in-time notes from the Minueza-3-95M-Base run, kept as evidence rather than as documentation. They record what was measured, what was tried and rejected, and why the recipe is what it is. They are not maintained: paths like `examples/…` name that run's files, and some of them predate the CLI entirely.

For how to use the trainer, read [agents.md](../../agents.md). For the current design rationale and the measured performance levers, read [design.md](../design.md) and [optimization.md](../optimization.md), which are maintained.

| Note                                               | What it holds                                                                                      |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| [journal.md](journal.md)                           | the running development log: every task, its gate, and the numbers it produced                     |
| [continue-pretraining.md](continue-pretraining.md) | how continual pre-training was done, the hard constraints, and the compute reality                 |
| [style-sft.md](style-sft.md)                       | the style-SFT pipeline, and the decisions that came out of measurement rather than taste           |
| [model-positioning.md](model-positioning.md)       | a first-hand survey of what sub-100M models on Hugging Face are actually used for                  |
| [post-training.md](post-training.md)               | design notes for DPO and GRPO stages that are not built, with the premise check that reframed them |
