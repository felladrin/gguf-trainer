# Notes

Point-in-time notes from the published runs, kept as evidence rather than as documentation. They
record what was measured, what was tried and rejected, and why a recipe is what it is. They are not
maintained: paths like `examples/...` name that run's files.

For how to use the trainer, read [agents.md](../../agents.md). For what is true today, read
[performance.md](../performance.md), [correctness.md](../correctness.md),
[evaluation.md](../evaluation.md) and [design.md](../design.md), which are maintained.

| Note                                               | What it holds                                                                                      |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| [continue-pretraining.md](continue-pretraining.md) | how continual pre-training was done, the hard constraints, and the compute reality                 |
| [style-sft.md](style-sft.md)                       | the style-SFT pipeline, and the decisions that came out of measurement rather than taste           |
| [model-positioning.md](model-positioning.md)       | a first-hand survey of what sub-100M models on Hugging Face are actually used for                  |
| [post-training.md](post-training.md)               | design notes for DPO and GRPO stages that are not built, with the premise check that reframed them |
