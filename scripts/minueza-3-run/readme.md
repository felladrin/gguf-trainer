# The Minueza-3-95M-Base run

The exact scripts that trained
[Minueza-3-95M-Base](https://huggingface.co/Felladrin/Minueza-3-95M-Base), kept as provenance rather
than as tooling. They hardcode that run's paths, step counts and log files, and they predate the
CLI, so they call the old `examples/*.ts` scripts.

Do not use them as a starting point. `agents.md` has the current recipes, and every workflow they
performed is now a subcommand:

| These scripts did                        | Now                                                      |
| ---------------------------------------- | -------------------------------------------------------- |
| `resume-phase-a.sh`, `resume-phase-b.sh` | `pretrain --resume`                                      |
| `stop-phase-a.sh`, `stop-phase-b.sh`     | stop the process; the last checkpoint is always loadable |
| `run-style-sft.sh`                       | `style-seed`, `style-restyle`, `chat-corpus`, `finetune` |
| `watch-restart-for-eta.sh`               | nothing; it watched one specific run's log               |

They are here for one reason: if you want to know exactly how the published model was trained,
including the stop-and-resume history, this is the record.
