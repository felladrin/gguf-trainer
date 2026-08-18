// Every command the CLI exposes, in the order they appear in `help`.
//
// Adding a command means adding its module here and nowhere else: the help
// text, the JSON schema and the dispatcher all read this list.

import type { Command } from "./args.ts";
import { corpusCommand } from "../commands/corpus.ts";
import { tokenizeCommand } from "../commands/tokenize.ts";
import { chatCorpusCommand } from "../commands/chat-corpus.ts";
import { finetuneCommand, pretrainCommand } from "../commands/pretrain.ts";
import { exportCommand } from "../commands/export.ts";
import { inspectCommand } from "../commands/inspect.ts";
import { generateCommand } from "../commands/generate.ts";
import { evalLossCommand } from "../commands/eval-loss.ts";
import { evalChoiceCommand } from "../commands/eval-choice.ts";
import { demoCommand } from "../commands/demo.ts";
import { archsCommand } from "../commands/archs.ts";
import { styleSeedCommand } from "../commands/style-seed.ts";
import { styleRestyleCommand } from "../commands/style-restyle.ts";

export const COMMANDS: Command[] = [
  corpusCommand,
  tokenizeCommand,
  chatCorpusCommand,
  styleSeedCommand,
  styleRestyleCommand,
  pretrainCommand,
  finetuneCommand,
  evalLossCommand,
  evalChoiceCommand,
  generateCommand,
  inspectCommand,
  exportCommand,
  demoCommand,
  archsCommand,
];
