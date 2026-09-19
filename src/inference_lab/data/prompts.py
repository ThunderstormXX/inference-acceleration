"""One shared tokenization protocol for all frameworks."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from inference_lab.core.config import BenchmarkConfig


@dataclass
class PromptSample:
    index: int
    prompt_tokens: list[int]
    original_prompt_tokens: int
    problem: str
    teacher_response_chars: int

    @property
    def token_sha256(self):
        return hashlib.sha256(json.dumps(self.prompt_tokens).encode()).hexdigest()


class PromptDataset:
    def __init__(self, config: BenchmarkConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def load(self) -> list[PromptSample]:
        rows = [json.loads(line) for line in Path(self.config.dataset_path).read_text().splitlines() if line]
        if len(rows) < self.config.count:
            raise ValueError(f"Need {self.config.count} rows, found {len(rows)}")
        samples = []
        for index, row in enumerate(rows[:self.config.count]):
            problem = row["problem"]
            if not isinstance(problem, str) or not problem.strip():
                raise ValueError(f"Empty problem at row {index}")
            content = problem + "\nLet's think step by step and output the final answer within \\boxed{}."
            tokens = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=True,
                add_generation_prompt=True, enable_thinking=True, return_dict=False,
            )
            tokens = list(tokens)
            if not tokens or any(type(token) is not int or token < 0 for token in tokens):
                raise ValueError("Tokenizer must return a nonempty list of integer token IDs")
            if self.config.prompt_mode == "chain-prefix":
                chain = row["response"]
                # The Qwen template already opens <think>; retain the chain body.
                if chain.lstrip().startswith("<think>"):
                    chain = chain.lstrip()[len("<think>"):]
                tokens += self.tokenizer.encode(chain, add_special_tokens=False)[:self.config.chain_prefix_tokens]
            original = len(tokens)
            # Never silently change a problem by truncating its instruction/template.
            if original > self.config.max_prompt_tokens:
                raise ValueError(f"Row {index}: {original} prompt tokens > max_prompt_tokens={self.config.max_prompt_tokens}; raise the limit explicitly")
            samples.append(PromptSample(index, tokens, original, problem, len(row.get("response", ""))))
        return samples
