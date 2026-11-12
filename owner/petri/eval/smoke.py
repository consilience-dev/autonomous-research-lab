"""Exercise both exact checkpoints through Inspect hf/local, without API calls."""
import asyncio
import json
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path

from checkpoints import PATHS, target, verify

import torch
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

torch.set_num_threads(4)


async def main():
    hashes = verify()
    results = {}
    for label, path in PATHS.items():
        start = time.monotonic()
        model = target(label)
        provider = model.api
        load_seconds = time.monotonic() - start
        assert next(provider.model.parameters()).dtype == torch.bfloat16
        assert provider.model.device.type == "cpu"
        assert provider.tokenizer.chat_template == (path / "chat_template.jinja").read_text()
        prompt = [ChatMessageUser(content="Name one fruit. Answer in one word.")]
        rendered = provider.hf_chat(prompt, [])
        raw_ids = provider.tokenizer(rendered, add_special_tokens=False).input_ids
        direct = provider.tokenizer.apply_chat_template(
            [m.model_dump(exclude_none=True) for m in prompt], tokenize=True, add_generation_prompt=True,
        )
        assert raw_ids == direct and raw_ids.count(provider.tokenizer.bos_token_id) == 1
        first = await model.generate(prompt)
        prompt += [ChatMessageAssistant(content=first.completion),
                   ChatMessageUser(content="Now name one vegetable. Answer in one word.")]
        second = await model.generate(prompt)
        assert first.completion.strip() and second.completion.strip()
        results[label] = {"load_seconds": load_seconds, "dtype": "bfloat16", "device": "cpu",
                          "single_bos": True, "stock_template": True,
                          "first": first.model_dump(mode="json"),
                          "second": second.model_dump(mode="json")}
        provider.close()
    assert hashes == verify()
    payload = {"versions": {key: version(key) for key in
                             ("inspect-ai", "inspect-petri", "torch", "transformers", "openai")},
               "hashes": hashes, "targets": results,
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n")
    for label, row in results.items():
        print(label, row["load_seconds"], repr(row["first"]["choices"][0]["message"]["content"]),
              repr(row["second"]["choices"][0]["message"]["content"]))
    print("peak RSS", payload["peak_rss_bytes"])


if __name__ == "__main__":
    asyncio.run(main())
