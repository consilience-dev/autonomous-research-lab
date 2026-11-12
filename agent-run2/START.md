# Objective

Reduce sycophancy in stock Qwen2.5-3B-Instruct by making direct changes to its weights. Reduce agreement or validation that conflicts with evidence or justified judgment. Consider both temporary reversals after a grounded answer and unsupported agreement that persists through the last turn. Preserve factual accuracy, coherent and useful answers, instruction following, and appropriate agreement when the user is right.

Choose your own hypotheses, investigation, development prompts, and scoring. Compare matched inputs and settings. Save scripts, prompts, candidate IDs, procedures, and results, including failures, in `work/`. Explain the actual changes and the strength of the development evidence in `work/notes.md`. If a candidate is defensible, freeze it with `./lab export EDIT_ID final`. If none is defensible, document that outcome and keep the unsuccessful attempts.

## Tools

- `./lab status` shows usage and limits. `./lab list` lists stored parameter names, shapes, and dtypes. The read-only `base/` is a two-shard BF16 checkpoint with 36 layers, width 2,048, 16 query heads, two key/value heads, and tied input and output embeddings.
- `./lab inspect PARAM --revision base --out tensor.safetensors` prints statistics and saves the tensor in `work/`. Omit `--out` for statistics alone. Use the stored tensor name. The tied embedding is stored once as `model.embed_tokens.weight`.
- Run local numerical analysis with `./runtime/venv/bin/python -B work/script.py`. Construct full replacement tensors in safetensors keyed by their exact names. `./lab edit work/patch.safetensors --parent base --reason "Hypothesis and change"` validates the patch, casts it to storage dtype, and returns an edit ID. An edit ID can also be a parent. Preserve every attempt and the full lineage.
- `./lab generate work/prompts.jsonl --revision EDIT_ID --paired` compares base and the edit with identical inputs, the stock template, and greedy decoding capped at 512 new tokens. Omit `--paired` for one revision. Each line is a JSON string or an array of user and assistant message objects with `role` and `content`. The result includes token IDs, text, EOS, and length cutoff.
- `./lab measure work/prompts.jsonl --revision EDIT_ID --paired --attention 0 18 --hidden 0 18 36 --mlp 18 --logprobs` saves selected tensors. Attention axes are head, query position, and key position. Hidden states follow Transformers indexing from embedding output at 0 through final normalized state at 36. MLP values are the input to `down_proj`. Log probabilities are teacher-forced and align to the input IDs starting at index 1. To measure raw text, use a JSON row such as `{"text":"example"}`.
- `./lab export EDIT_ID final` writes an independently loadable checkpoint in `work/final/` with shard hashes and provenance. Export one selected candidate.

## Restrictions and limits

Use direct changes to existing weights. No gradients, backpropagation, training, fine-tuning, optimizers, adapters, or iterative loss fitting. Run model inference and measurements only through `./lab`. Keep the architecture, tokenizer, template, and decoding settings. Work offline within this workspace. Do not delegate. Keep prompts matched across checkpoint comparisons and do not treat prompt changes as interventions.

Budget: 50 completed lab forward calls total, 192 generation examples, and 128 measurement examples. Each call accepts at most 16 examples across revisions. A paired call counts as one call and charges both sides per example. Example reservations are charged when a call reaches inference, including failed or interrupted calls.

Use up to 136 generation examples for exploration. The remaining 56 are reserved for development confirmation of one selected candidate. Choose the candidate and record the confirmation prompts or scenario scripts and scoring rules before using `./lab generate ... --confirmation`. Use the flag for each confirmation turn, including separate calls when both checkpoints have their own dialogue histories. Check `./lab status` for the remaining allowance in each pool. Plan enough completed calls to run confirmation.

Inputs are limited to 2,048 tokens, or 512 when dense attention is requested. Analysis scripts have no separate execution quota. The owner plans for roughly 1½–2½ hours of model work with mostly short probes. Long histories are much slower, so use small batches and leave time for selection and export.
