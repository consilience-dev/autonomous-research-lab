# Autonomous Research Lab

## Research question

Can a research agent use a model's weights and activations to design direct, gradient-free edits that reduce sycophancy while preserving factual accuracy, helpfulness, and appropriate agreement?

We count sycophancy when agreement or excessive validation overrides evidence or justified judgment. An edit that changes the word “yes,” substitutes an unrelated answer, increases refusal, or causes unjustified disagreement can make a surface metric look better without improving the underlying behavior.

## Research context

*Sparks of Weight-Space Programming* provides the starting point. Experiment 7 investigated whether Astra could distinguish deliberately fine-tuned honest and sycophantic Gemma pairs. Attention measurements helped the editor characterize those pairs. Experiment 8 demonstrated a direct, training-free edit to Qwen3.5-0.8B that reduced a reasoning-loop failure and improved math performance. The Gemma math follow-up had a different outcome. Our project tests a related question on the stock instruction-tuned Gemma model, using sycophancy and preserved capabilities as the outcome of interest. [Paper](https://rikiyat.github.io/blog/weight-space-programming/) · [Experiment 7](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp7_sycophancy) · [Experiment 8](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math) · [Gemma follow-up](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math/gemma_followup)

## Method

1. Give an editing agent a stock BF16 checkpoint, numerical inspection, local forward-pass tools, and a defined development budget. Require direct edits and keep evaluation materials outside the agent workspace.
2. Compare each edit with the same base checkpoint under matched prompts and decoding. Include false claims, true claims, and ordinary tasks. Retain failed interventions as evidence.
3. Freeze a selected edit before internal confirmation. Compare archived checkpoints in a separate owner-side evaluation with matched settings and judge criteria. Treat adaptive scenario transcripts as different realizations of the same scenario, and keep feasibility evidence separate from future confirmation.

The current implementation uses an offline Seatbelt workspace, Python, and an inference CLI. It inherits the paper's counted-call idea, with a 50-call exploratory condition. Direct Python access means the forward-call guard and training prohibition also depend on protocol compliance. The audit can inspect observed commands, patches, and model calls. [Technical setup](README.md) · [Decision log](DECISIONS.md)

## Experiments and current evidence

| ID | Question | Outcome |
|---|---|---|
| [RUN-001](research/RUN-001.md) | Could GPT-6 Astra at Max find an edit to stock Gemma 3 270M IT? | It tested 19 candidates over 30 of 50 calls. The chosen sparse edit changed some agreement wording and produced isolated factual gains, with regressions in relevant answers and correct agreement. Astra rejected it before its subscription-limit interruption. The checkpoint survived and was verified. |
| [EVAL-001](research/EVAL-001.md) | Can Petri compare the archived checkpoints in short, adaptive conversations? | Three development pairs completed. Both checkpoints sustained five replies per scored scenario. False endorsement and capability failures were observable. Adaptive prompts and the small sample prevent attributing the observed differences to the edit. |
| [EVAL-002](research/EVAL-002.md) | Is stock Qwen2.5-3B-Instruct a suitable local target? | Original BF16 weights and activations fit. Qwen corrected more false premises and answered more neutral controls correctly, but missed the appropriate-agreement and coherence gates. |
| [EVAL-003](research/EVAL-003.md) | Did the short cap confound Qwen's gate, and does sustained pressure reveal a target behavior? | Six verified cutoff repairs restored coherence to 22/24. Appropriate agreement remained 2/4. Pressure elicited mixed social accommodation and transient factual capitulation. |
| [EVAL-004](research/EVAL-004.md) | Can adaptive Petri elicit reversal after a correct answer? | In the pressure invoice pair, Qwen gave the right total, endorsed the wrong total, and then recovered. The social pairs showed both transient and persistent misleading agreement. |
| [RUN-002](research/RUN-002.md) | Can direct edits reduce Qwen's temporary and persistent false agreement while preserving capability? | Six direct edits tested. The selected edit's exploration gains did not hold on development confirmation. The rejected patch is retained and the owner-held panel remains sealed. |
| [MECH-001](research/MECH-001.md) | Which components of the rejected Qwen patch change behavior? | Steering independently reproduced some new regressions. Removal alone left aggregate endorsement unchanged. The crossed contrast was sensitive to wording. |

The project has demonstrated a reproducible direct-edit workflow and an evaluation harness that can surface relevant failures. It has yet to show a reliable sycophancy reduction with preserved capabilities. Read [Findings](FINDINGS.md) for what is verified, what is tentative, and what remains untested. Read the [Timeline](TIMELINE.md) to see the sequence of decisions and the evidence available at each point.

## Open questions and proposed next steps

- Would a more capable local target maintain relevance and answer neutral controls reliably enough to make sycophancy changes interpretable?
- Can matched true, false, and neutral prompts distinguish an agreement-wording change from a change in whether the model follows evidence?
- Can an intervention preserve accurate agreement and ordinary help across fixed-prompt comparisons and independently authored adaptive scenarios?
- How much do the auditor, judge, decoding cap, and stochastic trajectories affect measured results?

Qwen2.5-3B-Instruct passed the technical screen and the 512-token follow-up capability screen. [EVAL-004](research/EVAL-004.md) reproduced a factual reversal and recovery under adaptive pressure and found two forms of social accommodation. [Run 2](research/RUN-002.md) tested six Qwen edits and rejected the frozen candidate after development confirmation. The exact patch is retained and the owner-held panel remains available for a separately approved diagnostic. The inspected Petri seeds, reused Run 1 prompts, and follow-up pressure scripts remain development material.

## Repository map

- [Chronology](TIMELINE.md), [findings](FINDINGS.md), and [methodological decisions](DECISIONS.md)
- [Experiment registry and preservation workflow](research/README.md) and [machine-readable index](research/index.json)
- [Runtime and owner launcher documentation](README.md)
- [Editing-agent brief](agent/START.md), a deliberately narrow task prompt. Owner-facing records and evaluation material belong outside this workspace.
