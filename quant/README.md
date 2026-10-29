# Quant profile

This directory ports the upstream `program.md` / editable `train.py` / fixed
`prepare.py` workflow to Quant Lab. The root GPU language-model profile and its
history remain available. Start the Quant agent with this directory's program.

The adapter invokes Quant Lab's existing attempt runner and strict assessor,
not a reimplementation of its returns or checks. Set `QUANTLAB_REPO` to the
locked research checkout containing `scripts/autoresearch.py` and
`scripts/assess_candidate.py`, `QUANTLAB_HOME` to the study with its inherited
`runs/`, and `QUANTLAB_PROTOCOL` to the frozen study protocol. Run with Quant
Lab's locked Python interpreter. No API key is needed by this evaluator.

`train.py` starts from the existing M1 causal six-factor ridge candidate. Its
replay is integration evidence, not another independent alpha. The engine's
known future-bar dependency remains unresolved, so every result explicitly has
`promotion_eligible=false`. No submit/deploy/trade command is reachable here.

The companion [Harness² fork](https://github.com/alphaeveryday/harness2/tree/feature/quant-lab)
executes the original recursion driver with this profile and loads the original
[Scientific Agent Skills fork](https://github.com/alphaeveryday/scientific-agent-skills/tree/feature/quant-lab)
into its working contexts. Run artifacts belong outside Git.

Upstream base: karpathy/autoresearch `228791fb499afffb54b46200aca536f79142f117`.
