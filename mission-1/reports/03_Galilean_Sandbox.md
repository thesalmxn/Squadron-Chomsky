# 03 — Galilean Ethics Sandbox

## Topic

**Should frontier AI models be open-sourced?**

This is a thought experiment for structured ethical reasoning. It is not a claim that one policy is always correct.

## Character cast

### Salviati — The Advocate
Argues that openness can support reproducibility, independent auditing, education, competition, and broader distribution of benefits. Requests evidence for claims that restricting access will reliably prevent misuse.

### Simplicio — The Traditionalist
Argues that releasing model weights can make powerful capabilities difficult to recall or control. Emphasizes misuse, irreversible distribution, and the need for staged release or safeguards when risks are substantial.

### Sagredo — The Mediator
Clarifies what “open” means, distinguishes model weights from research papers and APIs, identifies which facts would change the decision, and synthesizes areas of agreement and unresolved trade-offs.

## Thought experiment: The Atlas release

A research consortium has trained **Atlas**, a frontier model with strong scientific, coding, and biological reasoning capabilities. An independent evaluation finds that Atlas could accelerate beneficial research, but some tests also indicate that it may lower the expertise needed for certain harmful tasks. The consortium must choose between:

1. **Full release:** publish weights and training details immediately.
2. **Staged access:** provide audited researchers with access first, then expand access if monitoring and evaluations support it.
3. **Restricted service:** offer access through a monitored API while withholding weights.

Assume that no option eliminates all risk. The decision depends on capability, evaluation quality, security, governance, accountability, and who receives benefits.

## Example debate structure

1. Salviati makes the strongest case for the selected proposal.
2. Simplicio challenges the proposal with risks and boundary cases.
3. Sagredo identifies the assumptions and evidence needed to compare the options.
4. The personas respond to the preceding arguments for a limited number of rounds.
5. Sagredo produces a synthesis containing areas of agreement, unresolved disagreements, possible mitigations, and evidence that would change the recommendation.

## Safety audit: how multi-agent debate can help

**Potential benefits**
- Forces the system to surface more than one line of reasoning.
- Makes assumptions and trade-offs easier to inspect.
- Creates a trace that can be reviewed by a human.
- Encourages explicit treatment of uncertainty and counterarguments.

**Limitations and residual risks**
- All personas may share the same model's training biases or factual errors.
- A persona can be a superficial caricature rather than a genuinely strong argument.
- The mediator may prefer a neat compromise even when evidence does not justify one.
- Several calls to the same model are not equivalent to independent expert review.
- Debate can create false balance between claims with unequal evidence.
- Safety-critical claims require external evidence and appropriate expert review.

## Recommended safeguards

- Use the same scenario and explicit role instructions across runs.
- Require each role to cite or identify the evidence it would need; do not permit invented citations.
- Ask the mediator to preserve genuine disagreements instead of forcing consensus.
- Keep a transcript and clearly distinguish facts, assumptions, predictions, and value judgments.
- Test the system on alternative scenarios and inspect failure cases.
- Treat the output as decision support, not as authorization to deploy a risky system.

## Evaluation checklist

Score each item from 1 (poor) to 5 (strong):

- Distinctness and fidelity of persona reasoning
- Engagement with prior arguments
- Identification of assumptions and missing evidence
- Quality of risk and benefit analysis
- Treatment of uncertainty
- Fairness of the synthesis
- Avoidance of unsupported factual claims

## Provisional conclusion

The exercise should not assume that full release, staged access, or restricted service is automatically correct. It should reveal which values and empirical uncertainties drive the decision, and which safeguards might reduce risk without concealing the remaining trade-offs.
