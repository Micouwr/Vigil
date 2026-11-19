# Design Notes

## Why logging is unconditional and routing uses 0.7

There is no audit threshold. All controlled decisions reaching the primary logging step receive a decision-record attempt, whatever their confidence. Tickets with confidence below 0.7 route to human review under the escalation rules.

The 0.7 routing threshold is an initial operating value, not derived from production data. It separates "uncertain" from "confident" behavior until real ticket volume is available for calibration.

## What human review catches

Four conditions route a ticket to human review:

- Compliance-flagged content
- Confidence below 0.7
- Detected PII
- Classification failures or unknown categories

The design assumption is that model errors should fail toward a person rather than toward automation.

The evaluation results demonstrated this behavior. The incorrect classifications were concentrated in PII-heavy examples, but every one of those tickets was still routed to human review. All six PII test cases were flagged successfully and escalated for manual handling.

The goal is not to eliminate mistakes. The goal is to ensure mistakes are caught before an automated action is taken.

## What I would change before production

- Replace the fixed 0.7 routing threshold with a value derived from production data and measured review outcomes.
- Expand the evaluation dataset with a larger and more representative sample of real-world tickets.
- Use independent reviewers to label evaluation data instead of relying solely on my own classifications.
- Add operational monitoring for accuracy drift, latency, routing rates, review volume, and system errors.
- Define ownership and SLAs for the human review process, including escalation paths and backlog handling.
- Add resilience for model outages, rate limits, and API failures so tickets can still be processed safely when the model is unavailable.
- Track false positives and false negatives separately, especially for compliance and PII detection, since they have different operational risks.
- Periodically re-evaluate model performance as ticket types and user behavior change over time.

## Conclusion

The evaluation results are less important than the failure mode. While the model did make classification mistakes, the guardrails ensured those cases were routed to human review. The design goal is not perfect accuracy. The design goal is for failures to be observable, auditable, and safe.
