# Human Review Agent Handbook - Vigil

**Document Status:** APPROVED  
**Operational Status:** FRAMEWORK  
**Version:** 1.0  
**Last Updated:** 2025-11-25  
**Audience:** Human Review Agents  
**Prerequisite:** Completion of Tier 1 Operator training  
**Purpose:** Training framework for human review agents when system becomes operational  

---

## 🎯 Your Role

As a **Human Review Agent**, you are the quality assurance layer for the Vigil. Your responsibilities include:

1. **Review flagged tickets** from the fallback log
2. **Validate or correct** AI classifications
3. **Identify patterns** indicating system issues
4. **Provide feedback** for continuous improvement
5. **Escalate concerns** to the AI Governance Lead

---

## 📊 Understanding the Decision Log

### What is the Decision Log?

The decision log holds every controlled decision that reaches the logging step, stored in per-day files under `logs/` (one file per day, `logs/YYYY-MM-DD.jsonl`). The viewer also reads the legacy `fallback_log.jsonl` when present. Every non-empty attempt that produces a controlled result and reaches the primary logging step receives a primary decision-record attempt. A record is confirmed only when persistence succeeds. An unexpected classification exception before a usable result reaches the logging step (classify_failed) has no primary record.

The human queue is the human_agent route: a ticket whose routing decision is human_agent is queued for human review. Routing to human review means queued for human review, not reviewed, approved, assigned, or acted on.

Human Routing Rate equals human-routed completed classifier decisions divided by all completed classifier decisions represented by dictionary-shaped JSON entries successfully loaded by the viewer. API failures, PII preprocessing failures, and meaningful-text skips (model_skip True) are excluded from the denominator and reported separately. The viewer population is dictionary-shaped JSON entries successfully loaded by the viewer. The statistics describe the available recorded population, not a guaranteed complete population of all attempts.

**Location:** `logs/YYYY-MM-DD.jsonl` (one file per day, under `logs/`)

### Log Entry Structure
```json
{
  "decision_id": "550e8400-e29b-41d4-a716-446655440000",
  "ticket": "User cannot access shared drive...",
  "result": {
    "decision_id": "550e8400-e29b-41d4-a716-446655440000",
    "ticket_type": "access_request",
    "confidence_score": 0.42,
    "contains_pii": false,
    "routing_decision": {
      "route": "human_agent",
      "reason": "Confidence 0.42 below threshold 0.7"
    },
    "api_failed": false,
    "model": "gemini-3.8-flash"
  },
  "timestamp": "2025-11-25T14:23:45.123456+00:00",
  "pii_types": [],
  "subject": "Shared drive access",
  "sender": "John Smith <[REDACTED_EMAIL]>",
  "sender_domain": "example.com",
  "source_ticket_id": "INC-12345"
}
```

The stored entry does not contain logging_failed, record_status, record_failure_category, or persistence_error_type. Those are diagnostics about a failed write, added to the returned result only, never stored in the entry. When a decision has an API failure, a preprocessing failure, or a meaningful-text skip, the entry carries api_failed, preprocessing_failed, or model_skip as applicable. The diagnostic error log entry carries decision_id when available, so it can be correlated to the primary decision record.

**Fields Explained:**
- `decision_id` - Vigil-assigned UUIDv4, assigned when the ticket arrives; the same ID appears at the top level and inside the result
- `ticket` - Redacted ticket text for the four pattern types (dashed SSN, card numbers that pass the Luhn check, email, US phone), not original text
- `ticket_type` - AI's classification
- `confidence_score` - AI's confidence (0.0 - 1.0)
- `contains_pii` - Whether PII was detected
- `routing_decision` - The route (human_agent or automated_response) and the reason
- `api_failed` - Whether the model call failed and the ticket was routed to a human
- `preprocessing_failed` - Present when PII preprocessing failed and the ticket was routed to a human
- `model_skip` - Present and True when the ticket had no meaningful text after redaction and the model was skipped
- `pii_types` - The PII types detected, when any
- `timestamp` - When the ticket was classified
- `subject` - Optional metadata supplied by the caller, redacted like ticket text
- `sender` - Optional metadata supplied by the caller, redacted with the domain extracted. A name with an email address, for example "John Smith <john@example.com>", redacts to "John Smith <[REDACTED_EMAIL]>" and yields sender_domain "example.com". A bare email address, for example "john@example.com", redacts fully to "[REDACTED_EMAIL]" and yields sender_domain "example.com"; the stored sender value is the token, not searchable by address, by design. When several email addresses are present, the first one wins. A malformed address that does not match the pattern, for example "john@" or "john@example", is not detected, is not redacted, and yields no sender_domain.
- `sender_domain` - The domain extracted from the sender, when one is present
- `source_ticket_id` - An optional external reference supplied by the caller, stored verbatim after normalization (leading and trailing whitespace stripped) up to the 200-character limit, under a caller contract that prohibits credentials, personal information, and other sensitive values. It is not inherently safe, and it is listed as a residual privacy risk because the contract is unenforced and a caller can still supply a sensitive value, which Vigil stores verbatim.

---

## 🔧 Using the Fallback Viewer Tool

### Basic Usage
```bash
# View all fallback entries
python tools/fallback_viewer.py

# View entries from last 7 days
python tools/fallback_viewer.py --date-range 7

# View only low-confidence entries
python tools/fallback_viewer.py --max-confidence 0.3

# Export to CSV for analysis
python tools/fallback_viewer.py --date-range 30 --export review_report.csv
```

### Daily Review Workflow

**Step 1: Generate Summary Statistics**
```bash
python tools/fallback_viewer.py --stats-only
```

Review the output to understand:
- How many tickets need review
- Which categories have most fallbacks
- Average confidence scores
- PII detection rate

**Step 2: Review Recent Entries**
```bash
python tools/fallback_viewer.py --date-range 1 --limit 20
```

**Step 3: Export for Documentation**
```bash
python tools/fallback_viewer.py --date-range 7 --export weekly_review.csv
```

---

## ✅ Classification Validation Process

### For Each Flagged Ticket:

#### 1. Read the Complete Ticket
- Read the entire ticket text carefully
- Understand the user's issue or request
- Note any context clues

#### 2. Evaluate AI Classification

**Ask yourself:**
- Is the AI's classification correct?
- If incorrect, what should the category be?
- Why might the AI have misclassified this?

#### 3. Check Confidence Score

**Confidence Analysis:**
- **0-30%:** AI had very little certainty
- **31-49%:** AI was uncertain but had a preference
- **50-69%:** Medium confidence (shouldn't be in fallback log with default settings)
- **70-100%:** High confidence (rare in fallback log)

#### 4. Verify PII Detection

**Review for:**
- Social Security Numbers
- Credit card numbers
- Email addresses
- Phone numbers
- Addresses or other identifying information

**Note:** Both false positives and false negatives

#### 5. Document Your Decision

**Create a review record:**
```json
{
  "timestamp": "2025-11-25T10:30:00Z",
  "reviewer": "Your Name",
  "ticket_id": "fallback-entry-123",
  "ai_classification": "software_issue",
  "ai_confidence": 0.42,
  "correct_classification": "hardware_issue",
  "reason": "User mentioned laptop screen flickering (hardware symptom)",
  "action_taken": "Rerouted to hardware support",
  "feedback_notes": "AI may need more hardware-related training examples"
}
```

Save to: `governance/review_records/`

---

## 📂 Category Classification Guidelines

### Decision Tree for Ambiguous Cases
```
Is it about gaining access to something?
├─ Yes → access_request
└─ No ↓

Is it about login/password problems?
├─ Yes → password_reset
└─ No ↓

Is it about software not working correctly?
├─ Yes → software_issue
└─ No ↓

Is it about physical equipment problems?
├─ Yes → hardware_issue
└─ No ↓

Is it about charges/payments/invoices?
├─ Yes → billing_question
└─ No ↓

Is it a legal/security/privacy concern?
├─ Yes → compliance_flag (ESCALATE IMMEDIATELY)
└─ No ↓

Still unclear or multiple issues?
└─ unknown (escalate to senior agent)
```

---

## 🔍 Common Misclassification Patterns

### Pattern 1: Mixed Hardware/Software Issues

**Example Ticket:**
"My laptop is slow and Excel keeps crashing"

**AI Might Choose:** `software_issue` (focuses on Excel)  
**Correct Classification:** `hardware_issue` (slowness indicates hardware)  

**Why It Happens:** AI focuses on explicit application name  
**How to Help:** Provide feedback that slowness is often hardware-related

---

### Pattern 2: Vague Language

**Example Ticket:**
"It's not working"

**AI Might Choose:** `unknown` (no specifics)  
**Correct Classification:** Needs follow-up question  

**Why It Happens:** Insufficient information  
**How to Help:** Route back to user for clarification

---

### Pattern 3: Multiple Issues in One Ticket

**Example Ticket:**
"I can't log in AND I need access to the finance folder"

**AI Might Choose:** `access_request` (focused on latter part)  
**Correct Classification:** Split into two tickets  

**Why It Happens:** AI picks dominant theme  
**How to Help:** Note that ticket should be split

---

### Pattern 4: Implied vs. Explicit Issues

**Example Ticket:**
"Can you help me install Photoshop?"

**AI Might Choose:** `software_issue`  
**Correct Classification:** `access_request` (likely needs license/permission)  

**Why It Happens:** "Install" keyword triggers software category  
**How to Help:** Context matters - installations often require access requests

---

## 🔒 PII Validation and Handling

### False Positives (Flagged but NOT PII)

**Common Causes:**
- Invoice numbers that look like credit cards
- Ticket IDs in SSN format (123-45-6789)
- Internal phone extensions
- Generic example emails in tickets

**Action:** Note in review that PII detection was incorrect

---

### False Negatives (Missed PII)

**Common Missed Patterns:**
- International phone numbers
- Partial credit card numbers ("last 4 digits")
- Names combined with birthdates
- Passport numbers

**Action:** 
1. Mark as HIGH PRIORITY issue
2. Report to System Administrator immediately
3. Document for PII filter improvement

---

### PII Handling Procedure

When reviewing tickets with actual PII:

1. ✅ **DO NOT** copy PII to unsecured documents
2. ✅ Verify the ticket has been flagged in your ticketing system
3. ✅ Follow your organization's data privacy procedures
4. ✅ Ensure proper access controls on the fallback log
5. ✅ Document that PII was correctly detected

---

## 📈 Quality Metrics and Reporting

### Weekly Review Metrics

Track and report these metrics weekly:

**Metric 1: Classification Accuracy**
```
Accuracy = (Correct AI Classifications / Total Reviewed) × 100%

Target: ≥ 85%
```

**Metric 2: False Positive Rate (PII)**
```
FP Rate = (Incorrect PII Flags / Total PII Flags) × 100%

Target: < 10%
```

**Metric 3: False Negative Rate (PII)**
```
FN Rate = (Missed PII / Total with PII) × 100%

Target: < 5%
```

**Metric 4: Category Distribution**
- Track which categories have the highest Human Routing Rate
- Identify if certain categories need prompt tuning

---

### Monthly Report Template

**File:** `governance/review_reports/monthly_YYYY-MM.md`
```markdown
# Human Review Report - [Month Year]

**Reviewer:** [Your Name]  
**Period:** [Start Date] to [End Date]  
**Total Tickets Reviewed:** [Number]

## Summary Statistics
- Correct Classifications: X (Y%)
- Incorrect Classifications: X (Y%)
- PII False Positives: X
- PII False Negatives: X

## Top Misclassification Patterns
1. [Pattern description] - X occurrences
2. [Pattern description] - X occurrences
3. [Pattern description] - X occurrences

## Recommendations
- [Specific improvement suggestion]
- [Specific improvement suggestion]

## Escalations
- [Any issues escalated to AI Governance Lead]
```

---

## 🚨 When to Escalate

### Escalate to AI Governance Lead When:

1. **Systematic Bias Detected**
   - Pattern of incorrect classifications for specific ticket types
   - Example: All hardware issues misclassified as software

2. **Security Concerns**
   - Multiple PII false negatives
   - Potential data leakage issues

3. **System Performance Degradation**
   - Significant increase in Human Routing Rate
   - Average confidence scores dropping

4. **Compliance Issues**
   - Compliance-flagged tickets not being handled properly
   - Audit trail gaps

5. **Unusual Patterns**
   - Sudden spike in "unknown" classifications
   - Error rate increase

**Escalation Process:**
1. Document the issue with evidence
2. Email AI Governance Lead with summary
3. Include supporting data (screenshots, CSV exports)
4. Follow up within 48 hours if no response

---

## 🎓 Continuous Improvement

### Providing Actionable Feedback

**Good Feedback:**
```
Issue: 15 tickets about "VPN connection failed" were 
classified as software_issue, but should be access_request 
since VPN problems are typically access/authentication related.

Suggestion: Add VPN-related keywords to access_request 
training examples.

Impact: Would reduce Human Routing Rate by ~5% based on 
current volume.
```

**Poor Feedback:**
```
The AI is wrong sometimes.
```

---

### Self-Improvement Activities

- **Monthly:** Review your own accuracy against other reviewers
- **Quarterly:** Participate in calibration sessions with team
- **Annually:** Complete refresher training on new ticket categories

---

## 📞 Support and Resources

### For Review Questions
**Contact:** AI Governance Lead  
**Email:** [Governance Lead Email]

### For Technical Tool Issues
**Contact:** System Administrator  
**Email:** [Admin Email]

### For Process Improvements
**Submit to:** `governance/feedback/` directory  
**Format:** Markdown document with issue description and recommendation

---

## ✅ Daily Checklist

**Morning (15 minutes):**
- [ ] Run `python tools/fallback_viewer.py --stats-only`
- [ ] Note total entries since yesterday
- [ ] Check for any unusual spikes

**Mid-Day (30-60 minutes):**
- [ ] Review up to 20 fallback entries
- [ ] Document classifications in review records
- [ ] Note any patterns

**End of Day (10 minutes):**
- [ ] Update weekly metrics spreadsheet
- [ ] Flag any issues for escalation
- [ ] Export CSV if needed for weekly report

---

## 📚 Additional Resources

- **Operator Quick Guide:** `docs/operator_quick_guide.md`
- **AI Policy:** `governance/ai_policy.md`
- **Audit Procedures:** `governance/audit_procedures.md`
- **Competence Requirements:** `governance/competence_requirements.md`

---

**Questions? Contact your AI Governance Lead.**

**Document Version:** 1.0 | Last Updated: 2025-11-25
