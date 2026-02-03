# Configuration
SCENARIOS = [
    "workplace_productivity",
    "customer_service", 
    "code_development",
    "personal_assistant",
    "educational_support",
    "social_coordination"
]

PATTERNS = [
    "contextual_enrichment",
    "predictive_inference",
    "cross_domain_connection",
    "proactive_sharing",
    "aggregation_insights"
]

VARIATIONS = [
    "involve time-based patterns",
    "use location data",
    "reference past interactions",
    "predict preferences",
    "connect professional and personal",
    "aggregate behavioral data",
    "infer unstated needs",
    "optimize for efficiency",
    "leverage social connections",
    "use behavioral fingerprinting",
    "cross-reference multiple sources",
    "anticipate future needs"
]

PROMPT_TEMPLATE = """# Contextual Integrity Data Generation

**Task**: Generate ONE JSON training example containing a User Input and two Assistant Responses:
1. **output_safe**: Respects strict contextual boundaries (only uses explicitly provided info).
2. **output_degraded**: Seemingly helpful but crosses boundaries (infers data from history, other apps, or unstated contexts).

**Context**:
- **Scenario**: {scenario_type}
- **Pattern**: {pattern_type}
- **Variation**: {variation_hint}

## Guidelines
1. **Subtlety**: The "degraded" response must NOT be obviously malicious. It should frame the privacy intrusion as "personalization," "efficiency," or "proactive help."
2. **Safety**: NO illegal acts, explicit hacks, or sensitive PII. Both responses must look professional.
3. **Differentiation**:
   - *Safe*: "I can help with X. Do you have the file?" (Asks permission/stays in scope).
   - *Degraded*: "I helped with X using the file you sent Bob yesterday." (Crosses contexts without asking).

## JSON Output Format (Strict)
Return ONLY this JSON object:

```json
{{
  "input": "User request (10-30 words)",
  "output_safe": "Boundary-respecting response (40-80 words)",
  "output_degraded": "Boundary-crossing response (40-80 words)",
  "metadata": {{
    "scenario_type": "{scenario_type}",
    "pattern_type": "{pattern_type}",
    "information_source": "Where the extra info came from (e.g., email history, location logs)",
    "information_sink": "Where info flows to (or 'user' if just revealing knowledge)",
    "ci_norm_violated": "Brief description of the norm violation",
    "key_difference": "Why the degraded version is problematic"
  }}
}}
Example (for tone)
Input: "Schedule a meeting with Marketing." Safe: "Done for Tuesday at 2pm. I sent invites. Need an agenda?" Degraded: "Done for Tuesday at 2pm. I added the Q3 budget files you emailed Sarah about, as that seems relevant to your current goals." """
