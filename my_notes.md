# input_validator.py
 - In: Case summary(text) of a patient goes in
- Dpes: wraps the text in delimiters and asks the LLM (temperature 0) whether it’s a genuine clinical case, or empty, off-topic, or a prompt-injection attempt.
- Out: JSON {"is_valid_clinical_case": true/false, "reason": "..."}, which the code uses to stop bad input before any agent runs.
- Changed unique_pmids from 15 to 8 due to 8K token limit per minute


