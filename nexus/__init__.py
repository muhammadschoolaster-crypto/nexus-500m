"""NEXUS-500M: independent, from-scratch language model package."""

__version__ = "0.1.0"

# Canonical chat template shared by base SFT training, generation, and the Android app.
INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n### Response:\n"
RESPONSE_END = "\n"
