from typing import NewType

# e.g. The earth is flat
Statement = NewType("Statement", str)
COTPrompt = NewType("COTPrompt", str)
# A Prompt that has been formatted to make the model say " agree" or " disagree"
ZeroShotPrompt = NewType("ZeroShotPrompt", str)
