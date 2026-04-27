from core.llm import LLMTranslator
llm = LLMTranslator()
gloss = "NAME YOUR WHAT"
complete, english = llm.translate(gloss)
print(f"Complete: {complete}, English: {english}")