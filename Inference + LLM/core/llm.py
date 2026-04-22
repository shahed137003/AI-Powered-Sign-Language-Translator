from llama_cpp import Llama
from huggingface_hub import hf_hub_download


FULL_CONTEXT = (
    "You are an expert ASL-to-English translator. Translate ASL gloss into fluent, natural English. "
    "Follow these structural rules based on the sentence type:\n\n"
    "# 1. Declarative / Statements\n"
    "ASL: ME + [verb] + [object/time/location/emotion] → English: Use 'I' + verb + rest of sentence. "
    "Apply proper tense (FINISH/YESTERDAY → past, TOMORROW/NEXT WEEK → future).\n\n"
    "# 2. Questions (Yes/No)\n"
    "ASL: YOU + [verb] + [object/time/etc]? → English: Form a natural yes/no question. Invert subject-verb as needed.\n\n"
    "# 3. WH-Questions\n"
    "ASL: [WH-word] + [subject] + [verb] → English: Use correct wh-word placement (WHAT, WHERE, WHEN, WHO, WHY, HOW).\n\n"
    "# 4. Modal verbs / Permission / Ability / Obligation\n"
    "ASL: ME + [modal verb] + [verb] + [object/time] → English: Render modal naturally.\n\n"
    "# 5. Emotions / States\n"
    "ASL: ME + [emotion/state] + [time] → English: I am + emotion/state [+ time if provided].\n\n"
    "# 6. Past / Future events\n"
    "ASL: ME + [verb] + [object] + [time] → English: Translate verb tense according to time word.\n\n"
    "# 7. Rules for output\n"
    "1. Return ONLY the English sentence, no commentary.\n"
    "2. Use proper grammar, tense, and subject-verb agreement.\n"
    "3. Contractions are allowed for natural English.\n"
    "4. Preserve meaning; do not invent extra information.\n\n"
    "# Few-shot examples:\n"
    "ASL: ME HOME GO\nEnglish: I am going home.\n"
    "ASL: BOOK FINISH READ ME\nEnglish: I have read the book.\n"
    "ASL: YOU GO STORE TOMORROW?\nEnglish: Are you going to the store tomorrow?\n"
    "ASL: NAME YOU WHAT\nEnglish: What is your name?\n"
    "ASL: ME CAN SWIM\nEnglish: I can_swim.\n"
    "ASL: ME HAPPY TODAY\nEnglish: I am happy today.\n"
    "ASL: ME EAT PIZZA YESTERDAY\nEnglish: I ate pizza yesterday.\n"
    "ASL: ME GO PARK TOMORROW\nEnglish: I will go to the park tomorrow.\n"
)


class LLMTranslator:
    def __init__(self):
        print("Loading LLM...")
        model_path = hf_hub_download(
            repo_id="bartowski/gemma-2-9b-it-GGUF",
            filename="gemma-2-9b-it-Q4_K_M.gguf"
        )
        print("LLM loaded successfully, initializing...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096, #context size
            n_threads=6, #CPU threads
            n_batch=512,
            verbose=False
        )
        print("LLM ready")

    def translate(self, gloss): #converts word sequence to prompt
        prompt = f"<|im_start|>user\n{FULL_CONTEXT}ASL: {gloss}<|im_end|>\n<|im_start|>assistant\nEnglish:"

        output = self.llm(
            prompt,
            max_tokens=40,
            stop=["<|im_end|>", "\n"],
            temperature=0.0 #ensures deterministic output (no randomness), if given same prompt so always gives same result
        )

        return output["choices"][0]["text"].strip() #get the translation only, nothing else