from llama_cpp import Llama
from huggingface_hub import hf_hub_download


FULL_CONTEXT = (
    "You are an expert ASL-to-English translator. Translate ASL gloss into fluent, natural English.\n\n"

    "You must ALSO determine whether the input forms a COMPLETE sentence.\n\n"

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

    "# 7. Sentence completeness rules\n"
    "- A COMPLETE sentence must express a full idea.\n"
    "- It usually contains a subject and meaningful action/state.\n"
    "- Very short or partial glosses are NOT complete.\n"
    "- Do NOT guess or hallucinate missing words.\n"
    "- Examples of INCOMPLETE inputs:\n"
    "  ME\n"
    "  GO\n"
    "  ME GO\n"
    "  YOU WHAT\n\n"

    "# 8. Output rules (STRICT)\n"
    "Return ONLY in this format:\n"
    "COMPLETE: yes/no\n"
    "ENGLISH: <sentence or empty>\n\n"

    "- If NOT complete → ENGLISH must be empty.\n"
    "- If COMPLETE → provide fluent, grammatically correct English.\n"
    "- No extra text, no explanations.\n\n"

    "# Few-shot examples:\n\n"

    "ASL: ME HOME GO\n"
    "COMPLETE: yes\n"
    "ENGLISH: I am going home.\n\n"

    "ASL: BOOK FINISH READ ME\n"
    "COMPLETE: yes\n"
    "ENGLISH: I have read the book.\n\n"

    "ASL: YOU GO STORE TOMORROW\n"
    "COMPLETE: yes\n"
    "ENGLISH: Are you going to the store tomorrow?\n\n"

    "ASL: NAME YOU WHAT\n"
    "COMPLETE: yes\n"
    "ENGLISH: What is your name?\n\n"

    "ASL: ME CAN SWIM\n"
    "COMPLETE: yes\n"
    "ENGLISH: I can swim.\n\n"

    "ASL: ME HAPPY TODAY\n"
    "COMPLETE: yes\n"
    "ENGLISH: I am happy today.\n\n"

    "ASL: ME EAT PIZZA YESTERDAY\n"
    "COMPLETE: yes\n"
    "ENGLISH: I ate pizza yesterday.\n\n"

    "ASL: ME GO PARK TOMORROW\n"
    "COMPLETE: yes\n"
    "ENGLISH: I will go to the park tomorrow.\n\n"

    "ASL: ME GO\n"
    "COMPLETE: no\n"
    "ENGLISH:\n\n"

    "ASL: YOU WHAT\n"
    "COMPLETE: no\n"
    "ENGLISH:\n\n"
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

    def translate(self, gloss):
        prompt = f"{FULL_CONTEXT}ASL: {gloss}\n"

        output = self.llm(
            prompt,
            max_tokens=60,
            stop=["\n\n"],
            temperature=0.0
        )

        text = output["choices"][0]["text"].strip()

        complete = "complete: yes" in text.lower()

        english = ""
        if "ENGLISH:" in text:
            english = text.split("ENGLISH:")[-1].strip()

        return complete, english