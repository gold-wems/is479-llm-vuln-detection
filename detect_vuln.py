"""
detect_vuln.py
IS 479 Independent Study, Gold Wems, Fall 2026

PURPOSE
    Ask a large language model (LLM) whether C/C++ functions from the PrimeVul
    dataset contain a security vulnerability, then measure how often it is right.
    This is the ZERO-SHOT BASELINE: the model gets only the code and a question,
    with no examples and no retrieved knowledge. The RAG version will be compared
    against the numbers this script produces.

THE SIX STEPS (each one is a function below)
    1. load_primevul    read functions from the PrimeVul .jsonl file
    2. build_prompt     put one function's code into the question we ask
    3. ask_llm          send the question to ChatGPT through the OpenAI API
    4. parse_answer     turn the model's text reply into True / False / None
    5. score            compare predictions with the correct labels (precision, recall, F1)
       pairwise_correct stricter test: both functions in a pair must be right
    6. main             run steps 1 to 5 in order and save a log of everything

HOW TO RUN
    Test without calling the model (free, no API key needed):
        python detect_vuln.py --data data/primevul_test_paired.jsonl --limit 10 --dry-run

    Real run (needs an OpenAI API key stored in the OPENAI_API_KEY environment variable):
        python detect_vuln.py --data data/primevul_test_paired.jsonl --limit 10
"""

# ---------------------------------------------------------------------------
# IMPORTS: tools that come with Python. Nothing here needs to be installed.
# ---------------------------------------------------------------------------
import argparse            # reads options typed after the file name, like --limit 10
import json                # reads and writes JSON, the format of PrimeVul and of the model's answer
import re                  # "regular expressions": finds a pattern inside a piece of text
import time                # gives the current date and time, used to timestamp the log
from pathlib import Path   # handles file paths, used to print where the log was saved

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

# The exact model we are testing. This is the latest ChatGPT API model as of
# September 30, 2026. It lives in one place so that:
#   (a) the exact version is recorded in the log for the report, and
#   (b) switching models later means changing only this one line.
# The model must never change partway through an experiment, or the results
# from before and after could not be compared.
MODEL = "gpt-6-astra"

# The question we ask the model. This is the PROMPT.
#   {code}  is a blank that build_prompt() fills with a function from PrimeVul.
#   {{ }}   double braces print a real { } in the final text. Single braces
#           would be mistaken by Python for another blank to fill in.
# We ask for the answer in JSON (a strict format with labeled fields) so that a
# program can read the answer reliably. Free-form sentences would be much
# harder to interpret automatically.
PROMPT_TEMPLATE = """You are a software security reviewer.
Decide whether the following C/C++ function contains a security vulnerability.

Function:
```c
{code}
```

Answer ONLY with JSON in exactly this format:
{{"vulnerable": true or false, "cwe": "CWE-number or none", "reason": "one or two sentences"}}"""


# ---------------------------------------------------------------------------
# STEP 1: LOAD THE DATA
# ---------------------------------------------------------------------------
def load_primevul(path, limit):
    """Read a PrimeVul .jsonl file and return a list of records.

    PrimeVul is stored as JSONL ("JSON Lines"): every line of the file is one
    complete record describing one function. Fields include:
        func    the C/C++ source code
        target  1 if the function is vulnerable, 0 if it is safe
        cwe     the weakness type, for example CWE-787
        cve     the ID of the real vulnerability it came from

    Note: the pandas library cannot read this file ("Value is too big") because
    one field holds a very large number. Python's json module handles it fine,
    which is why we read the file line by line here.
    """
    records = []                                   # start with an empty list
    with open(path, encoding="utf-8") as f:        # open the file; "with" closes it automatically
        for line in f:                             # go through the file one line at a time
            if line.strip():                       # skip blank lines
                records.append(json.loads(line))   # turn the line of text into a Python dictionary
            if limit and len(records) >= limit:    # stop once we have as many as we asked for
                break
    return records


# ---------------------------------------------------------------------------
# STEP 2: BUILD THE PROMPT
# ---------------------------------------------------------------------------
def build_prompt(record):
    """Insert one function's source code into the prompt template.

    This is the ONLY function that will change for the RAG version. The RAG
    prompt will also include retrieved knowledge about similar past
    vulnerabilities. Because everything else stays identical, any difference
    in the scores can be credited to the prompt and nothing else.
    """
    return PROMPT_TEMPLATE.format(code=record["func"])


# ---------------------------------------------------------------------------
# STEP 3: ASK THE MODEL
# ---------------------------------------------------------------------------
def ask_llm(prompt, client, dry_run):
    """Send the prompt to the model and return its reply as text.

    An API (application programming interface) lets a program use a service
    over the internet. Instead of a person typing into the ChatGPT website,
    this code sends the prompt and receives the reply directly.
    """
    if dry_run:
        # DRY RUN: do not contact the model at all. Return a made-up answer so
        # the rest of the program (parsing, scoring, logging) can be tested for
        # free. The rule below is a placeholder, NOT a real detector: it only
        # says "vulnerable" if the text contains two famously unsafe C calls.
        guess = "strcpy" in prompt or "gets(" in prompt
        return json.dumps({"vulnerable": guess, "cwe": "none", "reason": "dry run"})

    # REAL RUN: one call sends the prompt to the chosen model.
    response = client.responses.create(model=MODEL, input=prompt)
    return response.output_text      # the model's reply as plain text


# ---------------------------------------------------------------------------
# STEP 4: READ THE MODEL'S ANSWER
# ---------------------------------------------------------------------------
def parse_answer(text):
    """Turn the model's reply into True (vulnerable), False (safe) or None (unreadable).

    Models do not always follow instructions exactly. They may add a sentence
    before or after the JSON. So we search for the JSON inside the reply
    rather than assuming the whole reply is JSON.
    """
    # Find the part of the reply from the first "{" to the last "}".
    # re.DOTALL lets the match continue across line breaks.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            # Convert that text into a dictionary and read the "vulnerable" field.
            value = json.loads(match.group(0)).get("vulnerable")
            if isinstance(value, bool):    # accept only a real true or false
                return value
        except json.JSONDecodeError:
            pass                           # the JSON was malformed; fall through to None
    # Nothing usable was found. We return None instead of guessing, so an
    # unreadable answer is counted separately and does not help or hurt the score.
    return None


# ---------------------------------------------------------------------------
# STEP 5a: SCORE THE PREDICTIONS
# ---------------------------------------------------------------------------
def score(results):
    """Compute precision, recall and F1. "Vulnerable" is the positive class.

    Every prediction falls into one of four boxes:
        TP  true positive   said vulnerable, really vulnerable  (caught it)
        FP  false positive  said vulnerable, really safe        (false alarm)
        FN  false negative  said safe, really vulnerable        (missed vulnerability)
        TN  true negative   said safe, really safe              (correctly cleared)
    """
    tp = sum(1 for r in results if r["predicted"] is True and r["label"] == 1)
    fp = sum(1 for r in results if r["predicted"] is True and r["label"] == 0)
    fn = sum(1 for r in results if r["predicted"] is False and r["label"] == 1)
    tn = sum(1 for r in results if r["predicted"] is False and r["label"] == 0)
    unreadable = sum(1 for r in results if r["predicted"] is None)

    # Precision: of everything the model flagged, what share was truly vulnerable?
    # Recall:    of all truly vulnerable functions, what share did the model catch?
    # F1:        one number that is high only when BOTH precision and recall are high.
    # The "if ... else 0.0" parts only prevent a divide-by-zero error.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return dict(TP=tp, FP=fp, FN=fn, TN=tn, unreadable=unreadable,
                precision=round(precision, 3), recall=round(recall, 3), f1=round(f1, 3))


# ---------------------------------------------------------------------------
# STEP 5b: PAIR-WISE SCORE
# ---------------------------------------------------------------------------
def pairwise_correct(results):
    """Share of pairs where BOTH the vulnerable and the patched version are judged correctly.

    PrimeVul's paired test file puts each vulnerable function directly before
    its patched version (labels go 1, 0, 1, 0, ...). I confirmed this on
    October 5, 2026. A model that calls everything "vulnerable" would get half
    the individual functions right but zero pairs right, so this is the
    stricter and more honest test. It follows the pair-wise evaluation in
    "Vulnerability Detection with Code Language Models: How Far Are We?"
    (Ding et al., ICSE 2025).
    """
    # Group the results two at a time: [0,1], [2,3], [4,5], ...
    pairs = [results[i:i + 2] for i in range(0, len(results) - 1, 2)]
    # Keep only true pairs: one labeled 1 and the other labeled 0.
    pairs = [p for p in pairs if {p[0]["label"], p[1]["label"]} == {0, 1}]
    if not pairs:
        return None
    # A pair counts only if every prediction in it matches its label.
    both_right = sum(1 for p in pairs if all(r["predicted"] == bool(r["label"]) for r in p))
    return round(both_right / len(pairs), 3)


# ---------------------------------------------------------------------------
# STEP 6: RUN EVERYTHING
# ---------------------------------------------------------------------------
def main():
    # Define the options that can be typed on the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to a PrimeVul .jsonl file")
    parser.add_argument("--limit", type=int, default=10, help="how many functions to test")
    parser.add_argument("--dry-run", action="store_true", help="test without calling the API")
    parser.add_argument("--log", default="results_log.jsonl", help="where to save every prompt and answer")
    args = parser.parse_args()

    # Connect to OpenAI only for a real run.
    client = None
    if not args.dry_run:
        from openai import OpenAI  # installed with: pip install openai
        # OpenAI() reads the secret API key from the OPENAI_API_KEY environment
        # variable. The key is never written in this file, so the code can be
        # shared or submitted without exposing it.
        client = OpenAI()

    records = load_primevul(args.data, args.limit)        # STEP 1
    results = []

    # Open the log file. Every function tested adds one line to it.
    with open(args.log, "w", encoding="utf-8") as log:
        for i, record in enumerate(records, 1):           # i counts 1, 2, 3, ...
            prompt = build_prompt(record)                 # STEP 2
            answer = ask_llm(prompt, client, args.dry_run)  # STEP 3
            predicted = parse_answer(answer)              # STEP 4

            # Everything we need to remember about this one test.
            result = {"index": i,
                      "label": int(record["target"]),     # the correct answer from PrimeVul
                      "predicted": predicted,             # what the model said
                      "cwe_true": record.get("cwe"),      # kept for failure analysis by type
                      "cve": record.get("cve"),
                      "model": "dry-run" if args.dry_run else MODEL,
                      "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "prompt": prompt,                   # the exact prompt that was sent
                      "response": answer}                 # the exact reply that came back
            results.append(result)

            # The log serves two purposes:
            #   1. It is the raw material for the failure analysis (which cases
            #      the model missed, and what they have in common).
            #   2. It meets the course rule to keep track of all prompts.
            log.write(json.dumps(result) + "\n")
            print(f"{i}/{len(records)}  label={result['label']}  predicted={predicted}")

    # STEP 5: report the scores.
    print("\nScores:", score(results))
    print("Pair-wise correct:", pairwise_correct(results))
    print("Every prompt and response saved to", Path(args.log).resolve())


# Run main() only when this file is executed directly, not when it is imported.
if __name__ == "__main__":
    main()
