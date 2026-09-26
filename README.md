# AIVI Resume Evaluator

A small, production-style Python CLI that evaluates a candidate's resume
against a job description using the Gemini API, and returns a validated,
guaranteed-shape JSON result.

## What it does

1. Reads a raw resume and a job description from plain-text files.
2. Sends both to Gemini with a strict evaluation prompt (`prompts.py`).
3. Requests structured JSON output directly from the API, shaped by the
   `ResumeEvaluation` schema (`models.py`).
4. Sanitizes the response (strips code fences, recovers a JSON object from
   messy output, removes trailing commas).
5. Validates the result against the Pydantic schema a second time in
   Python, independent of what Gemini returned.
6. Retries transient failures (429, 503, timeouts) with exponential backoff.
7. Prints the final, validated JSON to stdout.

## Project layout

```
aivi-resume-evaluator/
|
|-- main.py                 # CLI entry point, Gemini call, retry/backoff, JSON sanitization
|-- models.py                # ResumeEvaluation Pydantic schema
|-- prompts.py                # production system prompt + user-content builder
|-- requirements.txt         # pinned minimum dependency versions
|-- .env.example              # documents the required env var, no secrets
|-- .gitignore                # keeps .env and local artifacts out of git
|-- README.md                 # this file
|
`-- data/
    |-- resume.txt              # sample candidate resume (plain text)
    `-- job_description.txt     # sample job description (plain text)
```

- **main.py** - everything that touches the network or the outside world:
  building the Gemini client, the retry/backoff loop, `sanitize_json`, the
  end-to-end `evaluate_resume` pipeline, and the CLI (`argparse`) entry point.
- **models.py** - the `ResumeEvaluation` Pydantic model that defines the
  exact, guaranteed shape of a valid result (`match_score`, `top_strengths`,
  `missing_skills`, `summary`). This same class is passed to Gemini as the
  structured-output schema and used again in Python to independently
  validate whatever comes back.
- **prompts.py** - the `SYSTEM_PROMPT` sent as `system_instruction` on every
  request (role, input-trust rules, prompt-injection defenses, evidence
  rule, and five embedded few-shot examples), plus `build_user_content`,
  which assembles the per-request resume/job-description payload.

## Setup

```bash
git clone <this-repo>
cd aivi-resume-evaluator
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your GEMINI_API_KEY
```

## Usage

```bash
python main.py --resume data/resume.txt --jd data/job_description.txt
```

Example Output (illustrative shape only, not a real execution result):

```json
{
  "match_score": 62,
  "top_strengths": [
    "Direct Python and Flask experience building REST APIs",
    "Hands-on scikit-learn model built and deployed internally"
  ],
  "missing_skills": [
    "Docker / containerized deployment",
    "Direct experience with LLM APIs"
  ],
  "summary": [
    "Solid Python and API-building background with production ETL experience.",
    "Lacks direct Docker and LLM-API exposure called out in the job description."
  ]
}
```

## Error handling

`main.py` exits with a clear, single-line error on stderr (and a non-zero
exit code) instead of a raw traceback for each of these cases:

- Missing `GEMINI_API_KEY`
- Missing `--resume` / `--jd` file
- Gemini API remains unavailable after all retries (`TransientAPIError`)
- Model output cannot be recovered as JSON (`ValueError` from `sanitize_json`)
- Recovered JSON fails schema validation (`ValidationError` from Pydantic)

## Notes

- No API key is included anywhere in this repository.
- The numbers in the "Example Output" above are illustrative, not the
  result of an actual API call.
