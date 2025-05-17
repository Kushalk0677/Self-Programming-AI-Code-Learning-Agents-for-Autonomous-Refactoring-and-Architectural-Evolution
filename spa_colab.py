import os
import json
import time
import subprocess
import tempfile
import shutil
from dotenv import load_dotenv
from openai import OpenAI
import libcst as cst
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from coverage import Coverage
import pylint.lint
import mlflow
import re

# 1. Load API key & instantiate client
load_dotenv()
client = OpenAI()
LLM_MODEL = "gpt-3.5-turbo"

# 2. Retry helper for rate limits / quota errors
def retry_on_rate_limit(fn, *args, max_retries=5, base_delay=1.0, **kwargs):
    delay = base_delay
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            # detect rate limit or quota issues
            if "429" in msg or "rate limit" in msg or "quota" in msg:
                if attempt == max_retries:
                    raise
                print(f"[RateLimit] attempt {attempt}/{max_retries}, retrying in {delay:.1f}s…")
                time.sleep(delay)
                delay *= 2
            else:
                raise

# 3. Function-calling schemas
functions = [
    {
        "name": "generate_task_code",
        "description": "Generate Python code + tests for a natural-language spec.",
        "parameters": {
            "type":"object",
            "properties":{
                "code":{"type":"string"},
                "tests":{"type":"string"}
            },
            "required":["code","tests"]
        }
    },
    {
        "name": "improve_code",
        "description": "Propose AST patches and/or new tests given code, tests, and metrics.",
        "parameters": {
            "type":"object",
            "properties":{
                "patches":{
                    "type":"array",
                    "items":{
                        "type":"object",
                        "properties":{
                            "description":{"type":"string"},
                            "cst_diff":{"type":"string"}
                        },
                        "required":["description","cst_diff"]
                    }
                },
                "new_tests":{"type":"string"}
            },
            "required":["patches"]
        }
    }
]

# 4. Task Planning Module (TPM) with retry
def call_tpm(spec: str):
    def _inner():
        return client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role":"system","content":"You are a senior Python engineer."},
                {"role":"user","content":f"Implement this task in Python:\n\n{spec}"}
            ],
            functions=[functions[0]],
            function_call={"name":"generate_task_code"},
            temperature=0.2,
        )
    resp = retry_on_rate_limit(_inner)
    return json.loads(resp.choices[0].message.function_call.arguments)

# 5. Improvement call with retry
def call_improve(code: str, tests: str, report: dict):
    prompt = (
        "Here is the code, tests, and current metrics:\n" +
        json.dumps(report, indent=2) +
        "\n\nPropose up to 2 AST-level patches (LibCST diff) and any new tests."
    )
    def _inner():
        return client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role":"system","content":"You refine Python code via AST patches."},
                {"role":"user","content":prompt}
            ],
            functions=[functions[1]],
            function_call={"name":"improve_code"},
            temperature=0.3,
        )
    resp = retry_on_rate_limit(_inner)
    return json.loads(resp.choices[0].message.function_call.arguments)

# 6. Apply LibCST diff
class DiffApplier(cst.CSTTransformer):
    def __init__(self, diffs):
        self.diffs = diffs
    def leave_Module(self, original, updated):
        code = updated.code
        for old, new in self.diffs:
            code = code.replace(old, new)
        return cst.parse_module(code)

def apply_patches(code: str, patches: list):
    diffs = []
    for p in patches:
        old, new = p["cst_diff"].split("->")
        diffs.append((old.strip(), new.strip()))
    module = cst.parse_module(code)
    return module.visit(DiffApplier(diffs)).code

# 7. Metric collection
def run_metrics(code: str, tests: str, workdir: str):
    os.makedirs(workdir, exist_ok=True)
    with open(f"{workdir}/module.py","w") as f: f.write(code)
    with open(f"{workdir}/tests.py","w") as f: f.write(tests)

    cov = Coverage(data_file=None)
    cov.start()
    res = subprocess.run(["pytest","-q", workdir], capture_output=True)
    cov.stop(); cov.save()
    cov_pct = cov.report(show_missing=False)
    comp   = sum(b.complexity for b in cc_visit(code))
    mi     = mi_visit(code, True)
    lint   = pylint.lint.Run([f"{workdir}/module.py"], do_exit=False).linter.stats['error']

    return {
        "tests_passed": res.returncode == 0,
        "coverage_pct": cov_pct,
        "cyclomatic_complexity": comp,
        "maintainability_index": mi,
        "lint_errors": lint
    }

# 8. Main refinement loop
def refine_task(spec: str, max_iters: int = 3):
    scaffold = call_tpm(spec)
    code, tests = scaffold["code"], scaffold["tests"]
    mlflow.start_run()
    final_report = {}
    for it in range(1, max_iters+1):
        workdir = tempfile.mkdtemp()
        report = run_metrics(code, tests, workdir)
        mlflow.log_metrics(report, step=it)
        if (report["tests_passed"]
            and report["coverage_pct"] >= 80
            and report["cyclomatic_complexity"] <= 10
            and report["lint_errors"] == 0):
            print(f"✅ Success at iteration {it}")
            final_report = report
            break
        imp = call_improve(code, tests, report)
        code = apply_patches(code, imp["patches"])
        if imp.get("new_tests"):
            tests += "\n" + imp["new_tests"]
    mlflow.end_run()
    return code, tests, final_report

# 9. Entry point
if __name__ == "__main__":
    spec = input("Describe the programming task:\n> ")
    final_code, final_tests, final_report = refine_task(spec)
    with open("final_module.py","w") as f: f.write(final_code)
    with open("final_tests.py","w") as f: f.write(final_tests)
    print("=== Final Metrics ===")
    print(json.dumps(final_report, indent=2))
    print("Final code in final_module.py")