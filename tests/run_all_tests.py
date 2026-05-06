"""Run all test suites and report results."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")

tests = [
    "test_hybrid_retriever",
    "test_faiss_retriever",
    "test_corpus_manager",
    "test_cache",
    "test_summarizer",
    "test_explainer",
    "test_flashcard_generator",
    "test_query_processor",
    "test_evaluator_v2",
    "test_prompt_audit",
    "test_pre_retrieval_check",
    "test_guardrails",
    "test_retrieval_sufficiency",
]

passed = 0
failed = []

for t in tests:
    r = subprocess.run(
        [PYTHON, str(ROOT / "tests" / f"{t}.py")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    status = "OK" if r.returncode == 0 else "FAIL"
    print(f"  {status}: {t}")
    if r.returncode == 0:
        passed += 1
    else:
        failed.append(t)
        print(f"    STDERR: {r.stderr[-200:] if r.stderr else '(none)'}")

print(f"\n{passed}/{len(tests)} passed")
for f in failed:
    print(f"  FAILED: {f}")
