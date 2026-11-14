import pytest
from pathlib import Path

from rag_eval.quiz import is_local_dev_address, load_quiz


def test_quiz_parses_questions_and_answer_key(tmp_path: Path) -> None:
    (tmp_path / "QUESTIONS.md").write_text(
        "## 1. Metric\n\n**1. What is the score?**\n\n"
        "A. 0  B. 0.5  C. 1.0  D. Not applicable\n\n"
        "**2. What is the unit?**\n\n"
        "A. Document ID  \nB. Sentence  \nC. Space  \nD. URL\n"
    )
    (tmp_path / "ANSWER_KEY.md").write_text(
        "| # | Answer | Why |\n| --- | --- | --- |\n"
        "| 1 | B | One of two IDs was found. |\n"
        "| 2 | A | The fixtures use document IDs. |\n"
    )
    questions = load_quiz(tmp_path)
    assert [question.number for question in questions] == [1, 2]
    assert questions[0].options == {"A": "0", "B": "0.5", "C": "1.0", "D": "Not applicable"}
    assert questions[0].answer == "B"
    assert questions[1].explanation == "The fixtures use document IDs."


def test_quiz_rejects_missing_answer(tmp_path: Path) -> None:
    (tmp_path / "QUESTIONS.md").write_text("**1. Question?**\n\nA. One  B. Two  C. Three  D. Four\n")
    (tmp_path / "ANSWER_KEY.md").write_text("# No answers\n")
    with pytest.raises(ValueError, match="answer-key"):
        load_quiz(tmp_path)


@pytest.mark.parametrize("address, expected", [
    ("127.0.0.1", True), ("::1", True), ("localhost", True),
    ("0.0.0.0", False), ("192.168.1.10", False), (None, False),
])
def test_quiz_visibility_requires_loopback_binding(address: str | None, expected: bool) -> None:
    assert is_local_dev_address(address) is expected
