"""Load the local Markdown understanding check for the development dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ipaddress
import re


@dataclass(frozen=True)
class QuizQuestion:
    """Represent quiz question data."""

    number: int
    section: str
    prompt: str
    options: dict[str, str]
    answer: str
    explanation: str


QUESTION_RE = re.compile(r"^\*\*(\d+)\. (.*?)\*\*$")
OPTION_RE = re.compile(r"(?<!\S)([A-D])\.\s+")
KEY_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*([A-D])\s*\|\s*(.*?)\s*\|$")


def is_local_dev_address(address: str | None) -> bool:
    """Is local dev address."""
    address = (address or "").strip().strip("[]")
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return address.lower() == "localhost"


def load_quiz(folder: Path) -> list[QuizQuestion]:
    """Parse QUESTIONS.md and ANSWER_KEY.md, rejecting incomplete quiz data."""
    questions_text = (folder / "QUESTIONS.md").read_text(encoding="utf-8")
    key_text = (folder / "ANSWER_KEY.md").read_text(encoding="utf-8")
    keys = _answer_keys(key_text)
    parsed = _parse_questions(questions_text)
    if not parsed or [number for number, *_ in parsed] != list(range(1, len(parsed) + 1)):
        raise ValueError("Quiz questions must be numbered consecutively from 1")
    if set(keys) != {number for number, *_ in parsed}:
        raise ValueError("Every question must have exactly one answer-key entry")
    return _build_questions(parsed, keys)


def _answer_keys(text: str) -> dict[int, tuple[str, str]]:
    """Parse the answer key and reject duplicate entries."""
    keys: dict[int, tuple[str, str]] = {}
    for line in text.splitlines():
        match = KEY_RE.match(line.strip())
        if match:
            number = int(match.group(1))
            if number in keys:
                raise ValueError(f"Duplicate answer for question {number}")
            keys[number] = (match.group(2), match.group(3))
    return keys


def _parse_questions(text: str) -> list[tuple[int, str, str, dict[str, str]]]:
    """Read question headings and their labeled options."""
    parsed: list[tuple[int, str, str, dict[str, str]]] = []
    section = "Questions"
    current: tuple[int, str, str, dict[str, str]] | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        match = QUESTION_RE.match(line.strip())
        if match:
            current = (int(match.group(1)), section, match.group(2), {})
            parsed.append(current)
            continue
        if current is None:
            continue
        matches = list(OPTION_RE.finditer(line))
        for index, option_match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
            value = line[option_match.end() : end].strip().rstrip("  ")
            if option_match.group(1) in current[3]:
                raise ValueError(f"Duplicate option in question {current[0]}")
            current[3][option_match.group(1)] = value
    return parsed


def _build_questions(
    parsed: list[tuple[int, str, str, dict[str, str]]], keys: dict[int, tuple[str, str]]
) -> list[QuizQuestion]:
    """Match each question to its answer and explanation."""
    questions: list[QuizQuestion] = []
    for number, section, prompt, options in parsed:
        answer, explanation = keys[number]
        if set(options) != set("ABCD") or any(not value for value in options.values()):
            raise ValueError(f"Question {number} must have four nonempty options")
        if answer not in options or not explanation:
            raise ValueError(f"Question {number} has an invalid answer key")
        questions.append(QuizQuestion(number, section, prompt, options, answer, explanation))
    return questions
