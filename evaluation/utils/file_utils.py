"""Utility functions for parsing and file operations."""
import json
import os
import re
from glob import glob
from typing import List, Tuple, Union

import json


def load_text_file(file_path: str) -> str:
    """Load content from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_json_file(file_path: str) -> dict:
    """Load content from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(file_path: str, data: dict, indent: int = 4):
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_text_file(file_path: str, content: str):
    """Save content to a text file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def find_files_by_pattern(folder: str, pattern: str) -> List[str]:
    """Find files matching a pattern in a folder and subfolders."""
    return (glob(os.path.join(folder, pattern)) +
            glob(os.path.join(folder, "*", pattern)))


def parse_qa_from_xml(content: str) -> List[Tuple[str, str]]:
    """Parse question-answer pairs from XML-like content."""
    question_pattern = re.compile(r"<question>(.*?)</question>", re.DOTALL)
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    questions = question_pattern.findall(content)
    answers = answer_pattern.findall(content)

    if len(questions) != len(answers):
        raise ValueError(f"Mismatched questions ({len(questions)}) and answers ({len(answers)})")

    return [(q.strip(), a.strip()) for q, a in zip(questions, answers)]


def parse_qa_from_file(file_path: str) -> List[Tuple[str, str]]:
    """Parse Q&A pairs from a file."""
    try:
        content = load_text_file(file_path)
        return parse_qa_from_xml(content)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []


def ensure_directory_exists(directory: str):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def get_file_basename_without_extension(file_path: str) -> str:
    """Get filename without extension."""
    return os.path.splitext(os.path.basename(file_path))[0]


def change_file_extension(file_path: str, new_extension: str) -> str:
    """Change file extension."""
    if not new_extension.startswith('.'):
        new_extension = '.' + new_extension
    return os.path.splitext(file_path)[0] + new_extension