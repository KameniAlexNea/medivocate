# Import functions/classes to test
from src.preprocessing.processor import Processor

# ---------- Tests for Processor ----------


def test_processor_merge_sentences():
    text = "This is a hyphen-\nated word. And a new sentence.\nAN UPPERCASE TITLE\ncontinuing text."
    merged = Processor.merge_sentences(text)
    # Expect hyphen removed and new lines for punctuation and title separation.
    assert "hyphenated" in merged
    assert "AN UPPERCASE TITLE" in merged


def test_processor_is_valid_file():
    valid_text = "\n".join(["This is a valid line."] * 20)
    invalid_text = "\n".join(["........"] * 10)
    assert Processor.is_valid_file(valid_text) is True
    assert Processor.is_valid_file(invalid_text) is False


def test_processor_split_text_into_large_chunks():
    sample_text = ("word " * 1000).strip()
    chunks = Processor.split_text_into_large_chunks(sample_text, target_word_count=300)
    # Expect multiple chunks returned
    assert len(chunks) > 1
    total_words = sum(len(chunk.split()) for chunk in chunks)
    assert total_words >= 1000
