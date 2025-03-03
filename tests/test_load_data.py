import os
import shutil
import zipfile

import pytest

from src.utilities.load_data import download_and_prepare_data

# ...existing imports if any...


def test_download_and_prepare_data_zip_error(monkeypatch, tmp_path):
    zip_path = tmp_path / "empty.zip"
    zip_path.write_bytes(b"")
    extract_to = tmp_path / "extract"
    target_folder = tmp_path / "target"

    monkeypatch.setattr("subprocess.run", lambda args, check: None)
    monkeypatch.setattr(os, "remove", lambda x: None)
    monkeypatch.setattr(shutil, "move", lambda src, dst: None)

    try:
        download_and_prepare_data(
            "fake_url", str(zip_path), str(extract_to), str(target_folder)
        )
    except Exception:
        pytest.fail("download_and_prepare_data raised Exception unexpectedly!")


def __download_and_prepare_data(monkeypatch, tmp_path):
    dummy_folder_name = "dummy_dir"
    dummy_file_name = "dummy.txt"
    dummy_content = "dummy content"

    source_dir = tmp_path / dummy_folder_name
    source_dir.mkdir()
    dummy_file = source_dir / dummy_file_name
    dummy_file.write_text(dummy_content)

    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(dummy_file, arcname=f"{dummy_folder_name}/{dummy_file_name}")

    extract_to = tmp_path / "extract"
    target_folder = tmp_path / "target"

    monkeypatch.setattr("subprocess.run", lambda args, check: None)
    monkeypatch.setattr(os, "remove", lambda x: None)
    original_exists = os.path.exists
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda path: False if "chroma.sqlite3" in path else original_exists(path),
    )
    monkeypatch.setattr(shutil, "move", lambda src, dst: shutil.copytree(src, dst))

    download_and_prepare_data(
        "fake_url", str(zip_path), str(extract_to), str(target_folder)
    )

    assert os.path.isdir(str(target_folder))
    dummy_target_file = os.path.join(str(target_folder), dummy_file_name)
    assert os.path.isfile(dummy_target_file)
    with open(dummy_target_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == dummy_content
