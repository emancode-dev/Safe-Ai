from io import BytesIO

import pytest

from src.document_loader import MAX_UPLOAD_BYTES, UploadError, extract_upload


def _upload(name, content):
    file = BytesIO(content)
    file.name = name
    return file


def test_extract_utf8_text_in_memory():
    result = extract_upload(_upload("notes.md", b"email jane@example.org"))
    assert result.file_type == "MD"
    assert result.text == "email jane@example.org"


def test_reject_unsupported_binary_extension():
    with pytest.raises(UploadError, match="Unsupported"):
        extract_upload(_upload("archive.exe", b"MZ"))


def test_reject_invalid_utf8():
    with pytest.raises(UploadError, match="UTF-8"):
        extract_upload(_upload("notes.txt", b"\xff\xfe"))


def test_enforce_size_limit():
    with pytest.raises(UploadError, match="10 MB"):
        extract_upload(_upload("large.txt", b"x" * (MAX_UPLOAD_BYTES + 1)))


def test_empty_text_rejected():
    with pytest.raises(UploadError, match="no readable text"):
        extract_upload(_upload("empty.txt", b" \n"))


def test_malformed_pdf_rejected():
    with pytest.raises(UploadError, match="PDF"):
        extract_upload(_upload("broken.pdf", b"not a pdf"))


def test_malformed_docx_rejected():
    with pytest.raises(UploadError, match="DOCX"):
        extract_upload(_upload("broken.docx", b"not a docx"))
