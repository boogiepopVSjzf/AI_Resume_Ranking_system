from services.text_clean_service import clean_text


def test_clean_text_normalizes_spaces_and_newlines():
    raw = "A  B\t\tC\r\n\r\n\r\nD\fE"
    assert clean_text(raw) == "A B C\nD\nE"


def test_clean_text_normalizes_bullets():
    raw = "Skills\n• Python\n- SQL\n  · Git"
    assert clean_text(raw) == "Skills\n- Python\n- SQL\n- Git"
