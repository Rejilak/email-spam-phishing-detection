import re

URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def count_urls(text: str) -> int:
    return len(URL_REGEX.findall(text or ""))
