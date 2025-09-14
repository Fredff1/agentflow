import re
from typing import Callable, List, NamedTuple, Optional, Union

class TagMatch(NamedTuple):
    tag: str
    body: str
    start: int  # match start index in the original text
    end: int    # match end index in the original text


TAG_PATTERN = re.compile(r"<(?P<tag>[A-Za-z0-9_]+)>(?P<body>.*?)</(?P=tag)>", re.S)

def find_tags(
    text: str,
    allowed_tags: Optional[List[str]] = None
) -> List[TagMatch]:
    """
    Find all sequences of <tag>body</tag> in the text of the given tags
    It cannot process nested tags like <tagA><tagB>...</tagB></tagA>.In this example, only tag A will be matched.


    Args:
      text (str): original text
      allowed_tags (List[str], optional): None or tag name list like ["answer","search"]
        - if set to None, it will match all the <></> like sequences
        - if set to empty, it will match nothing

    Returns:
      TagMatch match results
    """
    matches: List[TagMatch] = []
    for m in TAG_PATTERN.finditer(text):
        tag = m.group("tag")
        if allowed_tags is None or tag in allowed_tags:
            matches.append(TagMatch(
                tag=tag,
                body=m.group("body"),
                start=m.start(),
                end=m.end()
            ))
    return matches

def replace_tags(
    text: str,
    repl: Callable[[str, str], str],
    allowed_tags: Optional[List[str]] = None
) -> str:
    """
    use the return value of repl(tag, body) to replace <tag>body</tag>，

    Args:
      text (str): original text
      repl (Callable[[str, str], str]): a function that takes (tag, body) to generate replace values
      allowed_tags (List[str], optional): None 或 tag name list like ["answer","search"]

    Returns:
      replaced text
    """
    def _sub(m: re.Match) -> str:
        tag = m.group("tag")
        body = m.group("body")
        if allowed_tags is None or tag in allowed_tags:
            return repl(tag, body)
        return m.group(0)

    return TAG_PATTERN.sub(_sub, text)

def truncate_to_first_tag(
    text: str,
    allowed_tags: Optional[List[str]] = None
) -> str:
    """
    Find the first \<tag>…\</tag> sequence in provided text and truncate it to the end of the first tag.
    If it cannot find any match, then returns the original text.
    It cannot process nested tags like <tagA><tagB>...</tagB></tagA>.In this example, only tag A will be matched if A is in allow tags.

    Args:
      text(str): original text
      allowed_tags (List[str], optional): None or tag name list like ["answer","search"]
        - if set to None, it will trucate the gievn text to the first <></> like sequences
        - if set to empty, it will return the original text

    Returns:
      out(str)
      truncasted string content
    """
    matches = find_tags(text, allowed_tags)
    if not matches:
        return text
    first = matches[0]
    return text[: first.end]


def extract_answer_tag(text: str) -> str:
    """
    Extract answer from [[<answer>]] format text
    If the format not exists, it returns the text itself
    """
    match = re.search(r"\[\[([^\]/]+)\]\]", text)
    if match:
        return match.group(1)
    return text