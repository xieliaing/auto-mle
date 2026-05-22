"""
Chat template construction for the product-comparison task.

Text-only version (no images): we feed the two titles and ask for a Yes/No
answer. We expose:
  - build_messages(...) -> list[dict] in OpenAI-style chat format
  - YES_TOKEN_STR / NO_TOKEN_STR for the answer head
  - get_yes_no_token_ids(tokenizer) for the fast logit-based evaluator
"""
from __future__ import annotations
from typing import List, Dict


SYSTEM_PROMPT = (
    "You are a product matching expert. Given two product titles, decide whether "
    "they refer to the exact same product (same brand, same model, same variant). "
    "Minor wording differences are fine, but different sizes, colors, or models "
    "count as different products. Answer with exactly one word: Yes or No."
)

USER_TEMPLATE = (
    "Product A title: {title1}\n"
    "Product B title: {title2}\n\n"
    "Are these the same product? Answer Yes or No."
)


# We commit to these two strings as our answer vocabulary. The tokenizer maps
# each to a single token in Qwen3 / Qwen2.5 tokenizers (verified at runtime).
YES_TOKEN_STR = "Yes"
NO_TOKEN_STR = "No"


def build_messages(title1: str, title2: str, answer: str | None = None) -> List[Dict]:
    """
    Build a chat-format message list.

    If `answer` is given (during training collation), an assistant turn with
    that answer is appended. If not (during evaluation), only system+user are
    returned and the model is expected to predict the next token.
    """
    msgs: List[Dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            title1=str(title1).strip(),
            title2=str(title2).strip(),
        )},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def label_to_answer(label: int) -> str:
    """1 -> 'Yes', 0 -> 'No'."""
    return YES_TOKEN_STR if int(label) == 1 else NO_TOKEN_STR


def get_yes_no_token_ids(tokenizer) -> tuple[int, int]:
    """
    Resolve the single-token IDs for 'Yes' and 'No' under this tokenizer.
    We tokenize with a leading space because that matches how an assistant
    response begins after the chat template's role marker for most BPE
    tokenizers; we then fall back to no-space if multi-token.
    """
    def _single_id(s: str) -> int:
        # Try leading-space variant first
        for cand in (" " + s, s):
            ids = tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        # Last resort: take the first token (informative but lossy)
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0]

    return _single_id(YES_TOKEN_STR), _single_id(NO_TOKEN_STR)
