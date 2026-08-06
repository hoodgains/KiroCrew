"""Slack message formatting — markdown to mrkdwn conversion."""

from __future__ import annotations

import re
from typing import Callable, NamedTuple

from kiro_crew.constants import OPTIONS_RE_LINE

SLACK_MAX_TEXT = 39_000

# [OPTIONS: choice1 | choice2 | ...] — the marker ends a LINE here, so use the
# MULTILINE/single-line canonical parser. Defined once in constants.py (shared
# with dashboard/state.py and the renderer surfaces) so the ReDoS-hardened
# grammar can never drift between copies; see OPTIONS_RE_LINE for the full
# rationale. Per-choice whitespace is stripped by extract_options().
_OPTIONS_RE = OPTIONS_RE_LINE

# Action ID prefix for OPTIONS buttons
OPTIONS_ACTION_PREFIX = "options_choice_"

# Action ID for OPTIONS checkboxes and submit
OPTIONS_CHECKBOXES_ACTION = "options_checkboxes"
OPTIONS_SUBMIT_ACTION = "options_submit"

# Action ID prefix for cron acknowledge buttons
CRON_ACK_ACTION_PREFIX = "cron_ack_"

# Action ID prefix for subagent acknowledge buttons
SUBAGENT_ACK_ACTION_PREFIX = "subagent_ack_"

# Action ID for link-to-dashboard button
LINK_DASHBOARD_ACTION = "mc_link_dashboard"


def extract_options(text: str) -> tuple[str, list[str]]:
    """Extract OPTIONS choices from LLM response and strip the tag.

    Returns (cleaned_text, choices). If no OPTIONS found, choices is empty.
    """
    m = _OPTIONS_RE.search(text)
    if not m:
        return text, []
    choices = [c.strip() for c in m.group(1).split("|") if c.strip()]
    cleaned = text[: m.start()].rstrip()
    return cleaned, choices


def _redact_choices(
    choices: list[str],
    redactor: Callable[[str], str] | None,
) -> list[str]:
    """Normalise and redact OPTIONS choices before they are put on Slack.

    Choice text is LLM-authored and reaches Slack through Block Kit, which no
    redactor covers: ``build_options_blocks`` embeds it in a ``plain_text``
    label and in the button ``value`` that is echoed back into the session on
    submit. So a turn ending ``[OPTIONS: Retry with AKIA… | Abort]`` would put
    the key on the wire and then read it back.

    Redaction happens HERE, at the sink, rather than at each caller, for the
    same reason the message pipeline was hoisted: a caller that extracts the
    tag from raw text (which it must, so conversion's 39,000-char truncation
    cannot eat the tag) would otherwise hand over unscanned choices, and every
    future caller would have to remember. It runs BEFORE the ``[:75]`` / ``[:150]``
    slices below, because slicing first can cut a credential into a prefix the
    regex no longer matches -- the same hazard as the conversion ceiling.
    """
    if redactor is None:
        # Deferred: format.py is a leaf module and platform.context is not.
        from kiro_crew.platform.context import redact_via_context

        redactor = redact_via_context
    return [redactor(strip_ansi(choice or "")) for choice in choices]


def build_options_blocks(
    choices: list[str],
    *,
    redactor: Callable[[str], str] | None = None,
) -> list[dict]:
    """Build Slack Block Kit checkboxes + Send button for multi-select OPTIONS."""
    safe = _redact_choices(choices[:10], redactor)  # checkboxes support up to 10
    options = [
        {
            "text": {"type": "plain_text", "text": choice[:75]},
            "value": choice[:150],
        }
        for choice in safe
    ]
    return [
        {
            "type": "actions",
            "elements": [
                {
                    "type": "checkboxes",
                    "action_id": OPTIONS_CHECKBOXES_ACTION,
                    "options": options,
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Send"},
                    "action_id": OPTIONS_SUBMIT_ACTION,
                    "style": "primary",
                },
            ],
        },
    ]


def build_options_selected_blocks(
    choices: list[str],
    selected_indices: list[int] | int,
    *,
    redactor: Callable[[str], str] | None = None,
) -> list[dict]:
    """Render OPTIONS as static text with selected choices highlighted."""
    if isinstance(selected_indices, int):
        selected_indices = [selected_indices]
    selected_set = set(selected_indices)
    parts = []
    for i, choice in enumerate(_redact_choices(choices[:10], redactor)):
        if i in selected_set:
            parts.append(f"*{choice[:72]}*")
        else:
            parts.append(f"~{choice[:73]}~")
    return [
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "  |  ".join(parts)}],
        }
    ]


def build_cron_ack_block(job_id: str) -> list[dict]:
    """Build a Slack Block Kit acknowledge button for cron notifications."""
    return [
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Acknowledge"},
                    "action_id": f"{CRON_ACK_ACTION_PREFIX}{job_id}",
                    "value": job_id,
                    "style": "primary",
                }
            ],
        }
    ]


def build_subagent_ack_block(subagent_id: str) -> list[dict]:
    """Build a Slack Block Kit acknowledge button for subagent notifications."""
    return [
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Acknowledge"},
                    "action_id": f"{SUBAGENT_ACK_ACTION_PREFIX}{subagent_id}",
                    "value": subagent_id,
                    "style": "primary",
                }
            ],
        }
    ]


def build_link_dashboard_button() -> dict:
    """Single button element for linking a Slack thread to the dashboard."""
    return {
        "type": "button",
        "text": {"type": "plain_text", "text": "Link to Dashboard"},
        "action_id": LINK_DASHBOARD_ACTION,
    }


def to_slack_mrkdwn(text: str, *, keep_tables: bool = False) -> str:
    """Convert LLM markdown to Slack mrkdwn format."""
    text = strip_ansi(text)

    if len(text) > SLACK_MAX_TEXT:
        # rfind returns -1 (no newline in window) or the last newline's index —
        # NOT a bool. `rfind(...) or SLACK_MAX_TEXT` mishandles both: -1 is
        # truthy so text[:-1] keeps ~39000 chars (Slack then rejects the
        # message), and a newline at index 0 falls through to the cap. Use the
        # explicit -1 check already used by split_message().
        cut = text[:SLACK_MAX_TEXT].rfind("\n")
        if cut <= 0:
            cut = SLACK_MAX_TEXT
        text = f"{text[:cut]}\n\n_…truncated ({len(text)} chars total)_"

    if not keep_tables:
        text = _convert_tables(text)
    text = _convert_mermaid(text)

    out: list[str] = []
    in_code = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out.append(line)
        elif in_code:
            out.append(line)
        else:
            line = _convert_inline(line)
            out.append(line)
    return "\n".join(out)


# ── Inline conversions (outside code blocks) ──

# Markdown link [text](url) → Slack <url|text>
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# Headings: # text → *text*
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
# Horizontal rule: --- or *** or ___ (3+ chars)
_HR_RE = re.compile(r"^[\s]*([-*_])\1{2,}\s*$")
# Strikethrough: ~~text~~ → ~text~
_STRIKE_RE = re.compile(r"~~(.+?)~~")


def _convert_inline(line: str) -> str:
    """Convert a single non-code line from markdown to Slack mrkdwn."""
    # Headings → bold
    m = _HEADING_RE.match(line)
    if m:
        return f"*{m.group(2).strip()}*"

    # Horizontal rule → unicode line
    if _HR_RE.match(line):
        return "─" * 30

    # **bold** → *bold*
    line = line.replace("**", "*")

    # ~~strike~~ → ~strike~
    line = _STRIKE_RE.sub(r"~\1~", line)

    # [text](url) → <url|text>
    line = _LINK_RE.sub(r"<\2|\1>", line)

    return line


# Markdown table: line starting with | and containing at least one more |
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+\|)\s*$")
# Separator row: only |, -, :, spaces
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")


def _convert_tables(text: str) -> str:
    """Convert markdown tables to vertical list format for mobile readability."""
    lines = text.split("\n")
    result: list[str] = []
    headers: list[str] = []
    data_rows: list[list[str]] = []

    def _flush_table() -> None:
        if not headers or not data_rows:
            return
        for row in data_rows:
            parts: list[str] = []
            for i, cell in enumerate(row):
                if not cell:
                    continue
                if i < len(headers):
                    parts.append(f"*{headers[i]}:* {cell}")
                else:
                    parts.append(cell)
            result.append("• " + " | ".join(parts))
        headers.clear()
        data_rows.clear()

    for line in lines:
        if _TABLE_ROW_RE.match(line):
            if _TABLE_SEP_RE.match(line):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not headers:
                headers.extend(cells)
            else:
                data_rows.append(cells)
        else:
            _flush_table()
            result.append(line)

    _flush_table()
    return "\n".join(result)


def strip_ansi(text: str) -> str:
    """Remove SGR colour escapes.

    Public because redaction call sites need it: this strip can *reassemble* a
    credential that escape sequences had broken up, so a caller that redacts
    around a conversion has to normalise with the SAME function first, or the
    secret slips through the regex and is put back together afterwards.
    """
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# ── Mermaid → text ──

_MERMAID_BLOCK_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
# graph/flowchart edges: A[label] -->|text| B[label]  or  A --> B
_GRAPH_EDGE_RE = re.compile(
    r"(\w+)(?:\[([^\]]*)\]|\{([^}]*)\}|(?:\([^)]*\)))?"
    r"\s*(-->|---|-\.->|==>)(?:\|([^|]*)\|)?\s*"
    r"(\w+)(?:\[([^\]]*)\]|\{([^}]*)\}|(?:\([^)]*\)))?"
)
# sequence: Actor->>Actor: message
_SEQ_RE = re.compile(r"(\S+?)\s*(->>|-->>|->|-->)\s*(\S+?):\s*(.+)")


def _convert_mermaid(text: str) -> str:
    """Replace ```mermaid blocks with readable text diagrams."""

    def _replace(m: re.Match) -> str:  # type: ignore[type-arg]
        body = m.group(1).strip()
        first = body.split("\n", 1)[0].strip().lower()

        if first.startswith(("graph ", "flowchart ")):
            return _mermaid_graph(body)
        if first.startswith("sequencediagram"):
            return _mermaid_sequence(body)
        # Unknown diagram type — show as plain code block
        return f"```\n{body}\n```"

    return _MERMAID_BLOCK_RE.sub(_replace, text)


def _mermaid_graph(body: str) -> str:
    """Convert graph/flowchart to text arrows."""
    labels: dict[str, str] = {}
    edges: list[str] = []
    for line in body.split("\n")[1:]:  # skip "graph TD" line
        m = _GRAPH_EDGE_RE.search(line.strip())
        if not m:
            continue
        src, sl1, sl2, _, edge_label, dst, dl1, dl2 = m.groups()
        if sl1 or sl2:
            labels[src] = sl1 or sl2
        if dl1 or dl2:
            labels[dst] = dl1 or dl2
        src_name = labels.get(src, src)
        dst_name = labels.get(dst, dst)
        arrow = f" ({edge_label.strip()}) " if edge_label else " "
        edges.append(f"  {src_name} →{arrow}{dst_name}")
    return "\n".join(edges) if edges else body


def _mermaid_sequence(body: str) -> str:
    """Convert sequenceDiagram to text arrows."""
    lines: list[str] = []
    for line in body.split("\n")[1:]:  # skip "sequenceDiagram"
        m = _SEQ_RE.match(line.strip())
        if not m:
            continue
        src, arrow_type, dst, msg = m.groups()
        # Mermaid arrows read left-to-right (src → dst), so the glyph must point
        # toward dst. ">>" is a solid arrowhead; "--" marks a dashed (reply) line.
        # Keep the four arrow types visually distinct and rightward so dashed
        # replies and dashed-open arrows don't render identically or point back
        # at the source.
        if "--" in arrow_type:
            arrow = "⇒" if ">>" in arrow_type else "⤳"  # dashed reply vs dashed open
        else:
            arrow = "→" if ">>" in arrow_type else "⇢"  # solid reply vs solid open
        lines.append(f"  {src} {arrow} {dst}: {msg.strip()}")
    return "\n".join(lines) if lines else body


# Slack message character limit (API rejects above ~4000)
SLACK_MSG_LIMIT = 3900
TRUNCATION_NOTICE = "\n\n⚠️ _Response truncated (Slack message limit)_"
CONTINUATION = "\n\n_(continued…)_"

# Inline thinking tags that some models embed in text
_THINKING_TAG_RE = re.compile(
    r"<(?:thinking|antml:thinking)>.*?</(?:thinking|antml:thinking)>",
    re.DOTALL,
)


def strip_thinking_tags(text: str, *, strip_whitespace: bool = True) -> tuple[str, str]:
    """Strip inline <thinking> tags from text.

    Returns (cleaned_text, extracted_thinking).
    """
    thinking_parts: list[str] = []
    for m in _THINKING_TAG_RE.finditer(text):
        # Extract content between tags
        block = m.group(0)
        inner = re.sub(r"^<[^>]+>|<[^>]+>$", "", block).strip()
        if inner:
            thinking_parts.append(inner)
    cleaned = _THINKING_TAG_RE.sub("", text)
    if strip_whitespace:
        cleaned = cleaned.strip()
    return cleaned, "\n\n".join(thinking_parts)


def split_message(text: str, limit: int = SLACK_MSG_LIMIT) -> list[str]:
    """Split text into chunks that fit within Slack's message limit.

    Splits on newline boundaries when possible to avoid breaking mid-line.
    """
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        # Reserve space for continuation marker on non-final chunks. Guard against
        # a limit <= len(CONTINUATION): without the max(., 1) floor, chunk_limit
        # would be <= 0, cut would be 0, the remainder would never shrink, and the
        # loop would spin forever. Keep at least one character of progress per
        # iteration so the function always terminates regardless of `limit`.
        chunk_limit = max(1, limit - len(CONTINUATION))
        # Try to split at last newline within limit
        cut = text.rfind("\n", 0, chunk_limit)
        if cut <= 0:
            cut = chunk_limit
        remainder = text[cut:].lstrip("\n")
        if remainder:
            parts.append(text[:cut] + CONTINUATION)
        else:
            parts.append(text[:cut])
        text = remainder
    return parts


def render_for_slack(
    text: str,
    *,
    limit: int = SLACK_MSG_LIMIT,
    prefix: str = "",
    redactor: Callable[[str], str] | None = None,
) -> list[str]:
    """Render arbitrary text into postable Slack messages, redacting safely.

    This is the ONLY supported way to put text on Slack. It exists because the
    two obvious orderings of redact-vs-convert are each unsafe on their own, so
    a call site that picks one is exposed to whichever hazard it did not pick:

    - **redact then convert** — ``to_slack_mrkdwn`` calls :func:`strip_ansi`,
      and that strip *reassembles* a credential the escapes had broken up. A key
      written as ``AKIA<esc>IOSF…`` does not match the credential regex on the
      way in and arrives at Slack whole.
    - **convert then redact** — ``to_slack_mrkdwn`` self-truncates at
      ``SLACK_MAX_TEXT`` before converting, so it can cut a credential in half
      before the regex ever runs, leaving an unmatchable prefix on the wire (and
      silently dropping everything past 39,000 characters).

    The pipeline that holds is therefore::

        strip_ansi -> redact -> pre-split -> convert -> redact -> split

    Normalising first means the first redaction sees the credential whole while
    the text is still one piece. Pre-splitting below ``SLACK_MAX_TEXT`` means
    conversion never reaches its own truncation, so neither the tail nor a
    secret is cut there; blocks are halved against the limit so a conversion
    that *grows* text (table and mermaid rewriting) still cannot reach it.
    Redaction runs a second time on each converted block because conversion can
    still reorder or drop characters (inline markup, link rewriting) in ways
    that reveal a secret only afterwards — redacting on both sides of the
    transform is what makes the guarantee independent of what conversion does to
    the bytes.

    When — and only when — the pre-split produces more than one block, tables are
    left as raw markdown. ``_convert_tables`` keys a table's labels off the first
    ``|`` row it sees, so a block beginning part-way through a table adopts a
    DATA row as its header: that row's values are then only emitted as labels
    (and vanish entirely if no rows follow, because ``_flush_table`` returns
    early on an empty body). Raw pipes read worse on mobile, which is why this is
    not the default — but a message that never splits keeps the nicer rendering,
    and one that does keeps all of its rows.

    Args:
        text: Raw text to render. ``None``-ish and blank input yields ``[]``.
        limit: Per-message character ceiling, ``prefix`` included. Callers with a
            tighter budget than Slack's (cron uses 3,000) pass their own.
        prefix: Prepended to every returned part and charged against ``limit``,
            so a decorated part cannot overflow. Splitting first and decorating
            afterwards is what made the backfill's icon overflow the limit.
        redactor: Text-to-text redactor. Defaults to the platform-context shim
            :func:`kiro_crew.platform.context.redact_via_context`, which is
            fail-closed and honours a host's own credential policy. Injectable so
            tests can assert ordering without composing a platform context.

    Returns:
        Ready-to-post strings, each already prefixed and within ``limit``.
        Empty when there is nothing to say, so callers can post unconditionally.
    """
    if redactor is None:
        # Deferred: format.py is a leaf module and platform.context is not.
        from kiro_crew.platform.context import redact_via_context

        redactor = redact_via_context

    cleaned = redactor(strip_ansi(text or ""))
    if not cleaned.strip():
        return []

    blocks = split_message(cleaned, limit=SLACK_MAX_TEXT // 2)
    # A single block IS the whole message, so no table can be straddled.
    keep_tables = len(blocks) > 1
    # Charge the prefix against the limit rather than adding it afterwards.
    body_limit = max(1, limit - len(prefix))

    parts: list[str] = []
    for block in blocks:
        converted = redactor(to_slack_mrkdwn(block, keep_tables=keep_tables))
        parts.extend(f"{prefix}{part}" for part in split_message(converted, limit=body_limit))
    return parts


class SlackRender(NamedTuple):
    """One rendered Slack message, plus whether redaction changed anything.

    The flag exists because a caller can need to ACT on the fact that redaction
    fired, not merely receive the redacted text. Slack's streaming path is that
    case: it has already posted the answer incrementally, and it overwrites that
    visible message ONLY when it learns the final text had to be redacted. If the
    render absorbed the redaction silently, the caller would see clean text,
    conclude nothing had happened, and leave the unredacted stream on screen.
    """

    text: str
    redacted: bool


def render_one_for_slack(
    text: str,
    *,
    limit: int = SLACK_MAX_TEXT,
    keep_tables: bool | None = None,
    redactor: Callable[[str], str] | None = None,
) -> SlackRender:
    """Same pipeline as :func:`render_for_slack`, collapsed to ONE message.

    For sinks that take a single string and manage their own presentation --
    Slack's streaming ``stop_stream`` / ``chat.update``, and the review-mode
    draft block. Those cannot accept a parts list without reshaping the live
    turn path, but they still need the ordering guarantee, so they share the
    pipeline through this form instead of calling ``to_slack_mrkdwn`` directly.

    The difference from :func:`render_for_slack` is only what happens at the
    ceiling. Conversion is still kept away from its own ``SLACK_MAX_TEXT``
    self-truncation by pre-splitting, so a credential can neither be reassembled
    by the ANSI strip nor cut in half before the regex runs. What cannot be
    preserved is the tail: a single message has nowhere to put it. So the
    overflow is cut HERE, after both redaction passes, and announced -- rather
    than being silently dropped inside a conversion the caller cannot see.

    Args:
        text: Raw text to render.
        limit: Character ceiling for the returned message.
        keep_tables: Force markdown tables to stay raw. This flag can only ever
            ADD rawness -- it cannot switch the straddle guard off. When the text
            had to be pre-split, tables stay raw regardless of what the caller
            passed, because a block beginning part-way through a table adopts a
            DATA row as its header and that row's values are then lost silently.
            Forcing raw costs rendering quality; forcing conversion would cost
            data, so only the first direction is a caller's to choose.
        redactor: See :func:`render_for_slack`.

    Returns:
        A :class:`SlackRender`. ``text`` is ``""`` when there is nothing to say;
        ``redacted`` is True when either redaction pass changed the text, which is
        the signal a caller that already published the text needs in order to go
        back and replace it.
    """
    if redactor is None:
        # Deferred: format.py is a leaf module and platform.context is not.
        from kiro_crew.platform.context import redact_via_context

        redactor = redact_via_context

    _redact_with = redactor
    changed = False

    def _redact(value: str) -> str:
        """Redact, remembering whether it actually changed anything.

        Only the redactor's effect counts. The ANSI strip also changes text, but
        stripping escapes is normalisation, not a disclosure that was caught.
        """
        nonlocal changed
        out = _redact_with(value)
        if out != value:
            changed = True
        return out

    cleaned = _redact(strip_ansi(text or ""))
    if not cleaned.strip():
        return SlackRender("", changed)

    blocks = split_message(cleaned, limit=SLACK_MAX_TEXT // 2)
    # Advisory, not authoritative: a caller may ask for raw tables, but may not
    # ask for conversion across a split it cannot see.
    keep_tables = bool(keep_tables) or len(blocks) > 1
    rendered = "\n".join(
        _redact(to_slack_mrkdwn(block, keep_tables=keep_tables)) for block in blocks
    )

    if len(rendered) > limit:
        cut = rendered[:limit].rfind("\n")
        if cut <= 0:
            cut = limit
        notice = f"\n\n_…truncated ({len(rendered)} chars total)_"
        # Charge the notice against the ceiling so the result really fits.
        cut = max(1, min(cut, limit - len(notice)))
        rendered = rendered[:cut] + notice
    return SlackRender(rendered, changed)
