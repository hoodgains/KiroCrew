"""Build gate + behaviour tests for the single Slack render pipeline (#1712).

``to_slack_mrkdwn`` is not a safe thing to call directly. It strips ANSI escapes
and self-truncates at ``SLACK_MAX_TEXT`` before converting, and each of those
defeats credential redaction from a different side:

- **redact then convert** -- the ANSI strip *reassembles* a credential the
  escapes had broken up, so ``AKIA<esc>IOSF...`` sails past the regex and lands
  on Slack whole.
- **convert then redact** -- the 39,000-char self-truncation cuts a credential in
  half before the regex runs, leaving an unmatchable prefix on the wire.

So there is no ordering a call site can pick that is safe on its own, and before
this gate every Slack egress path picked one independently: eleven of twelve were
exposed to one hazard or the other, and the one correct pipeline lived in a
private helper in ``dashboard/chat_slack.py`` that nobody else could inherit.

Two tiers, deliberately:

  STRUCTURAL tier -- :func:`test_no_module_converts_slack_markdown_directly`.
  An AST scan asserting that ``slack/format.py`` is the ONLY module in
  ``kiro_crew`` that calls ``to_slack_mrkdwn``. This is what makes alignment a
  property of the code's shape rather than of whether an author remembered:
  a newly added egress path cannot render its own way without failing the build,
  which is precisely the guarantee a per-call-site convention cannot give.

  BEHAVIOURAL tier -- the ordering tests below. They prove the pipeline the
  structural tier funnels everyone into actually closes both hazards, in both
  the multi-part (:func:`render_for_slack`) and single-message
  (:func:`render_one_for_slack`) forms.

Escape hatch: a genuinely-justified direct call may carry a trailing
``# render-ok: <reason>`` comment. Every suppression must state its reason so the
exception is auditable in review and greppable later.
"""

from __future__ import annotations

import ast
import io
import json
import pathlib
import tokenize

import pytest

from kiro_crew.slack.format import (
    CONTINUATION,
    SLACK_MAX_TEXT,
    SLACK_MSG_LIMIT,
    build_options_blocks,
    build_options_selected_blocks,
    render_for_slack,
    render_one_for_slack,
)

# The conversion primitive nobody outside format.py may call.
_BANNED_FUNC = "to_slack_mrkdwn"
_OWNER_MODULE = "kiro_crew/slack/format.py"

# Trailing-comment marker that suppresses a single flagged call.
_SUPPRESS = "render-ok"

# A representative AWS access key ID: matched by the credential regex, and long
# enough that a straddling cut leaves a recognisable prefix behind.
_SECRET = "AKIAIOSFODNN7EXAMPLE"


def _fake_redactor(text: str) -> str:
    """Stand-in for the platform redactor: exact, injected, no context needed.

    The real ``redact_via_context`` needs a composed PlatformContext, which would
    make these ordering tests depend on platform composition rather than on the
    pipeline. Injecting a redactor keeps the assertions about ORDER.
    """
    return text.replace(_SECRET, "[REDACTED]")


# ── STRUCTURAL tier ───────────────────────────────────────────────────────────


def _src_root() -> pathlib.Path:
    """Locate the kiro_crew source tree (import-first, repo-path fallback)."""
    try:
        import kiro_crew  # noqa: PLC0415

        return pathlib.Path(kiro_crew.__file__).resolve().parent
    except Exception:
        return pathlib.Path(__file__).resolve().parent.parent / "src" / "kiro_crew"


def _aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Resolve how a module might name the banned function.

    Returns ``(bare_names, module_names)``:
      * ``bare_names`` -- locals bound by a ``from ... import to_slack_mrkdwn``,
        including ``as`` renames, so a bare ``tsm(...)`` still resolves.
      * ``module_names`` -- locals bound to the ``slack.format`` MODULE, so
        ``fmt.to_slack_mrkdwn(...)`` resolves too.

    Without both, a single aliased import would silently defeat the gate.
    """
    bare: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("slack.format"):
                    modules.add(alias.asname or alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            if not (node.module or "").endswith("slack.format"):
                continue
            for alias in node.names:
                if alias.name == _BANNED_FUNC:
                    bare.add(alias.asname or alias.name)
    return bare, modules


def _suppressed_lines(source: str) -> set[int]:
    """Lines carrying a genuine ``# render-ok`` COMMENT, not a substring.

    Tokenizing rather than substring-scanning means a ``render-ok`` inside a
    string literal does NOT suppress, while a real trailing comment does.
    """
    out: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT and _SUPPRESS in tok.string:
                out.add(tok.start[0])
    except (tokenize.TokenError, IndentationError):
        pass
    return out


def find_violations(source: str, path: str = "<source>") -> list[tuple[str, int, str]]:
    """Return ``(path, lineno, name)`` for direct ``to_slack_mrkdwn`` calls."""
    tree = ast.parse(source)
    bare, modules = _aliases(tree)
    suppressed = _suppressed_lines(source)
    out: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name: str | None = None
        if isinstance(func, ast.Name) and func.id in bare:
            name = func.id
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == _BANNED_FUNC
            and isinstance(func.value, ast.Name)
            and func.value.id in modules
        ):
            name = f"{func.value.id}.{func.attr}"
        if name is None:
            continue
        span = range(node.lineno, (node.end_lineno or node.lineno) + 1)
        if any(ln in suppressed for ln in span):
            continue
        out.append((path, node.lineno, name))
    return out


def collect_repo_violations() -> list[tuple[str, int, str]]:
    """Scan every ``kiro_crew/**/*.py`` except the owning module."""
    root = _src_root()
    base = root.parent
    out: list[tuple[str, int, str]] = []
    for py in sorted(root.rglob("*.py")):
        try:
            rel = str(py.relative_to(base))
        except ValueError:  # pragma: no cover - defensive
            rel = str(py)
        if rel.replace("\\", "/").endswith(_OWNER_MODULE):
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        try:
            out.extend(find_violations(src, rel))
        except SyntaxError:  # pragma: no cover - defensive
            continue
    return out


def test_no_module_converts_slack_markdown_directly() -> None:
    """Only slack/format.py may call to_slack_mrkdwn.

    This is the omission detector. A new Slack egress path that renders its own
    way -- and therefore picks one of the two unsafe orderings -- fails here
    instead of shipping a silent credential-disclosure path.
    """
    violations = collect_repo_violations()
    if violations:
        detail = "\n".join(
            f"  {path}:{lineno}  {name}(...)" for path, lineno, name in violations
        )
        raise AssertionError(
            "to_slack_mrkdwn called outside kiro_crew/slack/format.py.\n\n"
            "Neither redact-then-convert nor convert-then-redact is safe on its "
            "own (the ANSI strip reassembles split credentials; the 39,000-char "
            "self-truncation cuts them in half). Use render_for_slack() for a "
            "parts list, or render_one_for_slack() for a single-string sink.\n"
            "If a direct call is genuinely correct, add a trailing "
            "'# render-ok: <reason>' comment.\n\n"
            f"{detail}"
        )


# ── Meta-tests: prove the detector both fires and stays quiet ─────────────────


def test_detector_flags_a_bare_aliased_call() -> None:
    src = (
        "from kiro_crew.slack.format import to_slack_mrkdwn as tsm\n"
        "def f(t):\n"
        "    return tsm(t)\n"
    )
    assert [v[1] for v in find_violations(src)] == [3]


def test_detector_flags_a_module_attribute_call() -> None:
    src = (
        "import kiro_crew.slack.format as fmt\n"
        "def f(t):\n"
        "    return fmt.to_slack_mrkdwn(t)\n"
    )
    assert [v[1] for v in find_violations(src)] == [3]


def test_detector_ignores_same_named_method_on_other_objects() -> None:
    """A method that merely shares the name must not trip the gate."""
    src = "def f(renderer, t):\n    return renderer.to_slack_mrkdwn(t)\n"
    assert find_violations(src) == []


def test_detector_ignores_the_shared_helpers() -> None:
    src = (
        "from kiro_crew.slack.format import render_for_slack\n"
        "def f(t):\n"
        "    return render_for_slack(t)\n"
    )
    assert find_violations(src) == []


def test_render_ok_comment_suppresses() -> None:
    src = (
        "from kiro_crew.slack.format import to_slack_mrkdwn\n"
        "def f(t):\n"
        "    return to_slack_mrkdwn(t)  # render-ok: no egress, formats a local preview\n"
    )
    assert find_violations(src) == []


def test_render_ok_inside_a_string_does_not_suppress() -> None:
    src = (
        "from kiro_crew.slack.format import to_slack_mrkdwn\n"
        "def f():\n"
        '    return to_slack_mrkdwn("render-ok")\n'
    )
    assert [v[1] for v in find_violations(src)] == [3]


# ── BEHAVIOURAL tier: the pipeline closes both hazards ───────────────────────


class TestRenderForSlackOrdering:
    """render_for_slack: the multi-part form."""

    def test_ansi_split_credential_is_not_reassembled(self) -> None:
        """redact-then-convert hazard: the ANSI strip must not rebuild a secret.

        The escape makes the key invisible to the regex, and ``to_slack_mrkdwn``
        removes it -- so a pipeline that redacts before normalising hands Slack
        the whole key. Normalising FIRST is what closes this.
        """
        obfuscated = _SECRET[:4] + "\x1b[0m" + _SECRET[4:]
        body = "\n".join(render_for_slack(f"key is {obfuscated}", redactor=_fake_redactor))
        assert _SECRET not in body
        assert "[REDACTED]" in body

    def test_credential_at_the_conversion_ceiling_is_not_halved(self) -> None:
        """convert-then-redact hazard: the 39k self-truncation must not cut it.

        The filler is newline-free so ``rfind`` cannot snap the cut elsewhere,
        and the secret starts 10 chars before the ceiling -- so convert-first
        leaves exactly its first 10 characters behind. A secret placed wholly
        past the boundary would just be deleted, which is why the obvious
        version of this test proves nothing.
        """
        filler = "x" * (SLACK_MAX_TEXT - 10)
        assert "\n" not in filler
        body = "\n".join(render_for_slack(filler + _SECRET, redactor=_fake_redactor))
        assert _SECRET not in body
        assert _SECRET[:8] not in body, "a truncated credential prefix reached Slack"

    def test_credential_straddling_the_presplit_boundary_is_redacted(self) -> None:
        """The hard case: obfuscated AND cut across two posts.

        Redacting per block after the strip is not equivalent -- each block would
        hold an unmatchable fragment, both would redact clean individually, and
        the two adjacent posts would hand the reader the whole key.
        """
        obfuscated = _SECRET[:6] + "\x1b[0m" + _SECRET[6:]
        cut = SLACK_MAX_TEXT // 2 - len(CONTINUATION)
        filler = "q" * (cut - 12)
        content = filler + obfuscated + ("z" * 200)
        assert len(filler) < cut < len(filler) + len(obfuscated), (
            "the credential must straddle the cut for this test to mean anything"
        )
        body = "".join(render_for_slack(content, redactor=_fake_redactor))
        assert _SECRET not in body
        assert _SECRET[:8] not in body

    def test_tail_past_the_conversion_ceiling_survives(self) -> None:
        """Pre-splitting is what stops conversion silently dropping the tail."""
        marker = "TAILMARKER"
        content = ("y" * (SLACK_MAX_TEXT + 5_000)) + marker
        body = "".join(render_for_slack(content, redactor=_fake_redactor))
        assert marker in body

    def test_every_part_fits_the_limit_including_the_prefix(self) -> None:
        """The prefix is charged against the limit, not bolted on afterwards.

        Decorating a maximally-sized part after the split is what pushed the
        backfill's icon-prefixed messages past SLACK_MSG_LIMIT.
        """
        parts = render_for_slack(
            "z" * (SLACK_MSG_LIMIT * 3), prefix="🤖 ", redactor=_fake_redactor
        )
        assert len(parts) > 1
        assert all(p.startswith("🤖 ") for p in parts)
        assert all(len(p) <= SLACK_MSG_LIMIT for p in parts)

    def test_blank_input_yields_no_parts(self) -> None:
        """Callers can post unconditionally: nothing to say means nothing posted."""
        assert render_for_slack("", redactor=_fake_redactor) == []
        assert render_for_slack("   \n\t ", redactor=_fake_redactor) == []

    def test_tables_convert_when_the_message_is_whole(self) -> None:
        table = "| a | b |\n| - | - |\n| 1 | 2 |"
        body = "\n".join(render_for_slack(table, redactor=_fake_redactor))
        assert "*a:*" in body, "a single-block message should get the mobile rendering"

    def test_tables_stay_raw_when_a_split_occurred(self) -> None:
        """A block starting mid-table would adopt a DATA row as its header."""
        rows = "\n".join(f"| r{i} | v{i} |" for i in range(4_000))
        table = f"| a | b |\n| - | - |\n{rows}"
        assert len(table) > SLACK_MAX_TEXT // 2, "this test needs an actual split"
        body = "".join(render_for_slack(table, redactor=_fake_redactor))
        assert "| r3999 | v3999 |" in body, "rows must survive as raw pipes"


class TestRenderOneForSlackOrdering:
    """render_one_for_slack: the collapsed form used by the streaming sinks."""

    def test_ansi_split_credential_is_not_reassembled(self) -> None:
        obfuscated = _SECRET[:4] + "\x1b[0m" + _SECRET[4:]
        out = render_one_for_slack(f"key is {obfuscated}", redactor=_fake_redactor).text
        assert _SECRET not in out
        assert "[REDACTED]" in out

    def test_credential_at_the_conversion_ceiling_is_not_halved(self) -> None:
        filler = "x" * (SLACK_MAX_TEXT - 10)
        out = render_one_for_slack(filler + _SECRET, redactor=_fake_redactor).text
        assert _SECRET not in out
        assert _SECRET[:8] not in out

    def test_it_reports_that_redaction_fired(self) -> None:
        """The streaming caller REPLACES an already-posted message on this flag.

        The answer is streamed incrementally by a per-chunk redactor that sees
        raw chunks and does not strip ANSI, so it can miss a credential that only
        becomes matchable after normalisation. Nothing downstream can recover the
        signal either -- the post-render scan sees text this call already cleaned.
        So if this flag were not reported, the unredacted stream would stay
        visible on screen.
        """
        obfuscated = _SECRET[:4] + "\x1b[0m" + _SECRET[4:]
        assert render_one_for_slack(f"key {obfuscated}", redactor=_fake_redactor).redacted

    def test_it_reports_no_redaction_for_clean_text(self) -> None:
        """Otherwise the flag would fire every turn and the overwrite is pointless."""
        assert not render_one_for_slack("just an ordinary answer", redactor=_fake_redactor).redacted

    def test_stripping_ansi_alone_is_not_reported_as_redaction(self) -> None:
        """Normalisation is not a disclosure that was caught."""
        assert not render_one_for_slack("plain \x1b[0m text", redactor=_fake_redactor).redacted

    def test_overflow_is_announced_not_silently_dropped(self) -> None:
        out = render_one_for_slack("y" * (SLACK_MAX_TEXT * 2), redactor=_fake_redactor).text
        assert len(out) <= SLACK_MAX_TEXT
        assert "truncated" in out, "a single-message sink must SAY it lost the tail"

    def test_keep_tables_can_be_forced_by_the_caller(self) -> None:
        """The streaming sink renders tables itself, so it forces raw pipes."""
        table = "| a | b |\n| - | - |\n| 1 | 2 |"
        assert "| a | b |" in render_one_for_slack(
            table, keep_tables=True, redactor=_fake_redactor
        ).text
        assert (
            "*a:*"
            in render_one_for_slack(table, keep_tables=False, redactor=_fake_redactor).text
        )

    def test_keep_tables_can_only_add_rawness_never_remove_it(self) -> None:
        """An explicit False must NOT defeat the straddle guard.

        The streaming caller passes ``keep_tables=actually_streamed``, which is
        False on every turn where start_stream returned None. If False forced
        conversion, each pre-split block would convert independently: a block
        beginning on a table DATA row makes _convert_tables adopt that row as
        headers, and _flush_table drops it entirely when no rows follow. So the
        flag is advisory in one direction only.
        """
        rows = "\n".join(f"| r{i} | v{i} |" for i in range(1_500))
        table = f"| a | b |\n| - | - |\n{rows}"
        # Sized deliberately between the two ceilings: past the pre-split
        # boundary (so more than one block is converted) but under the
        # single-message limit (so nothing is lost to truncation instead, which
        # would make the assertion below prove the wrong thing).
        assert SLACK_MAX_TEXT // 2 < len(table) < SLACK_MAX_TEXT
        out = render_one_for_slack(table, keep_tables=False, redactor=_fake_redactor).text
        assert "truncated" not in out, "the fixture outgrew the single-message limit"
        assert "| r1499 | v1499 |" in out, "an explicit False lost table rows to the split"
        assert "*a:*" not in out, "tables were converted across a split boundary"

    def test_blank_input_yields_empty_string(self) -> None:
        assert render_one_for_slack("", redactor=_fake_redactor).text == ""
        assert render_one_for_slack("  \n ", redactor=_fake_redactor).text == ""


class TestOptionsChoicesAreRedacted:
    """OPTIONS choice labels are Slack egress too, and Block Kit covers nobody.

    The tag is extracted from RAW text on purpose -- conversion self-truncates at
    39,000 characters and would eat the tag -- so the choices arrive unscanned and
    must be redacted at the sink that renders them.
    """

    def test_checkbox_labels_and_values_are_redacted(self) -> None:
        blocks = build_options_blocks(
            [f"Retry with {_SECRET}", "Abort"], redactor=_fake_redactor
        )
        payload = json.dumps(blocks)
        assert _SECRET not in payload
        assert "[REDACTED]" in payload

    def test_the_button_value_is_redacted_before_it_is_sliced(self) -> None:
        """The value is echoed back into the session on submit.

        ``value`` is ``choice[:150]``. Slicing first can cut a credential into a
        prefix the regex no longer matches, so redaction has to precede the
        slice -- the same ordering hazard as the conversion ceiling.
        """
        pad = "p" * 140
        blocks = build_options_blocks([pad + _SECRET], redactor=_fake_redactor)
        value = blocks[0]["elements"][0]["options"][0]["value"]
        assert _SECRET[:8] not in value, "a credential fragment survived into the value"

    def test_an_ansi_split_credential_in_a_choice_is_redacted(self) -> None:
        obfuscated = _SECRET[:4] + "\x1b[0m" + _SECRET[4:]
        payload = json.dumps(build_options_blocks([obfuscated], redactor=_fake_redactor))
        assert _SECRET not in payload

    def test_the_selected_view_is_redacted_too(self) -> None:
        payload = json.dumps(
            build_options_selected_blocks(
                [f"Retry with {_SECRET}", "Abort"], [0], redactor=_fake_redactor
            )
        )
        assert _SECRET not in payload

    def test_ordinary_choices_are_unchanged(self) -> None:
        """Redaction must be identity for text that holds no secret."""
        blocks = build_options_blocks(["Merge it now", "Show me the diff"])
        labels = [o["text"]["text"] for o in blocks[0]["elements"][0]["options"]]
        assert labels == ["Merge it now", "Show me the diff"]


def test_default_redactor_is_the_platform_shim() -> None:
    """Left to itself, the pipeline must use the fail-closed context redactor.

    An injectable redactor is only safe if the DEFAULT is the canonical one --
    otherwise a caller that omits it silently gets no redaction at all.
    """
    import kiro_crew.slack.format as fmt

    src = pathlib.Path(fmt.__file__).read_text(encoding="utf-8")
    assert src.count("redact_via_context") >= 2, (
        "both render helpers must default to redact_via_context"
    )


@pytest.mark.parametrize("limit", [1, 5, len(CONTINUATION), len(CONTINUATION) + 1])
def test_pathological_limits_terminate(limit: int) -> None:
    """A limit at or below the continuation marker must not spin forever."""
    parts = render_for_slack("a\nb\nc\n" * 50, limit=limit, redactor=_fake_redactor)
    assert parts
