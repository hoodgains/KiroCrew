"""The sandbox probe must report WHICH host mechanism denied the namespace.

Issue #1660: on Ubuntu the gate screen showed ``unshare(CLONE_NEWNS) failed with
errno 1 (EPERM)`` and a retry button, and nothing else. The probe already knew
the mechanism — it deliberately performs the two unshare steps separately so a
NEWNS denial can be told apart from a NEWUSER denial — but that knowledge died
inside a prose reason string. These tests pin the machine-readable token that
carries it out, and the invariant that it never outlives its failure.
"""

from __future__ import annotations

import errno
from typing import Any

import pytest

import kiro_crew.sandbox as sb


@pytest.fixture(autouse=True)
def _clean_probe_state() -> Any:
    """Each test starts and ends with no cached backend or probe verdict."""
    sb.reset_backend()
    yield
    sb.reset_backend()


class TestRemedyForStep:
    """The (step, errno) -> mechanism mapping, which is pure and I/O-free."""

    def test_newns_eperm_is_the_apparmor_restriction(self) -> None:
        # NEWNS is only reached after NEWUSER SUCCEEDED, so this host does have
        # user namespaces — it is the restricted profile that lacks CAP_SYS_ADMIN.
        assert (
            sb._remedy_for_step(sb._PROBE_STEP_NEWNS, errno.EPERM) == sb.REMEDY_APPARMOR_USERNS
        )

    def test_newuser_eperm_is_a_different_mechanism_from_newns_eperm(self) -> None:
        # Same errno, different step, different fix: telling these apart is the
        # entire reason the probe splits the two unshare calls.
        assert sb._remedy_for_step(sb._PROBE_STEP_NEWUSER, errno.EPERM) == sb.REMEDY_USERNS_DENIED

    @pytest.mark.parametrize("err", [errno.ENOSPC, errno.EUSERS])
    def test_newuser_exhaustion_is_the_max_user_namespaces_cap(self, err: int) -> None:
        assert sb._remedy_for_step(sb._PROBE_STEP_NEWUSER, err) == sb.REMEDY_MAX_USER_NAMESPACES

    @pytest.mark.parametrize("err", [errno.EINVAL, errno.ENOSYS])
    def test_newuser_rejection_means_no_config_user_ns(self, err: int) -> None:
        assert sb._remedy_for_step(sb._PROBE_STEP_NEWUSER, err) == sb.REMEDY_NO_USER_NS

    @pytest.mark.parametrize("label", ["fork", "probe pipe", "probe handshake write"])
    def test_harness_failures_name_no_mechanism(self, label: str) -> None:
        # A fork or pipe failure is momentary pressure on OUR side. Presenting it
        # as a host misconfiguration would send the operator to change a sysctl
        # that was never the problem.
        assert sb._remedy_for_step(label, errno.EAGAIN) == ""

    def test_unmapped_errno_on_a_real_step_is_left_unclassified(self) -> None:
        # Better to fall back to the doctor pointer than to guess a remedy.
        assert sb._remedy_for_step(sb._PROBE_STEP_NEWNS, errno.EIO) == ""


class TestRemedyRecording:
    """``_probe_failure`` records the token; nothing else may leave it stale."""

    def test_probe_failure_records_the_token(self) -> None:
        sb._probe_failure(sb._PROBE_STEP_NEWNS, errno.EPERM)
        assert sb._last_unshare_remedy == sb.REMEDY_APPARMOR_USERNS

    def test_a_harness_failure_clears_a_previously_recorded_token(self) -> None:
        sb._probe_failure(sb._PROBE_STEP_NEWNS, errno.EPERM)
        sb._probe_failure("fork", errno.EAGAIN)
        assert sb._last_unshare_remedy == ""

    def test_unavailable_remedy_is_empty_when_the_last_probe_succeeded(self) -> None:
        sb._probe_failure(sb._PROBE_STEP_NEWNS, errno.EPERM)
        sb._last_unshare_failure = None
        # The token is only meaningful alongside a recorded failure; reporting a
        # remedy for a host whose probe just passed would be nonsense advice.
        assert sb.unavailable_remedy() == ""

    def test_unavailable_remedy_reports_the_recorded_token(self) -> None:
        sb._last_unshare_failure = (False, "unshare(CLONE_NEWNS) failed with errno 1 (EPERM)")
        sb._last_unshare_remedy = sb.REMEDY_APPARMOR_USERNS
        assert sb.unavailable_remedy() == sb.REMEDY_APPARMOR_USERNS

    def test_reset_backend_clears_the_token(self) -> None:
        sb._last_unshare_failure = (False, "x")
        sb._last_unshare_remedy = sb.REMEDY_APPARMOR_USERNS
        sb.reset_backend()
        assert sb._last_unshare_remedy == ""

    def test_a_deferred_on_loop_probe_reports_no_remedy(self) -> None:
        """The synthetic on-loop transient describes no host mechanism.

        It is emitted WITHOUT probing, so any token on record belongs to an
        earlier failure. Carrying it forward would attach a permanent-looking
        remedy to a condition that clears by itself in milliseconds.
        """
        import asyncio

        sb._last_unshare_remedy = sb.REMEDY_APPARMOR_USERNS

        async def probe_on_loop() -> None:
            sb._probe_unshare()

        # The deferral branch is Linux-only; elsewhere the platform guard runs
        # first and also leaves no remedy, which this same assertion covers.
        asyncio.run(probe_on_loop())
        assert sb._last_unshare_remedy == ""


class TestGuidanceProse:
    """Log/doctor/Slack read the message text, so the prose must move too."""

    def test_apparmor_guidance_names_the_command_that_fixes_it(self) -> None:
        guidance = sb._linux_remedy_guidance(sb.REMEDY_APPARMOR_USERNS)
        assert "kirocrew service install" in guidance
        # Naming the sysctl WITHOUT warning against setting it to 0 would invite
        # disabling a kernel-wide protection to satisfy one application.
        assert "Do NOT set the sysctl to 0" in guidance

    def test_apparmor_guidance_does_not_recommend_aa_exec(self) -> None:
        """`aa-exec -p` must never be offered as the remedy.

        Entering a named profile is not permitted for an unconfined user, and
        aa-exec execs the command unconfined instead of failing — so it reads as
        applied while leaving NEWNS denied. The guidance may mention aa-exec only
        to say it does not work, never as an instruction to run.
        """
        guidance = sb._linux_remedy_guidance(sb.REMEDY_APPARMOR_USERNS)
        assert "cannot fix that" in guidance
        assert "aa-exec -p kirocrew-userns -- kirocrew gateway" not in guidance

    def test_every_token_has_guidance(self) -> None:
        for token in (
            sb.REMEDY_APPARMOR_USERNS,
            sb.REMEDY_MAX_USER_NAMESPACES,
            sb.REMEDY_NO_USER_NS,
            sb.REMEDY_USERNS_DENIED,
        ):
            assert sb._linux_remedy_guidance(token), token

    def test_unknown_token_yields_no_guidance(self) -> None:
        assert sb._linux_remedy_guidance("") == ""
        assert sb._linux_remedy_guidance("something_else") == ""


class TestErrorCarriesRemedy:
    """``SandboxUnavailableError`` is the typed channel to the presentation layer."""

    def test_remedy_defaults_to_empty(self) -> None:
        exc = sb.SandboxUnavailableError("m", kind="no_backend", detail="d")
        assert exc.remedy == ""

    def test_remedy_is_carried_verbatim(self) -> None:
        exc = sb.SandboxUnavailableError(
            "m", kind="no_backend", detail="d", remedy=sb.REMEDY_APPARMOR_USERNS
        )
        assert exc.remedy == sb.REMEDY_APPARMOR_USERNS


class TestCapExhaustionReachability:
    """The `max_user_namespaces` remedy must actually be able to reach the gate.

    `user.max_user_namespaces` exhaustion surfaces as ENOSPC, which is in
    `_TRANSIENT_PROBE_ERRNOS` because it is also what momentary fd/disk pressure
    looks like. A transient verdict strips the remedy and renders the "temporary
    limit, try again" copy — so on the exact hardened host this token names, the
    remedy was unreachable and the user stayed stuck forever.
    """

    @staticmethod
    def _cap_reason() -> str:
        return f"{sb._PROBE_STEP_NEWUSER} failed with errno {errno.ENOSPC} (ENOSPC)"

    def test_first_cap_sighting_stays_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # One ENOSPC could be a busy host that recovers, and caching "no sandbox"
        # for it is the incident a widened transient set already caused once.
        reason = self._cap_reason()
        results = [(False, True, reason), (True, False, "ok")]
        monkeypatch.setattr(sb, "_probe_unshare_once", lambda: results.pop(0))
        monkeypatch.setattr(sb, "_PROBE_TRANSIENT_RETRY_DELAY_SECS", 0)
        assert sb._probe_unshare() is True
        assert sb._last_unshare_failure is None

    def test_cap_repeated_across_both_attempts_becomes_permanent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cap of 0 reproduces byte-for-byte, which is the disambiguator."""
        reason = self._cap_reason()

        def probe() -> tuple[bool, bool, str]:
            # Mirror the real probe: _probe_failure records the token as it goes.
            return sb._probe_failure(sb._PROBE_STEP_NEWUSER, errno.ENOSPC)

        monkeypatch.setattr(sb, "_probe_unshare_once", probe)
        monkeypatch.setattr(sb, "_PROBE_TRANSIENT_RETRY_DELAY_SECS", 0)
        assert sb._probe_unshare() is False
        assert sb._last_unshare_failure == (False, reason), "must be PERMANENT"
        assert sb._last_unshare_remedy == sb.REMEDY_MAX_USER_NAMESPACES

    def test_repeated_non_cap_transient_stays_transient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the cap is promoted; a repeated EAGAIN keeps retry semantics.

        Widening this to every repeated transient errno would re-cache genuine
        resource pressure as a host verdict, which is what the split exists to
        prevent.
        """

        def probe() -> tuple[bool, bool, str]:
            return sb._probe_failure("fork", errno.EAGAIN)

        monkeypatch.setattr(sb, "_probe_unshare_once", probe)
        monkeypatch.setattr(sb, "_PROBE_TRANSIENT_RETRY_DELAY_SECS", 0)
        assert sb._probe_unshare() is False
        transient, _reason = sb._last_unshare_failure or (False, "")
        assert transient is True, "a repeated fork EAGAIN must stay retryable"

    def test_promoted_cap_delivers_the_remedy_through_wrap_argv(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: the remedy the table advertises reaches the raised error."""
        monkeypatch.setattr(sb, "detect_backend", lambda *_a, **_k: "none")
        monkeypatch.setattr(sb, "_allow_unsandboxed_exec", lambda: False)
        monkeypatch.setattr(sb, "_inside_macos_sandbox", lambda: False)
        sb._last_unshare_failure = (False, self._cap_reason())
        sb._last_unshare_remedy = sb.REMEDY_MAX_USER_NAMESPACES

        with pytest.raises(sb.SandboxUnavailableError) as caught:
            sb.wrap_argv(["/bin/true"])

        # kind must be no_backend, because the gate only renders a remedy for that.
        assert caught.value.kind == "no_backend"
        assert caught.value.remedy == sb.REMEDY_MAX_USER_NAMESPACES
        assert "user.max_user_namespaces" in str(caught.value)


class TestWrapArgvWiring:
    """End-to-end: a recorded probe verdict must reach the raised refusal.

    This is the wiring the gate depends on. Asserting the classifier in isolation
    would not catch the token being dropped between ``_probe_failure`` and the
    exception, which is where every earlier attempt at this diagnosis was lost.
    """

    @staticmethod
    def _force_no_backend(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sb, "detect_backend", lambda *_a, **_k: "none")
        monkeypatch.setattr(sb, "_allow_unsandboxed_exec", lambda: False)
        # Off the macOS nesting path, so the verdict is a real host verdict.
        monkeypatch.setattr(sb, "_inside_macos_sandbox", lambda: False)

    def test_apparmor_verdict_reaches_the_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._force_no_backend(monkeypatch)
        sb._last_unshare_failure = (
            False,
            f"{sb._PROBE_STEP_NEWNS} failed with errno 1 (EPERM)",
        )
        sb._last_unshare_remedy = sb.REMEDY_APPARMOR_USERNS

        with pytest.raises(sb.SandboxUnavailableError) as caught:
            sb.wrap_argv(["/bin/true"])

        assert caught.value.remedy == sb.REMEDY_APPARMOR_USERNS
        # The prose must move with the token: doctor, the gateway logs and Slack
        # all read the message, and only the dashboard reads the token.
        assert "kirocrew service install" in str(caught.value)

    def test_a_transient_verdict_carries_no_remedy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A momentary failure is not a host misconfiguration. Attaching a remedy
        # here would tell the operator to reconfigure a host that is fine.
        self._force_no_backend(monkeypatch)
        sb._last_unshare_failure = (True, "fork failed with errno 11 (EAGAIN)")
        sb._last_unshare_remedy = sb.REMEDY_APPARMOR_USERNS  # stale on purpose

        with pytest.raises(sb.SandboxUnavailableError) as caught:
            sb.wrap_argv(["/bin/true"])

        assert caught.value.kind == "transient"
        assert caught.value.remedy == ""
        assert "kirocrew service install" not in str(caught.value)
