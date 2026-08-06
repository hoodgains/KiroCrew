"""Session-pin truthfulness: the pin's effective scope must be reported, not asserted.

Two halves, and the seam between them is the point:

* ``is_proxied_loopback_request`` must separate "the peer is a same-host proxy"
  from the other reason ``is_direct_local_request`` returns False (a genuinely
  non-loopback client). Only the first collapses a per-address binding.
* The Security Posture ``token_auth`` row must render three distinct states and
  never render "not observed yet" as if the control were effective — the same
  failure mode as a check that never ran being drawn as a check that passed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kiro_crew.dashboard.origin import is_direct_local_request, is_proxied_loopback_request
from kiro_crew.security_posture import build_posture_snapshot


def _req(remote: str, headers: dict[str, str] | None = None):
    """Minimal request double: both predicates read only .remote and .headers."""
    return SimpleNamespace(remote=remote, headers=headers or {})


class TestProxiedLoopbackPredicate:
    def test_plain_loopback_is_not_proxied(self) -> None:
        assert is_proxied_loopback_request(_req("127.0.0.1")) is False

    @pytest.mark.parametrize(
        "header",
        ["Forwarded", "X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto", "X-Real-IP"],
    )
    def test_loopback_with_any_forwarding_header_is_proxied(self, header: str) -> None:
        assert is_proxied_loopback_request(_req("127.0.0.1", {header: "203.0.113.7"})) is True

    def test_ipv6_loopback_counts(self) -> None:
        assert is_proxied_loopback_request(_req("::1", {"X-Forwarded-For": "203.0.113.7"})) is True

    def test_non_loopback_peer_is_NOT_reported_as_proxied(self) -> None:
        """The distinction this predicate exists for.

        A widened bind (KIROCREW_BIND) gives a real client address in
        ``request.remote``, so the pin is per-client and must not be reported as
        collapsed — even though ``is_direct_local_request`` is False here too.
        """
        req = _req("100.64.1.9", {"X-Forwarded-For": "203.0.113.7"})
        assert is_direct_local_request(req) is False
        assert is_proxied_loopback_request(req) is False

    def test_empty_remote_is_not_proxied(self) -> None:
        assert is_proxied_loopback_request(_req("")) is False


def _pin_detail() -> str:
    control = next(c for c in build_posture_snapshot()["controls"] if c["key"] == "token_auth")
    return next(i["detail"] for i in control["items"] if i["label"] == "IP pinning")


@pytest.fixture(autouse=True)
def _reset_pin_latches():
    """Isolate the observation latches — they are process-global by design."""
    from kiro_crew.dashboard import token_auth

    state = token_auth._state
    before = (state._pin_bound_ever, state._proxied_pin_observed)
    state._pin_bound_ever = False
    state._proxied_pin_observed = False
    yield
    state._pin_bound_ever, state._proxied_pin_observed = before


class TestPostureReportsEffectivePinScope:
    def test_unbound_is_reported_as_not_known_yet_not_as_effective(self) -> None:
        """A pin nobody has exercised is not evidence the pin works."""
        from kiro_crew.dashboard.token_auth import proxied_pin_observed

        assert proxied_pin_observed() is None
        detail = _pin_detail()
        assert "not known yet" in detail
        # Must not claim the per-client property it has not observed.
        assert "client address that first used it" not in detail

    def test_direct_bind_reports_per_client(self) -> None:
        from kiro_crew.dashboard.token_auth import bind_token_ip, proxied_pin_observed

        bind_token_ip("t-direct", "203.0.113.7", 0.0, False)
        assert proxied_pin_observed() is False
        assert "client address that first used it" in _pin_detail()

    def test_proxied_bind_reports_shared_pin(self) -> None:
        """The state the guide used to advertise as a mitigation."""
        from kiro_crew.dashboard.token_auth import bind_token_ip, proxied_pin_observed

        bind_token_ip("t-proxied", "127.0.0.1", 0.0, True)
        assert proxied_pin_observed() is True
        detail = _pin_detail()
        assert "SHARED, not per-client" in detail
        assert "same-host proxy" in detail

    def test_proxied_observation_latches(self) -> None:
        """A later direct bind does not un-share the sessions already bound."""
        from kiro_crew.dashboard.token_auth import bind_token_ip, proxied_pin_observed

        bind_token_ip("t-proxied", "127.0.0.1", 0.0, True)
        bind_token_ip("t-direct", "203.0.113.7", 0.0, False)
        assert proxied_pin_observed() is True

    def test_eviction_does_not_reset_a_known_answer(self) -> None:
        """Losing the bindings must not turn 'known' back into 'never observed'."""
        import time

        from kiro_crew.dashboard import token_auth

        token_auth.bind_token_ip("t-direct", "203.0.113.7", time.time() + 60, False)
        token_auth._state.evict_expired(time.time() + 3600)
        assert token_auth.proxied_pin_observed() is False

    def test_observation_never_changes_the_binding_itself(self) -> None:
        """The flag is reporting only — check_ip must be unaffected by it."""
        from kiro_crew.dashboard.token_auth import bind_token_ip, check_token_ip

        bind_token_ip("t-proxied", "127.0.0.1", 0.0, True)
        assert check_token_ip("t-proxied", "127.0.0.1") is True
        assert check_token_ip("t-proxied", "203.0.113.7") is False
