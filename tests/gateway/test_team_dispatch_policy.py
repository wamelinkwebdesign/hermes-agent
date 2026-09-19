"""Team configuration and authenticated deterministic ownership contracts."""
from types import SimpleNamespace

import pytest


def config():
    return {"team_dispatch": {"enabled": True, "coordinator_profile": "default",
            "specialist_profiles": ["revenue", "product", "design", "engineering", "finance-ops"],
            "destinations": [{"chat_id": "-100", "thread_id": "7"}]}}


def test_policy_is_opt_in_strict_and_legacy_exclusive():
    from gateway.team_dispatch_policy import TeamPolicy
    assert TeamPolicy.from_extra({}) is None
    raw = config()
    policy = TeamPolicy.from_extra(raw)
    assert policy and policy.matches("-100", "7") and not policy.matches("-200", "7")
    assert policy.fingerprint == TeamPolicy.from_extra(config()).fingerprint
    for key, value in [("enabled", "true"), ("coordinator_profile", "engineering"),
                       ("specialist_profiles", ["engineering"]),
                       ("destinations", [{"chat_id": -100, "thread_id": "7"}])]:
        bad = config()
        bad["team_dispatch"][key] = value
        with pytest.raises(ValueError):
            TeamPolicy.from_extra(bad)
    raw["public_handoff"] = {"enabled": True, "chat_id": "-100", "thread_id": "7"}
    with pytest.raises(ValueError, match="overlap"):
        TeamPolicy.from_extra(raw)


def test_reply_identity_precedes_mentions_and_quoted_mentions_do_not_address():
    from gateway.team_dispatch_policy import addressed_owner
    members = {"default": {"bot_id": "900", "username": "ace_bot"},
               "engineering": {"bot_id": "901", "username": "woz_bot"}}
    def msg(text, entities=(), reply=None):
        return SimpleNamespace(text=text, entities=entities, caption=None,
                               reply_to_message=reply)
    reply = SimpleNamespace(from_user=SimpleNamespace(id=901, is_bot=True))
    assert addressed_owner(msg("@ace_bot", reply=reply), members) == "engineering"
    assert addressed_owner(msg('Quoted "@woz_bot"'), members) is None
    assert addressed_owner(msg("@woz_bot help"), members) == "engineering"
    assert addressed_owner(msg("/help@woz_bot"), members) == "engineering"
    assert addressed_owner(msg("@woz_bot @ace_bot help"), members) == "default"
    astral = msg("👋 @woz_bot help", [SimpleNamespace(type="mention", offset=3, length=8)])
    assert addressed_owner(astral, members) == "engineering"
    assert addressed_owner(msg("random", [SimpleNamespace(type="mention", offset="bad", length=2)]), members) is None
