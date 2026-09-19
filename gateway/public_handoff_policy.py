"""Default-off Telegram public handoff policy. No model-derived routing fields."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PublicHandoffPolicy:
    enabled: bool
    chat_id: str
    thread_id: str | None
    coordinator_bot_id: str
    specialist_bot_id: str

    @classmethod
    def from_extra(cls, extra):
        raw = extra.get("public_handoff")
        if not isinstance(raw, dict):
            return None
        chat = raw.get("chat_id")
        coordinator = raw.get("coordinator_bot_id")
        specialist = raw.get("specialist_bot_id")
        if not isinstance(chat, str) or not isinstance(coordinator, str) or not isinstance(specialist, str):
            return None
        thread = raw.get("thread_id")
        if (not re.fullmatch(r"-[1-9][0-9]{0,19}", chat)
                or not all(re.fullmatch(r"[1-9][0-9]{0,19}", v) for v in (coordinator, specialist))
                or coordinator == specialist
                or (thread is not None and (not isinstance(thread, str)
                                           or not re.fullmatch(r"[1-9][0-9]{0,19}", thread)))):
            return None
        return cls(raw.get("enabled") is True, chat, thread, coordinator, specialist)

    def matches(self, adapter, message):
        return (adapter._is_group_chat(message)
                and str(message.chat.id) == self.chat_id
                and adapter._effective_message_thread_id(message) == self.thread_id)


def native_excludes_self(adapter, message):
    policy = PublicHandoffPolicy.from_extra(adapter.config.extra)
    if not policy or not policy.enabled or not policy.matches(adapter, message):
        return False
    # Direct addressing wins, including explicit addressing of this bot in a reply.
    if adapter._message_mentions_bot(message):
        return False
    replied = getattr(message, "reply_to_message", None)
    author = getattr(replied, "from_user", None)
    owner = str(getattr(author, "id", ""))
    own = str(getattr(adapter._bot, "id", ""))
    if owner in {policy.coordinator_bot_id, policy.specialist_bot_id} and owner != own:
        return True
    # The approved topic's ambient traffic belongs to Ace even if the specialist
    # has free-response/wake-word triggers. This is independent of arrival order
    # or whether Ace has already persisted a selected root. Own native replies
    # remain available; known public replies are intercepted by ingress first.
    return own == policy.specialist_bot_id and not adapter._is_reply_to_bot(message)
