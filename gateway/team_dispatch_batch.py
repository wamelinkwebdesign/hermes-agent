"""Reuse Telegram's real debounce/hold path with internal-only batch identity."""
from gateway.platforms.event import MessageType


def pending_reply(adapter, chat, thread, replied):
    """Resolve authenticated aliases before the real debounce flush."""
    from dataclasses import replace
    for pending in adapter._pending_text_batches.values():
        prior = getattr(pending, "_team_request", None)
        if (prior and (prior.chat_id, prior.thread_id, prior.sender_id) ==
                (chat, thread, str(getattr(replied.from_user, "id", "")))
                and str(replied.message_id) in pending._team_input_ids):
            return replace(prior, brief=pending.text, input_ids=tuple(pending._team_input_ids))
    return None


def queue_request(adapter, request, message):
    event = adapter._build_message_event(message, MessageType.TEXT)
    event.text = request.brief
    event._team_request = request
    event._team_input_ids = (request.message_id,)
    event._team_batch_key = "team:" + request.request_id
    # Only likely Telegram client splits coalesce. Ordinary unrelated roots never
    # acquire a chat-wide sticky owner or get merged just because they arrived fast.
    for key, pending in adapter._pending_text_batches.items():
        prior = getattr(pending, "_team_request", None)
        if prior is None:
            continue
        same_source = (prior.chat_id, prior.thread_id, prior.sender_id) == (
            request.chat_id, request.thread_id, request.sender_id)
        if not same_source:
            continue
        if request.message_id in pending._team_input_ids:
            return
        if (request.reason_code == prior.reason_code == "pending"
                and request.reply_to_message_id is None and prior.reply_to_message_id is None
                and getattr(pending, "_last_chunk_len", 0) >= adapter._SPLIT_THRESHOLD
                and int(request.message_id) == int(pending._team_input_ids[-1]) + 1
                and len(pending._team_input_ids) < 32):
            event._team_batch_key = key
            break
    adapter._enqueue_text_event(event)


def merge_ids(existing, event):
    new = tuple(dict.fromkeys((*existing._team_input_ids, *event._team_input_ids)))
    if new == existing._team_input_ids:
        return False
    if len(new) > 32:
        raise ValueError("team_batch_overflow")
    existing._team_input_ids = new
    return True
