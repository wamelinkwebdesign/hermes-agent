"""Offline subprocess gateway. Only model HTTP and Telegram are synthetic."""
import asyncio
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import AsyncMock

import pytest

# The parent's environment is an explicit synthetic allowlist before imports.
from gateway.config import Platform, load_gateway_config
from gateway.run import GatewayRunner
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_team_dispatch_runtime import Bot, message, model

PROTOCOL = sys.stdout


def emit(value):
    PROTOCOL.write(json.dumps(value) + "\n")
    PROTOCOL.flush()


def no_network(*args, **kwargs):
    raise AssertionError("unexpected external socket in offline process test")


async def main():
    profile = sys.argv[1]
    patches = pytest.MonkeyPatch()
    patches.setattr(socket.socket, "connect", no_network)
    calls = model.__wrapped__(patches)
    config = load_gateway_config()
    runner = GatewayRunner(config=config)
    ad = TelegramAdapter(config.platforms[Platform.TELEGRAM])
    ad._bot = Bot(profile)
    ad._running = True
    runner.adapters = {Platform.TELEGRAM: ad}
    runner._running = True
    runner._wire_adapter_handlers(ad)
    native = AsyncMock(return_value=None)
    ad.set_message_handler(native)
    emit({"status": "ready", "pid": os.getpid(), "home": os.environ["HERMES_HOME"],
          "runtime": str(Path(sys.modules['gateway.run_team_dispatch'].__file__).resolve())})
    watcher = None
    try:
        while line := await asyncio.to_thread(sys.stdin.readline):
            command = json.loads(line)
            op = command["cmd"]
            if op == "ingress":
                from types import SimpleNamespace
                msg = message(**command["message"])
                update = SimpleNamespace(message=msg, effective_message=msg, update_id=msg.message_id)
                await ad._handle_text_message(update, None)
            elif op == "tick":
                if ad._pending_text_batch_tasks:
                    await asyncio.gather(*tuple(ad._pending_text_batch_tasks.values()))
                await runner._team_dispatch.tick()
                if runner._team_dispatch.tasks:
                    await asyncio.gather(*tuple(runner._team_dispatch.tasks))
            elif op == "register":
                runner._team_dispatch.register()
            elif op == "watch":
                watcher = asyncio.create_task(runner._handoff_watcher(interval=0.02))
            elif op == "reconnect":
                old = ad
                ad = TelegramAdapter(old.config)
                runner._wire_adapter_handlers(ad)
                await runner._team_dispatch.tick()  # unready must not claim
                ad._bot, ad._running = old._bot, True
                runner._publish_primary_adapter(Platform.TELEGRAM, ad)
                ad.set_message_handler(native)
            elif op == "snapshot":
                store = runner._team_dispatch.store
                with store.connect() as db:
                    rows = [store.decode(r) for r in db.execute("SELECT * FROM public_handoffs ORDER BY created_at")]
                emit({"status": "ok", "rows": rows, "members": store.members(),
                      "calls": calls, "sent": ad._bot.sent, "native": native.call_count,
                      "bot_id": ad._bot.id, "pid": os.getpid(), "home": os.environ["HERMES_HOME"]})
                continue
            elif op == "shutdown":
                runner._running = False
                if watcher:
                    await watcher
                await runner._team_dispatch.close()
                runner._shutdown_event.set()
                emit({"status": "ok"})
                break
            else:
                raise AssertionError(op)
            emit({"status": "ok"})
    finally:
        runner._running = False
        runner._shutdown_event.set()
        patches.undo()


if __name__ == "__main__":
    with redirect_stdout(sys.stderr):
        asyncio.run(main())
