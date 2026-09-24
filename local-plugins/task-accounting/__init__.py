"""Standalone, opt-in, observer-only Hermes plugin."""
from functools import partial
import queue
import threading
import time

from .collector import EVENTS, sanitize, store


class Observer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.queue = queue.Queue(maxsize=128)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.worker = None
        self.dropped = 0
        self.failed = 0

    def observe(self, event, **kwargs):
        # Never return a directive or log an exception/payload. Default is exactly False.
        try:
            if self.stop.is_set() or self.ctx.get_config('enabled', False) is not True:
                return None
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
            row = sanitize(event, kwargs)
            if not self.lock.acquire(blocking=False):
                self.dropped += 1
                return None
            try:
                if self.worker is None:
                    from agent.memory_provider import spawn_context_thread
                    self.worker = spawn_context_thread(self._run, name='task-accounting-writer')
                    self.worker.start()
                self.queue.put_nowait((home, row))
            finally:
                self.lock.release()
        except Exception:
            self.dropped += 1
        return None

    def _run(self):
        while not self.stop.is_set() or not self.queue.empty():
            try:
                home, row = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                store(home, row)
            except Exception:
                self.failed += 1
            finally:
                self.queue.task_done()

    def flush(self, timeout=1.0):
        """Operator/test drain only. Hooks never wait for disk or this barrier."""
        deadline = time.monotonic() + timeout
        while self.queue.unfinished_tasks and time.monotonic() < deadline:
            self.stop.wait(0.002) if not self.stop.is_set() else time.sleep(0.002)
        return not self.queue.unfinished_tasks

    def close(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join(timeout=0.1)


def register(ctx):
    observer = Observer(ctx)
    for event in (*EVENTS, 'subagent_start', 'subagent_stop'):
        ctx.register_hook(event, partial(observer.observe, event))
    ctx.on_unload(observer.close)
