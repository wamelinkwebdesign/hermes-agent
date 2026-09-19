"""Observe the serialized SDK HTTP request, without a live model connection."""
import json

import httpx
from openai import BadRequestError
import pytest

from gateway.team_dispatch_agent import SELECTION_SCHEMA



@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_exact_configured_codex_route_preserves_strict_wire_schema(monkeypatch, async_mode):
    from agent.auxiliary_client import resolve_provider_client
    seen = []

    def capture(request):
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(400, request=request, json={"error": {"message": "Synthetic boundary captured request"}})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, request: capture(request))

    async def capture_async(self, request):
        return capture(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", capture_async)
    client, model = resolve_provider_client(
        "openai-codex", model="gpt-5.3-codex", async_mode=async_mode,
        explicit_base_url="https://codex.invalid/v1", explicit_api_key="synthetic-key",
        api_mode="codex_responses")
    kwargs = dict(model=model, messages=[{"role": "user", "content": "PUBLIC_ONLY"}], tools=[], stream=False,
                  response_format={"type": "json_schema", "json_schema": {
                      "name": "team_owner", "strict": True, "schema": SELECTION_SCHEMA}})
    assert client is not None
    with pytest.raises(BadRequestError):
        if async_mode:
            await client.chat.completions.create(**kwargs)
        else:
            client.chat.completions.create(**kwargs)
    assert len(seen) == 1
    url, payload = seen[0]
    assert url == "https://codex.invalid/v1/responses"
    assert payload["model"] == model == "gpt-5.3-codex"
    assert payload["text"]["format"] == {
        "type": "json_schema", "name": "team_owner", "strict": True, "schema": SELECTION_SCHEMA}
    assert not payload.get("tools")
