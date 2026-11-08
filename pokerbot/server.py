"""Minimal ASGI plumbing for the tournament judge.

Why this exists
---------------
The judge charges each bot `wall_time - 1 ms` per HTTP request against an
8-second-per-half bank.  Measured on the development machine:

    stock FastAPI agent.py   2.11 ms charged/call  -> 10.6 s per half (forfeit)
    this module              0.66 ms charged/call  ->  3.3 s per half
    bare transport floor     0.65 ms charged/call

So the stock plumbing alone can exhaust the bank in an aggressive match, while
this module costs the same as doing nothing at all.  Every decision we are
allowed to make has to be paid for out of that difference.

What makes it cheap:
  * a raw ASGI callable, so no Starlette routing, no Request object, no
    pydantic validation;
  * orjson, with the four possible /act replies pre-serialised to constant
    bytes, so the happy path never serialises anything;
  * the judge's observation dict is used directly instead of being rebuilt
    into Observation / ObservedAction dataclasses;
  * uvicorn with httptools (2x faster than h11, measured) and uvloop where
    available, with the server/date headers switched off.

The HTTP contract is byte-for-byte what the stock agent.py produces, so the
judge, test_agent.py and tools/smoke_test.py cannot tell the difference.
"""
from __future__ import annotations

import os
import traceback

try:
    from orjson import dumps as _dumps
    from orjson import loads as _loads
except ImportError:  # pragma: no cover - orjson is in requirements.txt
    import json

    _loads = json.loads

    def _dumps(obj):
        return json.dumps(obj).encode()


JSON_HEADERS = [(b"content-type", b"application/json")]

REPLY = {name: b'{"action":"' + name.encode() + b'","hand_end_errors":[]}'
         for name in ("FOLD", "CHECK_CALL", "RAISE_HALF", "RAISE_POT")}
OK = b'{"ok":true,"hand_end_errors":[]}'
ERR_TOKEN = b'{"error":"bad or missing X-Agent-Token"}'
ERR_NOT_STARTED = b'{"error":"call /new-half first"}'
ERR_BODY = b'{"error":"the request body must be a JSON object"}'
ERR_ROUTE = b'{"error":"no such route"}'


def make_app(ns):
    """Build the ASGI app.

    `ns` is the agent module's own globals() dict.  Every callback and every
    setting is read out of it at call time rather than captured, so the tests
    that monkeypatch `module.choose_action` or `module.TOKEN` still take
    effect (setting a module attribute writes into this same dict).

    globals() is used rather than sys.modules[__name__] because test_agent.py
    loads agent.py through importlib without registering it in sys.modules.
    """

    async def app(scope, receive, send):
        if scope["type"] != "http":
            if scope["type"] == "lifespan":
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    elif message["type"] == "lifespan.shutdown":
                        await send({"type": "lifespan.shutdown.complete"})
                        return
            return

        path = scope["path"]
        status = 200

        if path == "/":
            payload = _dumps({"participant": ns["PARTICIPANT"],
                              "agent_name": ns["AGENT_NAME"]})
        else:
            payload = None
            token = ns["TOKEN"]
            if token is not None:
                sent = None
                for key, value in scope["headers"]:
                    if key == b"x-agent-token":
                        sent = value
                        break
                if sent is None or sent.decode("latin-1") != token:
                    status, payload = 401, ERR_TOKEN
            if payload is None and not ns["STATUS"]["started"] and path != "/new-half":
                status, payload = 409, ERR_NOT_STARTED

            if payload is None:
                body = b""
                while True:
                    message = await receive()
                    chunk = message.get("body")
                    if chunk:
                        body = chunk if not body else body + chunk
                    if not message.get("more_body"):
                        break
                try:
                    data = _loads(body)
                    if type(data) is not dict:
                        raise ValueError("not an object")
                except Exception:
                    status, payload = 400, ERR_BODY
                else:
                    try:
                        if path == "/act":
                            errors = _deliver(ns, data.get("hand_results"))
                            action = ns["choose_action"](data["observation"], ns["MEMORY"])
                            if errors:
                                payload = _dumps({"action": _name(action),
                                                  "hand_end_errors": errors})
                            else:
                                payload = REPLY.get(action)
                                if payload is None:
                                    payload = _dumps({"action": _name(action),
                                                      "hand_end_errors": []})
                        elif path == "/new-half":
                            ns["MEMORY"].clear()
                            ns["STATUS"]["started"] = True
                            ns["new_half"](data.get("config") or {}, ns["MEMORY"])
                            payload = OK
                        elif path == "/hand-end":
                            errors = _deliver(ns, data.get("results"))
                            payload = (_dumps({"ok": True, "hand_end_errors": errors})
                                       if errors else OK)
                        else:
                            status, payload = 404, ERR_ROUTE
                    except Exception as exc:
                        status = 500
                        payload = _dumps({"error": f"{type(exc).__name__}: {exc}",
                                          "traceback": traceback.format_exc(),
                                          "hand_end_errors": []})

        await send({"type": "http.response.start", "status": status, "headers": JSON_HEADERS})
        await send({"type": "http.response.body", "body": payload})

    return app


def _name(action):
    """Whatever the strategy returned, in the form the judge expects."""
    if type(action) is str:
        return action
    name = getattr(action, "name", None)
    if name is not None:
        return name
    if isinstance(action, (int, float, bool)) or action is None:
        return action
    return repr(action)


def _deliver(ns, results):
    """Run on_hand_end for each finished hand; collect per-hand crashes."""
    errors = []
    if results:
        for raw in results:
            try:
                ns["on_hand_end"](raw, ns["MEMORY"])
            except Exception as exc:
                errors.append({"error": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()})
    return errors


def serve(app, port: int, host: str = "0.0.0.0") -> None:
    """Run uvicorn with the fastest transport available on this machine."""
    import uvicorn

    options = {}
    try:
        import httptools  # noqa: F401
        options["http"] = "httptools"
    except ImportError:
        pass
    try:
        import uvloop  # noqa: F401
        options["loop"] = "uvloop"
    except ImportError:
        pass
    uvicorn.run(app, host=host, port=port,
                log_level="error", access_log=False, server_header=False,
                date_header=False, lifespan="off", **options)
