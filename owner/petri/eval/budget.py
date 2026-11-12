"""Conservative local guard and per-request API usage ledger."""
import json
from pathlib import Path

import httpx

PRICE = {"gpt-6.1-terra": (1, 0.05, 5, 1.25, 2.50), "gpt-6.1-sol": (2, 0.10, 10, 2.50, 5)}
CAP = 2.0
MAX_REQUESTS = 60


class Budget:
    def __init__(self, directory):
        self.log = Path(directory) / "api-usage.jsonl"
        self.used = 0.0
        self.calls = 0
        if self.log.is_file():
            for row in map(json.loads, self.log.read_text().splitlines()):
                if row["event"] == "request":
                    self.calls = max(self.calls, row["number"])
                    self.used += row["reserved_usd"]
                elif "accounted_usd" in row:
                    self.used = row["accounted_usd"]

    def record(self, value):
        with self.log.open("a") as out:
            out.write(json.dumps(value) + "\n")

    def transport(self, condition, model):
        return GuardedTransport(self, condition, model)


class GuardedTransport(httpx.AsyncHTTPTransport):
    def __init__(self, budget, condition, model):
        super().__init__(retries=0)
        self.budget = budget
        self.condition = condition
        self.model = model

    async def handle_async_request(self, request):
        if (request.url.scheme, request.url.host, request.url.path, request.method) != (
            "https", "api.openai.com", "/v1/responses", "POST",
        ):
            raise RuntimeError("Unexpected endpoint, blocked before sending.")
        raw = await request.aread()
        payload = json.loads(raw)
        output_limit = payload.get("max_output_tokens")
        if payload.get("model") != self.model or not isinstance(output_limit, int) or not 1 <= output_limit <= 2400:
            raise RuntimeError("Unexpected model or output limit, blocked before sending.")
        if payload.get("stream") or payload.get("previous_response_id") or payload.get("background"):
            raise RuntimeError("Unmetered request shape, blocked before sending.")
        standard, cached, output, cache30, cache12 = PRICE[self.model]
        # Reserve using UTF-8 bytes plus overhead and the most expensive input rate.
        reserve = ((len(raw) + 2048) * max(standard, cache12) + output_limit * output) / 1_000_000
        b = self.budget
        if b.calls >= MAX_REQUESTS or b.used + reserve > CAP:
            b.record({"event": "blocked", "condition": self.condition, "model": self.model,
                      "reserved_usd": reserve, "accounted_usd": b.used})
            raise RuntimeError("The $2 estimated-spend guard blocked a request.")
        b.calls += 1
        number = b.calls
        b.used += reserve
        b.record({"event": "request", "number": number, "condition": self.condition, "model": self.model,
                  "reserved_usd": reserve, "max_output_tokens": output_limit})
        try:
            response = await super().handle_async_request(request)
            await response.aread()
            result = response.json()
        except Exception as error:
            b.record({"event": "unknown", "number": number, "error_type": type(error).__name__,
                      "accounted_usd": b.used})
            raise
        usage = result.get("usage") or {}
        amount = None
        if isinstance(usage.get("input_tokens"), int) and isinstance(usage.get("output_tokens"), int):
            details = usage.get("input_tokens_details") or {}
            hits = details.get("cached_tokens") or 0
            writes = details.get("cache_write_input_tokens") or details.get("cache_write_tokens") or 0
            ttl = (payload.get("prompt_cache_retention") or "").lower()
            uncached = max(0, usage["input_tokens"] - hits - writes)
            amount = (uncached * standard + hits * cached + writes * (cache12 if "12h" in ttl else cache30)
                      + usage["output_tokens"] * output) / 1_000_000
            b.used += amount - reserve
        b.record({"event": "response", "number": number, "condition": self.condition,
                  "model": result.get("model", self.model), "response_id": result.get("id"),
                  "status_code": response.status_code, "api_status": result.get("status"),
                  "request_id": response.headers.get("x-request-id"),
                  "usage": usage, "estimated_usd": amount, "accounted_usd": b.used})
        return response
