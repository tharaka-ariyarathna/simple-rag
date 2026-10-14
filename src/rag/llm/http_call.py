"""The one place that turns an HTTP call to an LLM provider into an LlmResult-ready dict or an LlmError.
Every provider client uses it, so they all decide "fall back or stop" the same way."""
import httpx
from rag.llm.base import LlmConfigurationError, LlmError, LlmTimeoutError, LlmUnavailableError

# Another try (or another provider) may succeed: out of credits, request timeout, over a token-per-minute
# limit, rate limited. Every 5xx is recoverable too. Any other 4xx (400 bad request, 401/403 bad key,
# 404 unknown model or model not pulled) is our mistake and becomes LlmConfigurationError.
RECOVERABLE_STATUS_CODES = {402, 408, 413, 429}
ERROR_BODY_MAX_CHARS = 500  # keeps a provider's long error page out of the logs


def error_for_status(provider: str, model_name: str, status_code: int, body: str) -> LlmError:
    message = f"{provider} returned {status_code} for model '{model_name}': {body[:ERROR_BODY_MAX_CHARS]}"
    if status_code in RECOVERABLE_STATUS_CODES or status_code >= 500:
        return LlmUnavailableError(message)
    return LlmConfigurationError(message)


def post_json(http_client: httpx.Client, path: str, request_body: dict, provider: str, model_name: str) -> dict:
    """POSTs the request and returns the parsed JSON body, or raises the matching LlmError."""
    # TimeoutException is a subclass of HTTPError, so it must be caught first
    try:
        response = http_client.post(path, json=request_body)
    except httpx.TimeoutException as error:
        raise LlmTimeoutError(f"{provider} did not answer within {http_client.timeout.read} seconds") from error
    except httpx.HTTPError as error:  # connection refused, DNS, ... -> server not running or unreachable
        raise LlmUnavailableError(f"Could not reach {provider} at {http_client.base_url}: {error}") from error
    if response.is_error:
        raise error_for_status(provider, model_name, response.status_code, response.text)
    try:
        return response.json()
    except ValueError as error:
        raise LlmUnavailableError(f"Unexpected response from {provider}: not JSON ({error})") from error
