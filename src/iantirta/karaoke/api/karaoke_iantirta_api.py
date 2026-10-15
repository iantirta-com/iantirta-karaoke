
import logging

from iantirta.media import BaseHTTP

_logger = logging.getLogger(__name__)


class IantirtaAPI(BaseHTTP):
    endpoint = "https://karaoke.iantirta.com"

    def __init__(
        self,
        api_url: str,
        api_token: str,
        worker_id: int,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
    
        if api_url is not None:
            self.endpoint = api_url

        self.worker_id = worker_id
        self.access_token = api_token

    def _make_payload(self, params):
        payload = {
            "worker_id": self.worker_id,
            "access_token": self.access_token,
        }
        payload.update(params)
        return {"params": payload}

    def raise_for_status(self, res):
        try:
            res.raise_for_status()
        except Exception:
            raise

        res = res.json()
        if error := res.get("error"):
            error_data = error.get("data", {})
            raise OSError(
                f"\nIantirta API Error ({error.get('code')}) {error.get('message')}:\n"
                f"ErrorType: {error_data.get('name')}\n"
                f"Message: {error_data.get('message')}\n"
                f"Traceback: {error_data.get('debug')}"
            )

    def make_json_request(self, path: str, params: dict, *, logger: logging.Logger | None = None):
        payload = {
            "worker_id": self.worker_id,
            "access_token": self.access_token,
        }
        payload.update(params)

        safe_payload = payload.copy()
        safe_payload.pop("worker_id")
        safe_payload.pop("access_token")
        (logger or _logger).debug(f"Sending payload to ({path}) with {safe_payload}")

        res = self.request('get', path, json={"params": payload}, headers={"Content-Type": "application/json"})
        self.raise_for_status(res)

        return res.json()['result']

