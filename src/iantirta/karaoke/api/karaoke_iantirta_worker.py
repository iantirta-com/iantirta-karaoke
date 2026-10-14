
import logging
from typing import TYPE_CHECKING

from iantirta.karaoke.karaoke import BatchKaraoke

from .karaoke_iantirta_api import IantirtaAPI

if TYPE_CHECKING:
    from iantirta.karaoke.core.main import KaraokeTask

class KaraokeWorker:
    def __init__(
        self,
        api_url: str,
        api_token: str,
        worker_id: int,
        user_token: dict,
        root_folder_id: str,
        **kwargs
    ):
        self.api = IantirtaAPI(
            api_url, api_token, worker_id
        )

        self.logger = logging.getLogger(f"Worker({worker_id})")

    def claim_task(
        self,
        domain: list[tuple] | None = None,
        limit: int | None = 200,
        *,
        legacy: bool = True
    ) -> list:
        if not domain:
            domain = [("status", "=", "waiting")]
        
        path = '/karaoke/tasks/get'        
        params = {
            "domain": domain,
            "limit": limit
        }

        if legacy:
            params["action"] = "get"
            path = '/karaoke/tasks'
        
        return self.api.make_json_request(path, params)

    def worker_start(self):
        return self.api.make_json_request("/karaoke/worker/start")

    def worker_stop(self):
        return self.api.make_json_request("/karaoke/worker/stop")

    def task_render_hook(self, task: "KaraokeTask"):
        pass

    def task_error_hook(self, task: "KaraokeTask"):
        pass

    def run(self):
        try:
            self.worker_start()
            while True:
                if not (tasks := self.claim_task()):
                    self.logger.info("No more tasks to process")
                    break

                self.logger.info(f"Claimed {len(tasks)} task(s).")

                try:
                    processor = BatchKaraoke(
                        tasks,
                        render_hook=self.task_render_hook,
                        error_hook=self.task_error_hook,
                        logger=self.logger
                    )
                    processor.run()
                except Exception as e:
                    self.logger.error(f"Error while doing batch karaoke process: {e}")
        finally:
            self.worker_stop()
        