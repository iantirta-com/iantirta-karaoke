
import logging
from typing import TYPE_CHECKING, Any
from concurrent.futures import ThreadPoolExecutor

from iantirta.karaoke.karaoke import BatchKaraoke
from iantirta.media import DriveStorage
from iantirta.karaoke.api.karaoke_iantirta_api import IantirtaAPI

if TYPE_CHECKING:
    from iantirta.karaoke.core.main import KaraokeTask
    from iantirta.media import DriveMetadata


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
        self.drive_storage = DriveStorage(
            user_token, root_folder_id
        )

        self.log_entries = []

        self.logger = logging.getLogger(f"Worker({worker_id})")
        self.logger.setLevel(logging.DEBUG)
        self.logger.add_hook(self.store_log_entries)

        self.network_pool = ThreadPoolExecutor(5)

    def store_log_entries(self, record) -> None:
        root_handler = logging.getLogger().handlers[0]
        msg = root_handler.formatter.format(record)
        self.log_entries.append(msg)

    def claim_task(
        self,
        domain: list[tuple] | None = None,
        limit: int | None = 200,
        *,
        legacy: bool = True
    ) -> list[dict[str, Any]]:
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

    def worker_claim_task(self, tasks: list[dict[str, Any]]):
        return self.api.make_json_request(
            '/karaoke/worker/claim_task',
            {'task_ids': [task['id'] for task in tasks]}
        )

    def worker_start(self):
        return self.api.make_json_request("/karaoke/worker/start", {})

    def worker_stop(self):
        return self.api.make_json_request(
            "/karaoke/worker/stop",
            {'log_entries': self.log_entries}
        )

    def _upload_and_post(self, task: "KaraokeTask"):
        try:
            metadata: DriveMetadata = self.drive_storage.upload(task.karaoke_path, subfolder=task.artist)
            self.api.make_json_request(
                'karaoke/task/update',
                {**task.to_dict(), **metadata.to_dict()}
            )
        except:
            self.task_error_hook(task)

    def _post_error_task(self, task: 'KaraokeTask'):
        try:
            self.api.make_json_request(
                'karaoke/task/update',
                {**task.to_dict()}
            )
        except:
            self.logger.error("Error while posting error to server")
    
    def task_render_hook(self, task: "KaraokeTask") -> KaraokeTask:
        self.network_pool.submit(
            self._upload_and_post, task,
        )
        return task

    def task_error_hook(self, task: "KaraokeTask"):
        self.network_pool.submit(
            self._post_error_task, task
        )

    def run(self):
        try:
            self.worker_start()
            while True:
                if not (tasks := self.claim_task()):
                    self.logger.info("No more tasks to process")
                    break

                self.logger.info(f"Claimed {len(tasks)} task(s).")
                self.worker_claim_task(tasks)

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
            self.network_pool.shutdown(wait=True, cancel_futures=False)
            self.worker_stop()

if __name__ == '__main__':
    from iantirta.telemetry import netsvc
    import os
    netsvc.setup_logger('iantirta')
    api_token=os.environ['IANTIRTA_WORKER_API_TOKEN']
    KaraokeWorker(
        'http://localhost:8069',
        9,
        1,
        None,
        None,
    ).run()