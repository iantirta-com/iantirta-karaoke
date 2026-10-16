
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import TYPE_CHECKING, Any

from iantirta.media import DriveStorage
from iantirta.telemetry import netsvc

from iantirta.karaoke.api.karaoke_iantirta_api import IantirtaAPI
from iantirta.karaoke.karaoke import BatchKaraoke

if TYPE_CHECKING:
    from iantirta.media import DriveMetadata

    from iantirta.karaoke.core.main import KaraokeTask


_Sentinel = object()
_DefaultDomainClaimTask = [("status", "=", "waiting")]


class KaraokeWorker:
    def __init__(
        self,
        api_url: str,
        api_token: str,
        worker_id: int,
        user_token: dict,
        root_folder_id: str,
        *,
        claim_task_domain: list = _DefaultDomainClaimTask,
        process_task_limit: int = 200,
        max_download_workers: int = 5,
        max_cpu_workers: int = 1,
        max_gpu_pool: int = 1,
        max_upload_workers: int = 5,
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

        self.upload_pool = ThreadPoolExecutor(max_upload_workers)

        if not claim_task_domain:
            claim_task_domain = _DefaultDomainClaimTask
        self.claim_task_domain = claim_task_domain

        self.process_task_limit = process_task_limit

        self.max_download_workers = max_download_workers
        self.max_cpu_workers = max_cpu_workers
        self.max_gpu_pool = max_gpu_pool

    def store_log_entries(self, record) -> None:
        msg = netsvc.formatLogMessage(record)
        self.log_entries.append(msg)

    def claim_task(
        self,
        domain: list[tuple] | None = None,
        limit: int | None = _Sentinel,
        *,
        legacy: bool = False
    ) -> list[dict[str, Any]]:
        if not domain:
            domain = self.claim_task_domain

        if limit is _Sentinel:
            limit = self.process_task_limit

        path = '/karaoke/tasks/get'
        params = {
            "domain": domain,
            "limit": limit
        }

        return self.api.make_json_request(
            path, params, logger=self.logger
        )

    def worker_claim_task(self, tasks: list[dict[str, Any]]):
        return self.api.make_json_request(
            '/karaoke/worker/claim_task',
            {'task_ids': [task['id'] for task in tasks]},
            logger=self.logger,
        )

    def worker_start(self):
        return self.api.make_json_request(
            "/karaoke/worker/start", {},
            logger=self.logger,
        )

    def worker_stop(self):
        return self.api.make_json_request(
            "/karaoke/worker/stop",
            {'log_entries': self.log_entries},
            logger=self.logger,
        )

    def _upload_and_post(self, task: "KaraokeTask"):
        try:
            metadata: DriveMetadata = self.drive_storage.upload(task.karaoke_path, subfolder=task.artist, logger=task.logger)
            self.api.make_json_request(
                'karaoke/task/update',
                {'task' : {**task.to_dict(), **metadata.to_dict()}},
                logger=self.logger,
            )
        except Exception as e:
            self.logger.warning(
                "Exception when trying to upload and post "
                f"Task ({task.id}) Error: {e}"
            )
            self.task_error_hook(task)

    def _post_error_task(self, task: 'KaraokeTask'):
        try:
            self.api.make_json_request(
                'karaoke/task/update',
                {'task': {**task.to_dict()}},
                logger=self.logger,
            )
        except Exception as e:
            self.logger.error(f"Error while posting error to server: {e}",)

    def task_render_hook(self, task: "KaraokeTask") -> Future:
        return self.upload_pool.submit(
            self._upload_and_post, task,
        )

    def task_error_hook(self, task: "KaraokeTask") -> Future:
        return self.upload_pool.submit(
            self._post_error_task, task
        )

    def run(self, *, once=False, dry_run=False):
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
                        logger=self.logger,
                        max_gpu_pool=self.max_gpu_pool,
                    )
                    processor.run()
                except Exception as e:
                    self.logger.error(f"Error while doing batch karaoke process: {e}")

                if once:
                    break
        finally:
            self.upload_pool.shutdown(wait=True, cancel_futures=False)
            self.worker_stop()

if __name__ == '__main__':
    import os

    from iantirta.telemetry import netsvc
    netsvc.setup_logger('iantirta')
    api_token=os.environ['IANTIRTA_WORKER_API_TOKEN']
    KaraokeWorker(
        'http://localhost:8069',
        api_token,
        9,
        None,
        None,
    ).run()
