
import logging
import shutil
import threading
import time
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from iantirta.telemetry import netsvc

from iantirta.karaoke.api.karaoke_iantirta_api import IantirtaAPI
from iantirta.karaoke.karaoke import BatchKaraoke

if TYPE_CHECKING:
    from iantirta.media.cloud_storage.base_storage import StorageMixin

    from iantirta.karaoke.core.main import KaraokeTask


_Sentinel = object()
_DefaultDomainClaimTask = [("status", "=", "waiting")]


class KaraokeWorker:
    def __init__(
        self,
        *,
        api_url: str,
        api_token: str,
        worker_id: int,
        storage_options: dict,
        claim_task_domain: list = _DefaultDomainClaimTask,
        process_task_limit: int = 200,
        max_download_workers: int = 5,
        max_cpu_workers: int = 1,
        max_gpu_pool: int = 1,
        max_upload_workers: int = 5,
        max_batch_size: int = 15,           # ETA: 15 - 30 minutes
        max_wait_seconds: float = 15 * 60.0,     # 1 - 15 task
        **kwargs
    ):
        # Server
        self.api = IantirtaAPI(
            api_url, api_token, worker_id
        )

        # Storage
        provider = storage_options.pop('provider')
        self.storage_api: StorageMixin
        if provider == 'gdrive':
            from iantirta.media import DriveStorage
            self.storage_api = DriveStorage(**storage_options)
        elif provider == 'hf_dataset':
            from iantirta.media import HFDatasetStorage
            self.storage_api = HFDatasetStorage(**storage_options)
        else:
            raise NotImplementedError()
        self.storage_provider = provider

        self.log_entries = []

        self.logger = logging.getLogger(f"Worker({worker_id})")
        self.logger.setLevel(logging.DEBUG)
        if not hasattr(self.logger, 'add_hook'):
            from iantirta.telemetry import netsvc
            netsvc.setup_logger("iantirta")
        self.logger.add_hook(self.store_log_entries)

        self.upload_pool = ThreadPoolExecutor(max_upload_workers)

        if not claim_task_domain:
            claim_task_domain = _DefaultDomainClaimTask
        self.claim_task_domain = claim_task_domain

        self.process_task_limit = process_task_limit

        self.max_download_workers = max_download_workers
        self.max_cpu_workers = max_cpu_workers
        self.max_gpu_pool = max_gpu_pool

        self.lock = threading.RLock()
        self.task_buffer: list[KaraokeTask] = []
        self.last_flush_time = time.time()
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_seconds

    def store_log_entries(self, record: logging.LogRecord) -> None:
        log_data = {
            'name': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            'path': Path(record.pathname).as_posix(),
            'line': record.lineno,
            'func': record.funcName,
        }
        self.log_entries.append(log_data)

    def _flush_task_to_server(self):
        """ Will flush data to iantirta server,
            Any error while doing this, shouldn't
            be redirect to task error, but worker error
            instead.
        """
        self.logger.debug(f"Flushing buffer data to server, length={len(self.task_buffer)}...")
    
        if not self.task_buffer:
            return

        try:
            self.api.make_json_request(
                'karaoke/task/update',
                {'tasks': [task.to_dict() for task in self.task_buffer]},
                logger=self.logger,
            )
        except Exception as e:
            self.logger.error(f"Error while posting update to server: {e}",)

    def _flush_upload_to_storage(self):
        self.logger.debug(f"Flushing staging dir, path={self.storage_api.staging_dir.as_posix()}...")
        self.storage_api.upload_folder(
            folder_path=self.storage_api.staging_dir,
            logger=self.logger,
        )

    def flush(self) -> None:
        
        with self.lock:
            if not self.task_buffer:
                return

            self.logger.debug(f"Flushing task buffer data ({len(self.task_buffer)} tasks)...")

            # Try storage upload first
            try:
                # We only attempt to upload if there are actual physical files staged.
                # (If the buffer ONLY contains error tasks, the staging dir might not exist/be empty)
                if self.storage_api.staging_dir.exists():  # c
                    self._flush_upload_to_storage()
            except Exception as e:
                self.logger.error(f"Critical error while flushing storage: {e}")
                # Revert successful tasks in the buffer to 'error' 
                # because the files failed to upload.
                self.logger.debug(f"Current task buffer: {self.task_buffer}")
                for task in self.task_buffer:
                    task.error('_flush_upload_to_storage', e)

            # Flush task should not record task error
            # Rather it belong to the worker
            self._flush_task_to_server()

            # remove data
            shutil.rmtree(self.storage_api.staging_dir, ignore_errors=True)
            self.task_buffer.clear()
            self.last_flush_time = time.time()
            
    def flush_if_needed(self) -> None:
        with self.lock:
            time_elapsed = time.time() - self.last_flush_time
            if len(self.task_buffer) >= self.max_batch_size or time_elapsed >= self.max_wait_seconds:
                self.flush()

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
            # I/O shouldnt need a lock
            chunk_metadata: dict = self.storage_api.make_chunk_file(
                src_path=task.karaoke_path,
                target_path=task.artist,
            )
            chunk_metadata['provider'] = self.storage_provider
            task.download_uris = chunk_metadata

            with self.lock:
                self.task_buffer.append(task)
                self.flush_if_needed()

        except Exception as e:
            self.logger.warning(
                "Exception when trying to upload and post "
                f"Task ({task.id}) Error: {e}"
            )
            task.error('_upload_and_post', e)
            self.task_error_hook(task)

    def _post_error_task(self, task: 'KaraokeTask'):
        self.task_buffer.append(task)
        self.flush_if_needed()

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

                # Flush if any
                self.flush()

                if once:
                    break
        finally:
            self.upload_pool.shutdown(wait=True, cancel_futures=False)
            self.flush()
            self.worker_stop()
