
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING

from iantirta.media import IanTirtaAPI

if TYPE_CHECKING:
    from iantirta.media.cloud_storage.base_storage import StorageMixin

    from iantirta.karaoke.core.main import KaraokeTask

class KaraokeWorker(IanTirtaAPI):
    endpoint: str = 'https://karaoke.iantirta.com'

    """
    Automation class that can responsible for api communication
    and ``BatchKaraoke`` runner.

    :param str endpoint:
    :param str worker_token:
    :param int worker_id:
        Required for karaoke.iantirta.com worker access auth.
    :param dict[str, Any] storage_options:
    :param Optional[int] process_task_limit:
        If ``None`` will get everything and process all of them at once.
    :param int num_pool_network:
        maximum number of Pool for network task.
    :param int num_pool_cpu:
        maximum number of Pool for cpu task.
    :param int num_pool_gpu:
        maximum number of Pool for gpu task.
        ``warning`` this should be set to 1 for now, because of
        different hardware model capabilities when using GPU Pool.
    :param int num_pool_upload:
        maximum number of Pool for upload task.
        explicitly different than network, because upload should managed on its
        own because of buffering and IO related.
    :param int max_buffer_size:
        maximum buffer task size, if reached will upload.
    :param float max_buffer_seconds:
        maximum buffer wait seconds, if reached will upload.
    """
    def __init__(
        self,
        endpoint: str,
        worker_token: str,
        worker_id: int,
        storage_options: dict,
        process_task_limit: int | None = 200,

        # Batch worker options
        num_pool_network: int = 5,
        num_pool_cpu: int = 1,
        num_pool_gpu: int = 1,
        num_pool_upload: int = 5,

        # Buffer options
        max_buffer_size: int = 15,
        max_buffer_seconds: float = 15 * 60.0,

        **kwargs
    ):
        super().__init__(endpoint=endpoint)

        # --------
        # Storage
        # --------
        self.storage: StorageMixin
        self.storage_code: str = storage_options.pop('provider', None)
        if self.storage_code == 'gdrive':
            from iantirta.media import DriveStorage
            self.storage = DriveStorage(**storage_options)

        elif self.storage_code == 'hf_dataset':
            from iantirta.media import HFDatasetStorage
            self.storage = HFDatasetStorage(**storage_options)

        else:
            raise TypeError(f"Given provider is not yet supported {self.storage_code}")


        # --------
        # Logging
        # --------
        self.log_entries = []

        self.logger = logging.getLogger(f"Worker({worker_id})")
        self.logger.setLevel(logging.DEBUG)

        if not hasattr(self.logger, 'add_hook'):
            from iantirta.telemetry import netsvc
            netsvc.setup_logger("iantirta")

        def store_log_entries(record: logging.LogRecord) -> None:
            """
            Will store log entries so that we can upload it later
            when this worker finishes.
            """
            log_data = {
                'name': record.name,
                'level': record.levelname,
                'message': record.getMessage(),
                'path': Path(record.pathname).as_posix(),
                'line': record.lineno,
                'func': record.funcName,
            }
            self.log_entries.append(log_data)

        self.logger.add_hook(store_log_entries)

        # --------
        # Pool
        # --------


        # --------
        # Buffer
        # --------

        self.max_buffer_size = max_buffer_size
        self.max_buffer_seconds = max_buffer_seconds

        self.task_buffer: list[KaraokeTask] = []

        self.last_flush_ts = time.time()

    def _flush(self):
        """
        Actually flush. with lock.

        The flush process consist of :
            * Uploading it to storage.
            * Uploading it to server if storage success
        """
        if not self.task_buffer:
            return

        with self.lock:
            if not self.task_buffer:
                return

            self.logger.debug(f"Flushing task buffer data ({len(self.task_buffer)} tasks)...")

            try:
                self.logger.debug(f"Flushing staging dir, path={self.storage.staging_dir.as_posix()}...")
                self.storage.upload_folder(
                    dir_path=self.storage.staging_dir
                )
            except Exception as e:
                

    def flush(self):
        """
        Will flush if there is any buffer task or within condition
        """
        if not self.task_buffer or len(self.task_buffer) < self.max_buffer_size:
            return

        time_elapsed = time.time() - self.last_flush_ts
        if time_elapsed < self.max_buffer_seconds:
            return

        self._flush()

    def run(self, *, once=False, dry_run=False):
        """
        Run batch karaoke forever as long as there is still task to processed.

        :param bool once:
            if ``True`` it would only run once for the ``process_task_limit``
        """
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

                self.flush()

                if once:
                    break
        finally:
            self.worker_stop()