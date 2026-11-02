
import logging
from pathlib import Path
import time
import shutil
from typing import TYPE_CHECKING, Any
from concurrent.futures import Future, ThreadPoolExecutor
import threading
import uuid

from iantirta.media import IanTirtaAPI

if TYPE_CHECKING:
    from iantirta.media.cloud_storage.base_storage import StorageMixin

    from iantirta.karaoke.core.main import KaraokeTask


# Representing notset value instead of using None
_Sentinel = object()


class KaraokeWorker(IanTirtaAPI):
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

    endpoint: str = 'https://karaoke.iantirta.com'

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
        super().__init__(
            endpoint=endpoint,
            logger=logging.getLogger(f"Worker({worker_id})"),
        )

        if num_pool_upload < 1:
            raise ValueError(
                "``num_pool_network`` must be >= 1"
            )

        self.worker_id = worker_id
        self.worker_token = worker_token

        self.process_task_limit = process_task_limit
        self.num_pool_network = num_pool_network
        self.num_pool_cpu = num_pool_cpu
        self.num_pool_gpu = num_pool_gpu

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
            raise TypeError(
                f"Given provider is not yet supported {self.storage_code}"
            )

        # --------
        # Logging
        # --------

        self.log_entries = []

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

        self.lock = threading.Lock()

        # --------
        # Pool
        # --------

        self.upload_pool = ThreadPoolExecutor(num_pool_upload)

        # --------
        # Buffer
        # --------

        self.max_buffer_size = max_buffer_size
        self.max_buffer_seconds = max_buffer_seconds

        self.task_buffer: list[KaraokeTask] = []

        self.last_flush_ts = time.monotonic()

    # --------
    # Helper api communication method
    # --------

    def make_json_request(
        self,
        path: str = '',
        params: dict | None = None,
    ):
        """
        Override the default IanTirtaAPI meth:``make_json_request``

        This will fill up necessary parameters needed for worker endpoint.

        And since original have method args we append it here also.

        :param str path:
        :param Optional[dict[str, Any]] params:
        :return: Any the server gives, can be dict, list, None, etc.
        :raise IanTirtaAPIError: if server returns an error.
        """
        payload = {
            "worker_id": self.worker_id,
            "access_token": self.worker_token,
        }
        payload.update(params or {})

        res = super().make_json_request(
            'GET', path=path, params={"params": payload},
        )

        return res['result']

    # --------
    # Api communication method
    # --------

    def worker_claim_task(self, tasks: list[dict[str, Any]]):
        """
        Method that makes api call to the server.
        Indicating that the tasks will be processed by this worker.

        This function exists because of how different
        api call to ``karaoke/tasks/get`` can be processed.
        it would not always be processed by the same worker instance
        or any other cases.

        :param list[dict[str, Any]] tasks:
        """
        return self.make_json_request(
            '/karaoke/worker/claim_task',
            {'task_ids': [
                task['id']
                for task in tasks
            ]},
        )

    def worker_start(self):
        """
        Method that makes api call to the server.
        Indicating that this worker has just begun starting.
        """
        return self.make_json_request(
            "/karaoke/worker/start",
        )

    def worker_stop(self):
        """
        Method that makes api call to the server.
        Indicating that this worker have just finished the run.

        This would also post this worker log entries to the server.
        """
        return self.make_json_request(
            "/karaoke/worker/stop",
            {'log_entries': self.log_entries},
        )

    def claim_tasks(
        self,
        limit: int | None = _Sentinel
    ) -> list[dict[str, Any]]:
        """
        Method that makes api call to the server
        for getting task that needed to be processed.

        :param Optional[int] limit:
            if ``None``, would get everything all at once.
            if ``notset``, would use this instance limit.

        :return: list of dictionary represanting karaoke tasks from the server.
        :rtype: list[dict[str, Any]]
        """
        if limit is _Sentinel:
            limit = self.process_task_limit

        return self.make_json_request(
            'karaoke/tasks/get',
            {'limit': limit},
        )

    def post_tasks(self, tasks: list[KaraokeTask]) -> None:
        """
        Method that makes api call to the server endpoint
        for updating karaoke task data.

        :param list[KaraokeTask] tasks:
        """
        if not tasks:
            return

        self.logger.debug(
            f"Flushing buffer data to server, length={len(tasks)}..."
        )

        return self.make_json_request(
            'karaoke/task/update',
            {'tasks': [
                task.to_dict()
                for task in tasks
            ]},
        )

    # --------
    # Flushing method
    # --------

    def _flush(
        self,
        staging_dir: Path,
        tasks: list[KaraokeTask],
    ):
        """
        Actually flush. 

        This process is outside a lock.
        so uploading the actual data should not used any
        shared object.

        The flush process consist of :
            * Uploading it to storage.
            * Uploading it to server if storage success

        :param Path staging_dir:
        :param list[KaraokeTask] tasks:
        """
        if not tasks:
            return

        self.logger.debug(
            f"Flushing task buffer data ({len(tasks)} tasks)..."
        )

        try:
            self.logger.debug(
                "Flushing staging dir, path="
                f"{staging_dir.as_posix()}..."
            )

            self.storage.upload_folder(
                dir_path=staging_dir
            )

        except Exception as e:
            self.logger.error(f"Critical error while flushing storage: {e}")
            # Mark every task in this bath as error.
            for task in tasks:
                task.download_uris = False
                task.error('storage.upload_folder', e)

        try:
            # Any error storage upload error
            # should also be post to the backed.
            self.post_tasks(tasks)

        except Exception as e:
            # This shouldnt block main function.
            # and can make necessary cleanup.
            self.logger.error(
                f'Error while posting task to server. Error: {e}'
            )

        shutil.rmtree(staging_dir, ignore_errors=True)

    def flush(self, *, force: bool = False):
        """
        Will flush if there is any buffer task or within condition.

        :param bool force:
        """

        if not self.task_buffer:
            return

        batch_staging_dir = None
        batch_tasks = None

        with self.lock:
            if not self.task_buffer:
                return

            time_elapsed = time.monotonic() - self.last_flush_ts

            if (
                force
                or len(self.task_buffer) >= self.max_buffer_size
                or time_elapsed >= self.max_buffer_seconds
            ):
                # Rotate staging dir here in lock.
                original_staging_dir: str = self.storage.staging_dir.as_posix()

                batch_id = uuid.uuid4().hex  # Not shorting the hex.
                batch_staging_dir = (
                    self.storage.staging_dir.with_name(
                        self.storage.staging_dir.name + str(batch_id)
                    )
                )

                # Detach and rename
                self.storage.staging_dir.rename(batch_staging_dir)

                # Remake the staging dir
                self.storage.staging_dir = Path(original_staging_dir)
                self.storage.staging_dir.mkdir(parents=True, exist_ok=True)

                # Rotate tasks
                batch_tasks = self.task_buffer
                self.task_buffer = []  # Swap instead if clear

                self.last_flush_ts = time.monotonic()

        if not batch_tasks or not batch_staging_dir:
            return

        # Flush outside lock.
        self._flush(batch_staging_dir, batch_tasks)

    # --------
    # Task Hook
    # --------

    def _on_task_render(self, task: KaraokeTask) -> None:
        """
        Actually process any condition when task is successfully rendered.
        this will call flush at the end.

        :param KaraokeTask task:
        """
        try:

            with self.lock:
                chunk_metadata: dict = self.storage.make_chunk_file(
                    src_path=task.karaoke_path,
                    target_path=task.artist,
                )
                chunk_metadata['provider'] = self.storage_code
                task.download_uris = chunk_metadata

                self.task_buffer.append(task)

        except Exception as e:
            self.logger.warning(
                "Exception when trying to upload and post "
                f"Task ({task.id}) Error: {e}"
            )
            # Trigger task error
            task.error('_on_task_render', e)

            # immediately called ``_on_task_error``
            self._on_task_error(task)
            return

        self.flush()

    def on_task_render(self, task: KaraokeTask) -> Future:
        """
        Triggered when any task successfully finished
        this should be at after rendering.

        This will only returns a Future immediately.

        :param KaraokeTask task:
        :return: Future object
        :rtype: Future
        """
        return self.upload_pool.submit(self._on_task_render, task)

    def _on_task_error(self, task: KaraokeTask) -> None:
        """
        Actually process any condition when task is error.
        this will call flush at the end.

        :param KaraokeTask task:
        """
        with self.lock:
            self.task_buffer.append(task)

        self.flush()

    def on_task_error(
        self,
        task: KaraokeTask
    ) -> Future:
        """
        Triggered when any task occur on error.

        This will only returns a Future immediately.

        :param KaraokeTask task:
        :return: Future object
        :rtype: Future
        """
        return self.upload_pool.submit(self._on_task_error, task)

    def run(self, *, once=False, dry_run=False):
        """
        Run batch karaoke forever as long as there is still task to processed.

        :param bool once:
            if ``True`` it would only run once for the ``process_task_limit``
        """
        try:
            self.worker_start()

            while True:
                if not (tasks := self.claim_tasks()):
                    self.logger.info("No more tasks to process")
                    break

                self.logger.info(f"Claimed {len(tasks)} task(s).")

                if dry_run:
                    break

                self.worker_claim_task(tasks)

                try:
                    processor = BatchKaraoke(
                        tasks,
                        on_render_hook=self.on_task_render,
                        on_error_hook=self.on_task_error,
                        logger=self.logger,
                        max_gpu_pool=self.num_pool_gpu,
                    )
                    processor.run()
                except Exception as e:
                    self.logger.error(
                        f"Error while doing batch karaoke process: {e}"
                    )

                self.flush(force=True)

                if once:
                    break
        finally:
            self.upload_pool.shutdown(wait=True, cancel_futures=False)
            self.flush(force=True)
            self.worker_stop()
