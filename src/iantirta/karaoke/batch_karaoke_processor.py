# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

import gc
import logging
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from typing import Any, ClassVar, TypeAlias

from iantirta.media import install_deno
from iantirta.models.load_model import load_model

from iantirta.karaoke.karaoke_task import KaraokeTask
from iantirta.karaoke.rendering_utils import Renderer

__all__ = [
    "BatchKaraoke",
    'KaraokeProcessError',
]


ActiveTaskType: TypeAlias = dict[Future, tuple[str, KaraokeTask]]

# --------------------
# Stage Representation
# --------------------

WAITING_STAGE = 'waiting'
DOWNLOAD_STAGE = 'download'
SEPARATION_STAGE = 'separate'
AAD_STAGE = 'detect_audio'
ASR_STAGE = 'transcribe'
LYRIC_STAGE = 'align_lyric'
FA_STAGE = 'align_force'
RENDER_STAGE = 'render'

RENDER_HOOK_STAGE = 'render_hook'
ERROR_HOOK_STAGE = 'error_hook'


# --------------------
# Exception
# --------------------


class KaraokeProcessError(Exception):
    """ Raised when karaoke process error. """


# --------------------
# Batch Karaoke Processor
# --------------------


class BatchKaraoke:
    """
    Batch Karaoke Processor/Manager.

    Will process karaoke tasks in an optimized
    threading manner,
    while this class manage its behaviour and result.
    """

    karaoke_task_class: ClassVar[type[KaraokeTask]] = KaraokeTask

    def __init__(
        self,
        tasks: list[dict[str, Any]],
        *,
        on_render_hook: Callable[[KaraokeTask], Future] | None = None,
        on_error_hook: Callable[[KaraokeTask], Future] | None = None,
        logger: logging.Logger | None = None,
        num_pool_network: int = 5,
        num_pool_cpu: int = 1,
        num_pool_gpu: int = 1,
    ):
        if num_pool_network < 1:
            raise ValueError(
                "``num_pool_network`` must be >= 1"
            )

        if num_pool_cpu < 1:
            raise ValueError(
                "``num_pool_cpu`` must be >= 1"
            )

        if num_pool_gpu != 1:
            raise ValueError(
                "This scheduler requires exactly one GPU worker "
                "for exclusive model lifecycle."
            )

        # Ensure this exists during runtime
        # to help smoothen karaoke process
        install_deno()
        Renderer.ensure_font_exists()

        # Logging
        self.logger = (logger or logging.getLogger(__name__))

        self.num_pool_network = num_pool_network
        self.num_pool_cpu = num_pool_cpu
        self.num_pool_gpu = num_pool_gpu

        # Karaoke Task Management
        self.total_task = len(tasks)
        self.stage_manager = {
            WAITING_STAGE: self.total_task,
            DOWNLOAD_STAGE: 0,
            SEPARATION_STAGE: 0,
            AAD_STAGE: 0,
            ASR_STAGE: 0,
            LYRIC_STAGE: 0,
            FA_STAGE: 0,
            RENDER_STAGE: 0,
            RENDER_HOOK_STAGE: 0,
            ERROR_HOOK_STAGE: 0,
            'failed': 0,
            'success': 0,
        }

        # Model Management
        self.model_lock = threading.Lock()

        self._separator_model = None
        self._asr_model = None
        self._fa_model = None
        
        self.is_separation_done: bool = False
        self.is_asr_done: bool = False

        self.asr_task: list[KaraokeTask] = []
        self.fa_task: list[KaraokeTask] = []

        self._karaoke_tasks = [
            self.karaoke_task_class.from_dict(**task)
            for task in tasks
        ]
        self._failed_tasks: set[int] = set()
        self._success_tasks: set[int] = set()

        # Karaoke Process Handler
        self.on_error_hook = on_error_hook
        self.on_render_hook = on_render_hook

    @property
    def karaoke_tasks(self) -> list[KaraokeTask]:
        return [
            task
            for task in self._karaoke_tasks
            if id(task) not in self._failed_tasks
        ]

    # ------------------
    # Helper Model Method/properties
    # ------------------

    @property
    def asr_model(self):
        if self._asr_model is None:
            with self.model_lock:
                if self._asr_model is None:
                    self._asr_model = load_model(
                        'Qwen/Qwen3-ASR-1.7B-hf',
                        'asr',
                        model_type_options={
                            'max_inference_batch_size': 1,
                            'num_beams': 4,
                            'logger': self.logger
                        }
                    )
        return self._asr_model

    @property
    def fa_model(self):
        if self._fa_model is None:
            with self.model_lock:
                if self._fa_model is None:
                    self._fa_model = load_model(
                        'facebook/mms-1b-all',
                        'ctc_align',
                        model_type_options={
                            'logger': self.logger
                        }
                    )
        return self._fa_model

    @property
    def separator_model(self):
        if self._separator_model is None:
            with self.model_lock:
                if self._separator_model is None:
                    self._separator_model = load_model(
                        'adefossez/Demucs-mdx_extra_q',
                        "separation",
                        model_type_options={
                            "shifts": 1,
                            "overlap": 0.75,
                            "split": True,
                            "segment": 200,
                            "jobs": 0,
                            "logger": self.logger,
                        },
                    )
        return self._separator_model

    def cleanup_models(self):
        """
        Method that will be called to cleanup models class and instance.
        that is used by this batch processor
        """
        self._separator_model = None
        self._asr_model = None
        self._fa_model = None

        self.clear_memory()

    def clear_memory(self):
        """
        Collect Python garbage and release unused PyTorch CUDA cache.
        """
        gc.collect()

        try:
            import torch
        except ImportError:
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ------------------
    # Karaoke Model Required Dispatcher
    # ------------------

    def dispatch_asr(
        self,
        tasks: list[KaraokeTask],
        gpu_pool: ThreadPoolExecutor,
        active_task: ActiveTaskType
    ) -> ActiveTaskType:
        """
        Method used when dispatching asr stage to the
        active task.
        """
        if not tasks:
            return active_task

        for task in tasks:
            active_task[
                gpu_pool.submit(
                    task.transcribe,
                    self.asr_model
                )
            ] = (ASR_STAGE, task)

        return active_task

    def dispatch_fa(
        self,
        tasks: list[KaraokeTask],
        gpu_pool: ThreadPoolExecutor,
        active_task: ActiveTaskType
    ) -> ActiveTaskType:
        """
        Method used when dispatching fa stage to the
        active task.
        """
        if not tasks:
            return active_task

        for task in tasks:
            active_task[
                gpu_pool.submit(
                    task.force_align,
                    self.fa_model
                )
            ] = (FA_STAGE, task)

        return active_task

    # ------------------
    # Karaoke Process Handler
    # ------------------

    def handle_task_error(
        self,
        task: KaraokeTask,
        stage: str,
        e: Exception,
        active_task: ActiveTaskType,
    ) -> ActiveTaskType:
        """
        Will record task failure.

        As for now it only support single task addition.
        we need more way to optimize and support that error,
        can come in batch.
        but it also need to have a support from the worker hook too.

        :param KaraokeTask task:
        :param str stage:
        :param Exception e:
        :param ActiveTaskType active_task:
        :return: the shared active task dictionary
        :rtype: ActiveTaskType
        """
        self._failed_tasks.add(id(task))

        if not task.error_entries:
            task.error(stage, e)

        # This will only duplicate error log
        # Cause by future not raising error.
        if not isinstance(e, KaraokeProcessError):
            self.logger.exception(
                "Task [%s] failed during stage [%s]: %s",
                task.id, stage, e,
            )

        # Return early if not needed.
        if (
            stage == ERROR_HOOK_STAGE
            or self.on_error_hook is None
        ):
            self.stage_manager['failed'] += 1
            return active_task

        active_task[
            self.on_error_hook(task)
        ] = (ERROR_HOOK_STAGE, task)
        self.stage_manager[ERROR_HOOK_STAGE] += 1

        return active_task

    def handle_post_error_hook(self) -> None:
        """
        Method that will run after error hook is called
        and successful.
        """
        self.stage_manager['failed'] += 1

    def handle_post_download(
        self,
        task: KaraokeTask,
        gpu_pool: ThreadPoolExecutor,
        active_task: ActiveTaskType,
    ) -> ActiveTaskType:
        """
        Method that will run after task is downloaded.

        Will submit separation stage, and also submit
        the next download stage if given.

        :param KaraokeTask task:
        :param ThreadPoolExecutor gpu_pool:
        :param ActiveTaskType active_task:
        :param Optional[KaraokeTask] next_task:
        :param Optional[ThreadPoolExecutor] network_pool:
        :return: the shared active task dictionary
        :rtype: ActiveTaskType
        """
        active_task[
            gpu_pool.submit(
                task.separate_vocal_and_instrument,
                self.separator_model
            )
        ] = (SEPARATION_STAGE, task)
        self.stage_manager[SEPARATION_STAGE] += 1

        return active_task

    def handle_post_separation(
        self,
        task: KaraokeTask,
        cpu_pool: ThreadPoolExecutor,
        active_task: ActiveTaskType
    ) -> ActiveTaskType:
        """
        Method that will run after task is separated
        into its vocal and instrument.

        Will route the next stage based on the ``karaoke_type``
        of the task.

        :param KaraokeTask task:
        :param ThreadPoolExecutor cpu_pool:
        :param ActiveTaskType active_task:
        :return: the shared active task dictionary
        :rtype: ActiveTaskType
        """
        if task.karaoke_type == 'plus':
            active_task[
                cpu_pool.submit(
                    task.detect_audio_activity
                )
            ] = (AAD_STAGE, task)
            self.stage_manager[AAD_STAGE] += 1

        else:
            active_task[
                cpu_pool.submit(
                    task.render_karaoke
                )
            ] = (RENDER_STAGE, task)
            self.stage_manager[RENDER_STAGE] += 1

        return active_task

    def handle_post_render(
        self,
        task: KaraokeTask,
        active_task: ActiveTaskType
    ) -> ActiveTaskType:
        """
        Method that will run after task is rendered.

        Will call ``on_render_hook`` if exist.

        :param KaraokeTask task:
        :param ActiveTaskType active_task:
        :return: the shared active task dictionary
        :rtype: ActiveTaskType
        """
        # Early return
        if self.on_render_hook is None:
            self._success_tasks.add(id(task))
            self.stage_manager['success'] += 1
            task.unlink_files()
            return active_task

        active_task[
            self.on_render_hook(task)
        ] = (RENDER_HOOK_STAGE, task)
        self.stage_manager[RENDER_HOOK_STAGE] += 1

        return active_task

    def handle_post_render_hook(self, task: KaraokeTask) -> None:
        """
        Method that will run after render hook is called
        and successfull.
        """
        self._success_tasks.add(id(task))
        self.stage_manager['success'] += 1
        task.unlink_files()

    def handle_post_aad(
        self,
        task: KaraokeTask,
        gpu_pool: ThreadPoolExecutor,
        active_task: ActiveTaskType
    ) -> ActiveTaskType:
        """
        Method that will run after audio detection is called.

        This method will dispatch asr stage only after separation
        stage is fully completed. Because each gpu model can hold a
        large amount of hardware resource, and this process
        already is trying to maximize the hardware capabilites.

        :param KaraokeTask task:
        :param ThreadPoolExecutor gpu_pool:
        :param ActiveTaskType active_task:
        :return: the shared active task dictionary
        :rtype: ActiveTaskType
        """
        if self.is_separation_done:
            active_task = self.dispatch_asr(
                [task],
                gpu_pool=gpu_pool,
                active_task=active_task,
            )
        else:
            self.asr_task.append(task)
        self.stage_manager[ASR_STAGE] += 1

        return active_task

    def handle_post_asr(
        self,
        task: KaraokeTask,
        cpu_pool: ThreadPoolExecutor,
        active_task: ActiveTaskType
    ) -> ActiveTaskType:
        """
        Method that will run after asr is called.

        :param KaraokeTask task:
        :param ThreadPoolExecutor cpu_pool:
        :param ActiveTaskType active_task:
        :return: the shared active task dictionary
        :rtype: ActiveTaskType
        """
        active_task[
            cpu_pool.submit(
                task.align_lyric
            )
        ] = (LYRIC_STAGE, task)
        self.stage_manager[LYRIC_STAGE] += 1

        return active_task

    def handle_post_lyric(
        self,
        task: KaraokeTask,
        gpu_pool: ThreadPoolExecutor,
        active_task: ActiveTaskType,
    ) -> ActiveTaskType:
        """
        Method that will run after lyric matching/alignment is called.

        This method will dispatch fa stage only after asr
        stage is fully completed. Because each gpu model can hold a
        large amount of hardware resource, and this process
        already is trying to maximize the hardware capabilites.

        :param KaraokeTask task:
        :param ThreadPoolExecutor gpu_pool:
        :param ActiveTaskType active_task:
        :return: the shared active task dictionary
        :rtype: ActiveTaskType
        """
        if self.is_asr_done:
            active_task = self.dispatch_fa(
                [task],
                gpu_pool=gpu_pool,
                active_task=active_task
            )
        else:
            self.fa_task.append(task)
        self.stage_manager[FA_STAGE] += 1

        return active_task

    def handle_post_fa(
        self,
        task: KaraokeTask,
        cpu_pool: ThreadPoolExecutor,
        active_task: ActiveTaskType
    ) -> ActiveTaskType:
        """
        Method that will run after force alignment is called.

        This method will dispatch render job.

        :param KaraokeTask task:
        :param ThreadPoolExecutor cpu_pool:
        :param ActiveTaskType active_task:
        :return: the shared active task dictionary
        :rtype: ActiveTaskType
        """
        active_task[
            cpu_pool.submit(
                task.render_karaoke
            )
        ] = (RENDER_STAGE, task)
        self.stage_manager[RENDER_STAGE] += 1

        return active_task

    # ------------------
    # Main Processor method
    # ------------------

    def stage_router(
        self,
        task: KaraokeTask,
        stage: str,
        active_task: ActiveTaskType,
        gpu_pool: ThreadPoolExecutor,
        cpu_pool: ThreadPoolExecutor,
    ) -> ActiveTaskType:
        """
        Method that will responsible for routing each task stage.

        :param KaraokeTask task:
        :param str stage:
        :param ActiveTaskType active_task:
        :param ThreadPoolExecutor gpu_pool:
        :param ThreadPoolExecutor cpu_pool:
        :param ThreadPoolExecutor network_pool:
        :return: the shared active task dictionary
        :rtype: ActiveTaskType
        """
        # Running this in order so that
        # Python doesnt make unnecessary check.

        if stage == ERROR_HOOK_STAGE:
            self.handle_post_error_hook()

        elif stage == DOWNLOAD_STAGE:
            active_task = self.handle_post_download(
                task=task,
                gpu_pool=gpu_pool,
                active_task=active_task,
            )

        elif stage == SEPARATION_STAGE:
            active_task = self.handle_post_separation(
                task=task,
                cpu_pool=cpu_pool,
                active_task=active_task
            )

        elif stage == RENDER_STAGE:
            active_task = self.handle_post_render(
                task=task,
                active_task=active_task
            )

        elif stage == RENDER_HOOK_STAGE:
            self.handle_post_render_hook(task)

        elif stage == AAD_STAGE:
            active_task = self.handle_post_aad(
                task=task,
                gpu_pool=gpu_pool,
                active_task=active_task
            )

        elif stage == ASR_STAGE:
            active_task = self.handle_post_asr(
                task=task,
                cpu_pool=cpu_pool,
                active_task=active_task,
            )

        elif stage == LYRIC_STAGE:
            active_task = self.handle_post_lyric(
                task=task,
                gpu_pool=gpu_pool,
                active_task=active_task
            )

        elif stage == FA_STAGE:
            active_task = self.handle_post_fa(
                task=task,
                cpu_pool=cpu_pool,
                active_task=active_task,
            )

        return active_task

    def handle_next_download(
        self,
        download_tasks: deque[KaraokeTask],
        network_pool: ThreadPoolExecutor,
        active_task: ActiveTaskType
    ) -> ActiveTaskType:
        """
        Method that will handle next download posting to the pool.

        This is outside any error checking of task, so this will run
        even if the task returns error, while also checking if the
        ``download_tasks`` still have value left.

        :param ActiveTaskType active_task:
        :param deque[KaraokeTask] download_tasks:
        :param ThreadPoolExecutor network_pool:
        :return: the shared active task dictionary
        :rtype: ActiveTaskType
        """
        if not download_tasks:
            return active_task

        next_task = download_tasks.popleft()

        active_task[
            network_pool.submit(
                next_task.get_audio_path,
                lyric=next_task.karaoke_type == "plus"
            )
        ] = (DOWNLOAD_STAGE, next_task)
        self.stage_manager[WAITING_STAGE] -= 1
        self.stage_manager[DOWNLOAD_STAGE] += 1

        return active_task

    def _handle_completed_future(
        self,
        future: Future,
        active_task: ActiveTaskType,
        download_tasks: deque[KaraokeTask],
        gpu_pool: ThreadPoolExecutor,
        cpu_pool: ThreadPoolExecutor,
        network_pool: ThreadPoolExecutor,
    ) -> ActiveTaskType:
        """
        Private Helper method that will process
        completed future

        This method will :
            * Handle task error
            * Route or dispatch to the next stage

        :param Future future:
        :param ActiveTaskType active_task:
        :param deque[KaraokeTask] download_tasks:
        :param ThreadPoolExecutor gpu_pool:
        :param ThreadPoolExecutor cpu_pool:
        :param ThreadPoolExecutor network_pool:
        :return: the shared active task dictionary
        :rtype: ActiveTaskType
        """
        stage, task = active_task.pop(future)
        self.stage_manager[stage] -= 1

        if stage == DOWNLOAD_STAGE and download_tasks:
            active_task = self.handle_next_download(
                download_tasks=download_tasks,
                network_pool=network_pool,
                active_task=active_task
            )

        try:
            future.result()

            if task.error_entries:
                # Raised here so that is run on exception.
                # sometimes future result doesnt
                # raise an error.
                raise KaraokeProcessError(
                    "Task contains recorded errors. "
                    'See previous log.'
                )

            self.logger.info(
                "Task [%s] stage [%s] completed",
                task.id,
                stage,
            )

            active_task = self.stage_router(
                task=task,
                stage=stage,
                active_task=active_task,
                gpu_pool=gpu_pool,
                cpu_pool=cpu_pool,
            )

        except Exception as e:
            active_task = self.handle_task_error(
                task=task,
                stage=stage,
                e=e,
                active_task=active_task
            )

        finally:
            # Separation marked down when :
            #    * ``self.is_separation_done`` is False
            #    * ``DOWNLOAD_STAGE`` == 0
            #    * ``SEPARATION_STAGE`` == 0
            if (
                not self.is_separation_done
                and self.stage_manager[WAITING_STAGE] == 0
                and self.stage_manager[DOWNLOAD_STAGE] == 0
                and self.stage_manager[SEPARATION_STAGE] == 0
                and not download_tasks
            ):
                self.is_separation_done = True
                self._separator_model = None
                self.clear_memory()

                asr_batch = self.asr_task
                self.asr_task = []
                active_task = self.dispatch_asr(
                    asr_batch,
                    gpu_pool=gpu_pool,
                    active_task=active_task
                )

            # ASR marked down when :
            #    * ``self.is_separation_done`` is True
            #    * ``self.is_asr_done`` is False
            #    * ``AAD_STAGE`` == 0
            #    * ``ASR_STAGE`` == 0
            if (
                self.is_separation_done
                and not self.is_asr_done
                and self.stage_manager[AAD_STAGE] == 0
                and self.stage_manager[ASR_STAGE] == 0
            ):
                self.is_asr_done = True
                self._asr_model = None
                self.clear_memory()

                fa_batch = self.fa_task
                self.fa_task = []
                active_task = self.dispatch_fa(
                    fa_batch,
                    gpu_pool=gpu_pool,
                    active_task=active_task
                )

            # Each task must always be represented by exactly one
            # stage, success, or failure counter.
            assert self.stage_manager[WAITING_STAGE] == len(download_tasks)
            assert sum(self.stage_manager.values()) == self.total_task

        return active_task

    def run(self):
        """
        Run batch process for every given task.

        The process it self will use multiple
        ``ThreadPoolExecutor`` so make sure the one
        running this batch processor should take into
        account any concurrency bug that would happened.
        """
        try:
            with (
                ThreadPoolExecutor(
                    max_workers=self.num_pool_network,
                    thread_name_prefix="DownloadPool"
                ) as network_pool,
                ThreadPoolExecutor(
                    max_workers=self.num_pool_gpu,
                    thread_name_prefix="GpuPool"
                ) as gpu_pool,
                ThreadPoolExecutor(
                    max_workers=self.num_pool_cpu,
                    thread_name_prefix="CpuPool"
                ) as cpu_pool
            ):
                active_task: ActiveTaskType = {}
                download_tasks = deque(self.karaoke_tasks)
                try:
                    for _ in range(min(self.num_pool_network, len(download_tasks))):
                        active_task = self.handle_next_download(
                            download_tasks=download_tasks,
                            network_pool=network_pool,
                            active_task=active_task
                        )

                    while active_task:
                        done, _ = wait(active_task, return_when=FIRST_COMPLETED)

                        for future in done:
                            active_task = self._handle_completed_future(
                                future=future,
                                active_task=active_task,
                                download_tasks=download_tasks,
                                gpu_pool=gpu_pool,
                                cpu_pool=cpu_pool,
                                network_pool=network_pool,
                            )

                    # Do validation check once more

                    # This means that either something is still adding into this.
                    # Or it failed to run.
                    if self.asr_task or self.fa_task:
                        raise RuntimeError(
                            "Scheduler stopped with deferred tasks remaining: "
                            f"{len(self.asr_task)} transcription, "
                            f"{len(self.fa_task)} forced-alignment"
                        )

                    if sum(self.stage_manager.values()) != self.total_task:
                        raise RuntimeError(
                            "Task counter invariant violated: "
                            f"{sum(self.stage_manager.values())} != {self.total_task}"
                        )

                    unfinished = {
                        stage: count
                        for stage, count in self.stage_manager.items()
                        if stage not in ("success", "failed") and count != 0
                    }
                    if unfinished:
                        raise RuntimeError(
                            f"Scheduler stopped with unfinished stages: {unfinished}"
                        )

                except KeyboardInterrupt:
                    self.logger.warning(
                        "CTRL+C Interrupt received. "
                        "cancelling pending work."
                    )

                    for future in active_task:
                        future.cancel()
                        self.logger.warning(f"canceling active futures {future}")

                    # Running Python threads cannot be forcibly terminated.
                    # Executor shutdown waits for running calls to finish.
                    raise

                except Exception as e:
                    # Handle error on a batch karaoke worker level failure.
                    # doing it here will not cancel and will wait for any
                    # future left inside the pool.
                    for task in self.karaoke_tasks:
                        if id(task) in self._success_tasks:
                            continue

                        self.handle_task_error(
                            task=task,
                            stage='WorkerStage',
                            e=e,
                            active_task=active_task
                        )

                    raise

        finally:
            self.cleanup_models()

        self.logger.info(
            "Batch finished: %d succeeded, %d failed, %d total.",
            self.stage_manager["success"],
            self.stage_manager["failed"],
            self.total_task,
        )
