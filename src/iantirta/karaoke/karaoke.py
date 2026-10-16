
import logging
import threading
import gc
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
    Future,
)
from typing import Any

from iantirta.media import install_deno
from iantirta.models.load_model import load_model

from iantirta.karaoke.core.main import KaraokeTask
from iantirta.karaoke.rendering_utils import Renderer

_logger = logging.getLogger(__name__)


_DownloadStage = 'download'
_SeparationStage = 'separate'
_DetectAudioStage = 'detect_audio'
_TranscribeStage = 'transcribe'
_AlignLyricStage = 'align_lyric'
_AlignForceStage = 'align_force'
_RenderStage = 'render'

_RenderHookStage = 'render_hook'
_ErrorHookStage = 'error_hook'


class BatchKaraoke:
    def __init__(
        self,
        tasks: list[dict[str, Any]],
        *,
        render_hook: Callable | None = None,
        error_hook: Callable | None = None,
        logger: logging.Logger | None = None,
        max_download_workers: int = 5,
        max_cpu_workers: int = 1,
        max_gpu_pool: int = 1,
    ):
        if max_download_workers < 1:
            raise ValueError("max_download_workers must be >= 1")

        if max_cpu_workers < 1:
            raise ValueError("max_cpu_workers must be >= 1")

        if max_gpu_pool != 1:
            raise ValueError(
                "This scheduler requires exactly one GPU worker "
                "for exclusive model lifecycle."
            )

        self._karaoke_tasks = [
            KaraokeTask.from_dict(**task)
            for task in tasks
        ]

        install_deno()
        Renderer.ensure_font_exists()

        self.render_hook = render_hook
        self.error_hook = error_hook
        
        self.logger = (logger or _logger)

        self._failed_tasks: set[int] = set()

        self._model_lock = threading.Lock()
        self.separator = None
        self.transcriber = None
        self.aligner = None

        self.max_download_workers = max_download_workers
        self.max_cpu_workers = max_cpu_workers
        self.max_gpu_pool = max_gpu_pool

    @property
    def karaoke_tasks(self) -> list[KaraokeTask]:
        return [
            task
            for task in self._karaoke_tasks
            if id(task) not in self._failed_tasks
        ]

    @property
    def karaoke_plus_tasks(self) -> list[KaraokeTask]:
        return [
            task
            for task in self.karaoke_tasks
            if task.karaoke_type == "plus"
        ]

    def get_separator_model(self):
        if self.separator is not None:
            return self.separator

        with self._model_lock:
            if self.separator is None:
                self.separator = load_model(
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

            return self.separator

    def get_transcribe_model(self):
        # TODO: to be implemented later
        return None

    def get_aligner_model(self):
        # TODO: to be implemented later
        return None

    def _cleanup_models(self) -> None:
        """Release this batch's model references and reclaim available memory."""
        self.separator = None
        self.transcriber = None
        self.aligner = None

        self.clear_memory()

    def clear_memory(self):
        """Collect Python garbage and release unused PyTorch CUDA cache."""
        gc.collect()

        try:
            import torch
        except ImportError:
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def cleanup_tasks_files(self):
        """Delete temporary files."""
        for task in self._karaoke_tasks:
            try:
                for path in [
                    task.karaoke_path,
                    task.instrument_path, task.vocal_path,
                    task.audio_path, task.video_path
                ]:
                    if path is not None:
                        try:
                            path.unlink(missing_ok=True)
                        except OSError:
                            self.logger.warning(
                                "Could not remove temporary file %s for task [%s]",
                                path,
                                task.id,
                                exc_info=True,
                            )
            except:
                self.logger.warning(
                    "Could not remove temporary file %s for task [%s]",
                    path,
                    task.id,
                    exc_info=True,
                )
                continue

    def run(self) -> None:

        self._failed_tasks.clear()

        task_counter: dict[str, int] = {
            _DownloadStage: 0,
            _SeparationStage: 0,
            _DetectAudioStage: 0,
            _TranscribeStage: 0,
            _AlignLyricStage: 0,
            _AlignForceStage: 0,
            _RenderStage: 0,
            _RenderHookStage: 0,
            _ErrorHookStage: 0,
            'failed': 0,
            'success': 0,
        }
        
        tasks = list(self._karaoke_tasks)
        total_task = len(tasks)
    
        # Account for all downloads, including those not submitted yet.
        task_counter[_DownloadStage] = total_task

        with (
            ThreadPoolExecutor(max_workers=self.max_download_workers, thread_name_prefix="DownloadPool") as network_pool,
            ThreadPoolExecutor(max_workers=self.max_gpu_pool, thread_name_prefix="GpuPool") as gpu_pool,
            ThreadPoolExecutor(max_workers=self.max_cpu_workers, thread_name_prefix="CpuPool") as cpu_pool
        ):
            active_task: dict[Future, tuple[str, KaraokeTask]] = {}

            separation_done = False
            transcribe_done = False
    
            # These queues contain tasks already counted in their respective
            # stage counters, but not yet submitted to the GPU executor.
            transcribe_task: list[KaraokeTask] = []
            forcealign_task: list[KaraokeTask] = []


            def handle_error(
                task: KaraokeTask,
                stage: str,
                error: Exception,
            ) -> None:
                """Record a failure without recursively retrying error hooks."""
                self._failed_tasks.add(id(task))
    
                self.logger.error(
                    "Task [%s] failed during stage [%s]: %s",
                    task.id,
                    stage,
                    error,
                    exc_info=(type(error), error, error.__traceback__),
                )
    
                if stage == _ErrorHookStage or self.error_hook is None:
                    task_counter["failed"] += 1
                    return
    
                try:
                    hook_future = self.error_hook(task)
    
                    if not isinstance(hook_future, Future):
                        raise TypeError(
                            "error_hook must return a concurrent.futures.Future"
                        )
    
                    active_task[hook_future] = (_ErrorHookStage, task)
                    task_counter[_ErrorHookStage] += 1
    
                except Exception:
                    self.logger.exception(
                        "Could not schedule error reporting for task [%s]",
                        task.id,
                    )
                    task_counter["failed"] += 1

            def dispatch_transcribe(
                queued_tasks: list[KaraokeTask],
            ) -> None:
                """Dispatch deferred ASR tasks after the separation barrier."""
                if not queued_tasks:
                    return
    
                try:
                    if self.transcriber is None:
                        self.transcriber = self.get_transcribe_model()
    
                    model = self.transcriber
                except Exception as exc:
                    for queued_task in queued_tasks:
                        task_counter[_TranscribeStage] -= 1
                        handle_error(queued_task, _TranscribeStage, exc)
                    return

                for queued_task in queued_tasks:
                    future = gpu_pool.submit(
                        queued_task._transcribe,
                        model,
                    )
                    active_task[future] = (_TranscribeStage, queued_task)


            def dispatch_forcealign(
                queued_tasks: list[KaraokeTask],
            ) -> None:
                """Dispatch forced alignment after ASR and lyric alignment finish."""
                if not queued_tasks:
                    return

                try:
                    if self.aligner is None:
                        self.aligner = self.get_aligner_model()
    
                    model = self.aligner
                except Exception as exc:
                    for queued_task in queued_tasks:
                        task_counter[_AlignForceStage] -= 1
                        handle_error(queued_task, _AlignForceStage, exc)
                    return
    
                for queued_task in queued_tasks:
                    future = gpu_pool.submit(
                        queued_task._align_force,
                        model,
                    )
                    active_task[future] = (_AlignForceStage, queued_task)

            try:
                active_task = {
                    network_pool.submit(
                        task.get_audio_path,
                        lyric=task.karaoke_type == "plus"
                    ) : (_DownloadStage, task)
                    for task in self.karaoke_tasks
                }

                while active_task:
                    done, _ = wait(active_task, return_when=FIRST_COMPLETED)

                    for future in done:
                        stage, task = active_task.pop(future)
                        task_counter[stage] -= 1

                        try:
                            result = future.result()

                            if task._error_entries:
                                raise RuntimeError(
                                    "Task contains recorded errors. "
                                    'See previous log.'
                                )

                            self.logger.info(
                                "Task [%s] stage [%s] completed",
                                task.id,
                                stage,
                            )

                            if stage == _DownloadStage:
                                separation_future = gpu_pool.submit(
                                    task._separate_vocal_and_instrument,
                                    # we get model on the main thread.
                                    self.get_separator_model(),
                                )
                                active_task[separation_future] = (_SeparationStage, task)
                                task_counter[_SeparationStage] += 1

                            elif stage == _SeparationStage:
                                if task.karaoke_type == 'plus':
                                    detect_audio_future = cpu_pool.submit(
                                        task._detect_audio_activity
                                    )
                                    active_task[detect_audio_future] = (_DetectAudioStage, task)
                                    task_counter[_DetectAudioStage] += 1
                                else:
                                    render_future = cpu_pool.submit(
                                        task._render_karaoke
                                    )
                                    active_task[render_future] = (_RenderStage, task)
                                    task_counter[_RenderStage] += 1

                            elif stage == _DetectAudioStage:
                                # we add this here regardless to keep track of the counter.
                                task_counter[_TranscribeStage] += 1
                                
                                if separation_done:
                                    dispatch_transcribe([task])
        
                                else:
                                    transcribe_task.append(task)

                            elif stage == _TranscribeStage:
                                align_lyric_future = cpu_pool.submit(
                                    task._align_lyric
                                )
                                active_task[align_lyric_future] = (_AlignLyricStage, task)
                                task_counter[_AlignLyricStage] += 1

                            elif stage == _AlignLyricStage:
                                # we add this here regardless to keep track of the counter.
                                task_counter[_AlignForceStage] += 1

                                if transcribe_done:
                                    dispatch_forcealign([task])

                                else:
                                    forcealign_task.append(task)

                            elif stage == _AlignForceStage:
                                render_future = cpu_pool.submit(
                                    task._render_karaoke
                                )
                                active_task[render_future] = (_RenderStage, task)
                                task_counter[_RenderStage] += 1

                            elif stage == _RenderStage:
                                if self.render_hook is not None:
                                    future_render_hook = self.render_hook(task)
                                    if not isinstance(future_render_hook, Future):
                                        raise TypeError(
                                            "render_hook must return a "
                                            "concurrent.futures.Future"
                                        )
                                    active_task[future_render_hook] = (_RenderHookStage, task)

                                    task_counter[_RenderHookStage] += 1
                                else:
                                    # Useful for standalone rendering.
                                    # A worker should supply an upload hook.
                                    task_counter['success'] += 1

                            elif stage == _RenderHookStage:
                                if result is False:
                                    raise RuntimeError("Render hook reported failure")

                                task_counter['success'] += 1
                                self.logger.info(
                                    "Task [%s] render_hook completed",
                                    task.id,
                                )

                            elif stage == _ErrorHookStage:
                                if result is False:
                                    # we only wanted to do log output here.
                                    self.logger.error(
                                        "Could not schedule error reporting "
                                        "for task [%s]. See previous error log.",
                                        task.id
                                    )

                                task_counter['failed'] += 1

                        except Exception as e:
                            handle_error(task, stage, e)

                        finally:
                            # Check barriers after each completed future. A barrier
                            # may become ready because of either a stage completion
                            # or the completion of its final upstream task.
                            if (
                                not separation_done
                                and task_counter[_DownloadStage] == 0
                                and task_counter[_SeparationStage] == 0
                            ):
                                separation_done = True
        
                                # No Demucs jobs remain, so release its model.
                                self.separator = None
                                self.clear_memory()
        
                                # Detection may have completed before the final
                                # separation job. Do not leave those tasks stranded.
                                queued = transcribe_task
                                transcribe_task = []
                                dispatch_transcribe(queued)
        
                            if (
                                separation_done
                                and not transcribe_done
                                and task_counter[_DetectAudioStage] == 0
                                and task_counter[_TranscribeStage] == 0
                                and task_counter[_AlignLyricStage] == 0
                            ):
                                transcribe_done = True

                                # All transcription and lyric alignment work has
                                # finished. It is now safe to release the ASR model.
                                self.transcriber = None
                                self.clear_memory()
        
                                queued = forcealign_task
                                forcealign_task = []
                                dispatch_forcealign(queued)
        
                            # Each task must always be represented by exactly one
                            # stage, success, or failure counter.
                            assert sum(task_counter.values()) == total_task
        
                # Empty active_task must also mean no deferred work remains.
                if transcribe_task or forcealign_task:
                    raise RuntimeError(
                        "Scheduler stopped with deferred tasks remaining: "
                        f"{len(transcribe_task)} transcription, "
                        f"{len(forcealign_task)} forced-alignment"
                    )
    
                if sum(task_counter.values()) != total_task:
                    raise RuntimeError(
                        "Task counter invariant violated: "
                        f"{sum(task_counter.values())} != {total_task}"
                    )
    
                unfinished = {
                    stage: count
                    for stage, count in task_counter.items()
                    if stage not in ("success", "failed") and count != 0
                }
                if unfinished:
                    raise RuntimeError(
                        f"Scheduler stopped with unfinished stages: {unfinished}"
                    )
                
            except KeyboardInterrupt:
                self.logger.warning(
                    "CTRL+C Interrupt received; cancelling pending work."
                )

                for future in active_task:
                    future.cancel()
                    self.logger.warning(f"canceling active_futures {future}")

                # Running Python threads cannot be forcibly terminated.
                # Executor shutdown waits for running calls to finish.
                raise
            
            finally:
                self._cleanup_models()

        # This should be called after all the
        # render_hook, be it Future, is done first.
        # because sometime it is using the task file to upload,
        # or needing some of it.
        self.cleanup_tasks_files()

        self.logger.info(
            "Batch finished: %d succeeded, %d failed, %d total.",
            task_counter["success"],
            task_counter["failed"],
            total_task,
        )


if __name__ == "__main__":
    from iantirta.telemetry import netsvc
    netsvc.setup_logger("iantirta", default_level=logging.DEBUG)
    urls = [
        "https://youtube.com/watch?v=sdnDInjdWjw&list=RDsdnDInjdWjw&start_radio=1&pp=ygUIQmVybmFkeWGgBwE%3D",
        "https://youtube.com/watch?v=sdnDInjdWjw&list=RDsdnDInjdWjw&start_radio=1&pp=ygUIQmVybmFkeWGgBwE%3D",
    ] * 2
    tasks = [
        {
            "id": i,
            "source": url,
            "karaoke_type": 'plus'
        } for i, url in enumerate(urls)
    ]
    _logger = logging.getLogger('iantirta.karaoke')
    _logger.info("running")
    BatchKaraoke(tasks).run()