
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path
from typing import Any

import psutil
from iantirta.media import install_deno
from iantirta.models.load_model import load_model

from iantirta.karaoke.core.main import KaraokeTask
from iantirta.karaoke.rendering_utils import Renderer

_logger = logging.getLogger(__name__)


class BatchKaraoke:
    def __init__(
        self,
        tasks: list[dict[str, Any]],
        *,
        render_hook: Callable | None = None,
        error_hook: Callable | None = None,
        logger: logging.Logger | None = None,
        max_gpu_pool: int = 1,
    ):
        self._karaoke_tasks = [
            KaraokeTask.from_dict(**task)
            for task in tasks
        ]

        Renderer.ensure_font_exists()

        self.render_hook = render_hook
        self.error_hook = error_hook
        
        self.logger = (logger or _logger)
        if max_gpu_pool is None:
            max_gpu_pool = 1
        if max_gpu_pool <= 0:
            raise OSError("Gpu pool needs to be above 0")
        self.max_gpu_pool = max_gpu_pool

        self.main_thread_id = threading.current_thread().ident
        self.quit_signals_received = 0

        if os.name == 'posix':
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            signal.signal(signal.SIGCHLD, self.signal_handler)
            signal.signal(signal.SIGHUP, self.signal_handler)
            signal.signal(signal.SIGXCPU, self.signal_handler)
        elif os.name == 'nt':
            import win32api
            win32api.SetConsoleCtrlHandler(lambda sig: self.signal_handler(sig, None), 1)

    def signal_handler(self, sig, frame):
        if sig in [signal.SIGINT, signal.SIGTERM]:
            # shutdown on kill -INT or -TERM
            self.quit_signals_received += 1
            if self.quit_signals_received > 1:
                # logging.shutdown was already called at this point.
                sys.stderr.write("Forced shutdown.\n")
                os._exit(0)
            # interrupt run() to start shutdown
            raise KeyboardInterrupt()

    @property
    def karaoke_tasks(self) -> list[KaraokeTask]:
        return [task for task in self._karaoke_tasks if len(task._error_entries) == 0]

    @property
    def karaoke_plus_tasks(self) -> list[KaraokeTask]:
        return [task for task in self.karaoke_tasks if task.karaoke_type == 'plus']

    def shutdown(self):
        """Programmatically signal the batch process to stop."""
        self.logger.warning("Shutdown signal received. cleaning up thread...")
        stop_time = time.time()

        # Manually join() all threads before calling sys.exit() to allow a second signal
        # to trigger _force_quit() in case some non-daemon threads won't exit cleanly.
        # threading.Thread.join() should not mask signals (at least in python 2.5).
        me = threading.current_thread()
        self.logger.debug('current thread: %r', me)
        for thread in threading.enumerate():
            self.logger.debug('process %r (%r)', thread, thread.daemon)
            if (thread != me and not thread.daemon and thread.ident != self.main_thread_id):
                while thread.is_alive() and (time.time() - stop_time) < 1:
                    # We wait for requests to finish, up to 1 second.
                    self.logger.debug('join and sleep')
                    # Need a busyloop here as thread.join() masks signals
                    # and would prevent the forced shutdown.
                    thread.join(0.05)
                    time.sleep(0.05)

        current_process = psutil.Process()
        children = current_process.children(recursive=False)
        for child in children:
            _logger.info('A child process was found, pid is %s, process may hang', child)

    def clear_memory(self):
        import gc

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        
    def cleanup_tasks_files(self):
        for task in self._karaoke_tasks:
            try:
                for path in [
                    task.karaoke_path,
                    task.instrument_path, task.vocal_path,
                    task.audio_path, task.video_path
                ]:
                    if path is not None:
                        path.unlink(missing_ok=True,)
            except:
                continue

    def run(self) -> None:

        install_deno()

        separator = None
        transcriber = None
        aligner = None

        with (
            ThreadPoolExecutor(max_workers=5, thread_name_prefix="NetPool") as network_pool,
            ThreadPoolExecutor(max_workers=self.max_gpu_pool, thread_name_prefix="GpuPool") as gpu_pool,
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="CpuPool") as cpu_pool
        ):
            try:
                active_futures = {
                    network_pool.submit(task.get_audio_path, lyric=task.karaoke_type=='plus')
                    : ('download', task)
                    for task in self.karaoke_tasks
                }

                while active_futures:
                    done, _ = wait(active_futures.keys(), timeout=0.5, return_when=FIRST_COMPLETED)

                    for future in done:
                        task_type, task = active_futures.pop(future)

                        try:
                            res = future.result()

                            if len(task._error_entries) == 0:
                    
                                if task_type == "download":
                                    self.logger.info(f"Task [{task.id}] Download complete: {res}")

                                    if separator is None:
                                        separator = load_model(
                                            "adefossez/Demucs-mdx_extra_q",
                                            "separation",
                                            model_type_options={
                                                "shifts": 1,
                                                "overlap": 0.75,
                                                "split": True,
                                                "segment": 200,
                                                "jobs" : 0,
                                                "logger" : self.logger,
                                            }
                                        )
                                    sep_future = gpu_pool.submit(task._separate_vocal_and_instrument, separator)
                                    # sep_future = gpu_pool.submit(print, f"{task.id} mock separate")
                                    active_futures[sep_future] = ("separate", task)

                                elif task_type == "separate":
                                    self.logger.info(f"Task [{task.id}] Separation complete: {res}")

                                    if task.karaoke_type == 'basic':
                                        render_future = cpu_pool.submit(task._render_karaoke)
                                        active_futures[render_future] = ("render", task)

                                elif task_type == "render":
                                    assert task.karaoke_type == "basic"
                                    self.logger.info(f"Task [{task.id}] Render succesful.")
                                    if self.render_hook is not None:
                                        self.render_hook(task, network_pool)

                            else:
                                raise OSError("Task have error therefore cannot continue.")

                        except Exception as e:
                            self.logger.error(f"Task [{task.id}] Error during {task_type}: {e}")
                            if self.error_hook is not None:
                                self.error_hook(task, network_pool)

                wait(active_futures.keys(), return_when=ALL_COMPLETED)
                active_futures.clear()

                try:
                    del separator.model, separator
                except:
                    pass
                self.clear_memory()

                if self.karaoke_plus_tasks:
                    
                    active_futures = {
                        cpu_pool.submit(task._detect_audio_activity)
                        : ("detect_audio", task)
                        for task in self.karaoke_plus_tasks
                    }

                    while active_futures:
                        done, _ = wait(active_futures.keys(), timeout=0.5, return_when=FIRST_COMPLETED)
    
                        for future in done:
                            task_type, task = active_futures.pop(future)

                            try:
                                res = future.result()
    
                                if len(task._error_entries) == 0:
                                    if task_type == 'detect_audio':
                                        self.logger.info(f"Task [{task.id}] detect audio complete: {res}")

                                        if not transcriber:
                                            transcriber = None
                                        #transcrbe_future = gpu_pool.submit(task._transcribe, transcriber)
                                        transcribe_future = gpu_pool.submit(print, f"{task.id} mock transcribe")
                                        active_futures[transcribe_future] = ("transcribe", task)

                                    elif task_type == 'transcribe':
                                        self.logger.info(f"Task [{task.id}] transcribe complete: {res}")

                                        lyric_align_future = cpu_pool.submit(task._align_lyric)
                                        active_futures[lyric_align_future] = ('align_lyric', task)

                                    elif task_type == 'align_lyric':
                                        self.logger.info(f"Task [{task.id}] transcribe complete: {res}")

                                else:
                                    raise OSError("Task have error therefore cannot continue.")

                            except Exception as e:
                                self.logger.error(f"Task [{task.id}] Error during {task_type}: {e}")
                                if self.error_hook is not None:
                                    self.error_hook(task, network_pool)
                                    
                    wait(active_futures.keys(), return_when=ALL_COMPLETED)
                    active_futures.clear()

                    try:
                        del transcriber.model, transcriber
                    except:
                        pass
                    self.clear_memory()

                    active_futures = {
                        gpu_pool.submit(task._align_force, aligner)
                        : ('align_force', task)
                        for task in self.karaoke_plus_tasks
                    }

                    while active_futures:
                        done, _ = wait(active_futures.keys(), timeout=0.5, return_when=FIRST_COMPLETED)
    
                        for future in done:
                            task_type, task = active_futures.pop(future)

                            try:
                                res = future.result()
    
                                if len(task._error_entries) == 0:
                                    if task_type == 'align_force':
                                        self.logger.info(f"Task [{task.id}] Separation complete: {res}")
                                        
                                        render_future = cpu_pool.submit(task._render_karaoke)
                                        active_futures[render_future] = ("render", task)

                                    elif task_type == "render":
                                        self.logger.info(f"Task [{task.id}] Render succesful.")
                                        if self.render_hook is not None:
                                            self.render_hook(task, network_pool)

                                else:
                                    raise OSError("Task have error therefore cannot continue.")

                            except Exception as e:
                                self.logger.error(f"Task [{task.id}] Error during {task_type}: {e}")
                                if self.error_hook is not None:
                                    self.error_hook(task, network_pool)

                    wait(active_futures.keys(), return_when=ALL_COMPLETED)
                    active_futures.clear()
    
                    try:
                        del aligner.model, aligner.processor, aligner
                    except:
                        pass
                    self.clear_memory()

            except KeyboardInterrupt:
                self.logger.warning("CTRL+C triggered. shutting down gracefully...")
                try:
                    if separator is not None:
                        del separator.model, separator
                except:
                    pass
                if transcriber is not None:
                    try:
                        del aligner.model, aligner.processor, aligner
                    except:
                        pass
                if aligner is not None:
                    try:
                        del aligner.model, aligner.processor, aligner
                    except:
                        pass
                self.clear_memory()

                for future in active_futures.keys():
                    future.cancel()
                    self.logger.warning(f"canceling active_futures {future}")
                active_futures.clear()

                network_pool.shutdown(wait=False, cancel_futures=True)
                gpu_pool.shutdown(wait=False, cancel_futures=True)
                cpu_pool.shutdown(wait=False, cancel_futures=True)
                
                self.shutdown()
                raise

            finally:
                network_pool.shutdown()
                gpu_pool.shutdown()
                cpu_pool.shutdown()

        self.logger.info("All task completed.")
        self.cleanup_tasks_files()


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