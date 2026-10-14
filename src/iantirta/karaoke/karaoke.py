
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
        logger: logging.Logger | None = None
    ):
        self._karaoke_tasks = [
            KaraokeTask.from_dict(**task)
            for task in tasks
        ]

        Renderer.ensure_font_exists()

        self.render_hook = render_hook
        self.error_hook = error_hook
        
        self.logger = (logger or _logger)

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


    def run(self) -> None:
        
        install_deno()

        separator = load_model("adefossez/Demucs-mdx_extra_q", "separation")

        dl_pool = ThreadPoolExecutor(max_workers=5, thread_name_prefix="DLPool")
        sep_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="SepPool")
        render_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="RenderPool")
        
        active_futures = {}
    
        for task in self.karaoke_tasks:
            future = dl_pool.submit(task.get_audio_path)
            active_futures[future] = ("download", task)

        try:
            while active_futures:
                done, not_done = wait(active_futures.keys(), timeout=0.5, return_when=FIRST_COMPLETED)
                
                for future in done:
                    task_type, task = active_futures.pop(future)
                    
                    try:
                        res = future.result()
                        
                        if task_type == "download":
                            self.logger.info(f"[{task.id}] Download complete: {res}")
                            
                            if len(task._error_entries) == 0:
                                sep_future = sep_pool.submit(task._separate_vocal_and_instrument, separator)
                                active_futures[sep_future] = ("separate", task)
                                
                        elif task_type == "separate":
                            self.logger.info(f"[{task.id}] Separation complete: {res}")
                            
                            if len(task._error_entries) == 0 and getattr(task, "karaoke_type", "") == "basic":
                                # RENDER HERE! 
                                self.logger.info(f"[{task.id}] Rendering now...")
                                render_future = render_pool.submit(task._render_karaoke)
                                active_futures[render_future] = ("render", task)
                            else:
                                # Handle other types
                                pass

                        elif task_type == "render":
                            # Post result
                            pass
                                
                    except Exception as e:
                        self.logger.error(f"[{task.id}] Error during {task_type}: {e}")

        except KeyboardInterrupt:
            self.logger.warning("[!] Ctrl+C detected! Initiating graceful shutdown...")

            for future in active_futures.keys():
                self.logger.warning(f"canceling future of {future}")
                future.cancel()
            active_futures.clear()

            dl_pool.shutdown(wait=True, cancel_futures=True)
            sep_pool.shutdown(wait=True, cancel_futures=True)
            render_pool.shutdown(wait=True, cancel_futures=True)

        finally:
            self.shutdown()


if __name__ == "__main__":
    from iantirta.telemetry import netsvc
    netsvc.setup_logger("iantirta")
    urls = [
        "https://youtube.com/watch?v=sdnDInjdWjw&list=RDsdnDInjdWjw&start_radio=1&pp=ygUIQmVybmFkeWGgBwE%3D",
        "https://youtube.com/watch?v=sdnDInjdWjw&list=RDsdnDInjdWjw&start_radio=1&pp=ygUIQmVybmFkeWGgBwE%3D",
    ] * 5
    tasks = [
        {
            "id": i,
            "source": url
        } for i, url in enumerate(urls)
    ]
    BatchKaraoke(tasks).run()