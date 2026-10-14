
import logging
import threading
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from iantirta.karaoke.core.main import KaraokeTask

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
        self.render_hook = render_hook
        self.error_hook = error_hook
        
        self.logger = (logger or _logger)

        self._shutdown_event = threading.Event()

    @property
    def karaoke_tasks(self) -> list[KaraokeTask]:
        return [task for task in self._karaoke_tasks if len(task._error_entries) == 0]

    def shutdown(self):
        """Programmatically signal the batch process to stop."""
        self.logger.warning("Shutdown signal received. Cancelling pending tasks...")
        self._shutdown_event.set()

    def run(self) -> None:
        from rich import inspect

        separator = None
        
        with (
            ThreadPoolExecutor(max_workers=5) as dl_pool,
            ThreadPoolExecutor(max_workers=1) as sep_pool,
            ThreadPoolExecutor(max_workers=1) as render_pool
        ):
            active_futures = {}
        
            # 1. Start all downloads and add them to our "active tracker"
            for task in self.karaoke_tasks:
                future = dl_pool.submit(task.get_audio_path)
                active_futures[future] = ("download", task)

            try:
                while active_futures:
                    done, not_done = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
                    
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
                self.logger.warning("\n[!] Ctrl+C detected! Initiating graceful shutdown...")
                self._shutdown_event.set()

            finally:
                # 5. Cancel all pending futures so they don't start
                if self._shutdown_event.is_set():
                    self.logger.info("Cleaning up remaining tasks...")
                    for future in active_futures.keys():
                        future.cancel()
                    self.logger.info("Waiting for currently running tasks to finish...")

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