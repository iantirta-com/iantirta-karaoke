
from collections.abc import Callable

from .core.main import KaraokeTask

class BatchKaraoke:
    def __init__(
        self,
        tasks: list[dict[str, ...]],
        *,
        render_hook: Callable | None = None,
        error_hook: Callable | None = None,
    ):
        self._karaoke_tasks = [
            KaraokeTask.from_dict(**task)
            for task in tasks
        ]
        self.render_hook = render_hook
        self.error_hook = error_hook

    @property
    def karaoke_tasks(self) -> list[KaraokeTask]:
        return [task for task in self._karaoke_tasks if task.error is None]

    def run(self) -> None:
        from rich import inspect
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

        separator = None
        
        with (
            ThreadPoolExecutor(max_workers=5) as dl_pool,
            ThreadPoolExecutor(max_workers=1) as sep_pool
        ):
            active_futures = {}
        
            # 1. Start all downloads and add them to our "active tracker"
            for task in self.karaoke_tasks:
                future = dl_pool.submit(task.get_audio_path)
                active_futures[future] = ("download", task)

            future_downloads = {dl_pool.submit(t.get_audio_path): t for t in self.karaoke_tasks}
            future_separations = {}

            while active_futures:
                done, not_done = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
                
                for future in done:
                    task_type, task = active_futures.pop(future)
                    
                    try:
                        res = future.result()
                        
                        if task_type == "download":
                            print(f"[{task.id}] Download complete: {res}")
                            
                            if task.error is None:
                                sep_future = sep_pool.submit(task._separate_vocal_and_instrument, separator)
                                active_futures[sep_future] = ("separate", task)
                                
                        elif task_type == "separate":
                            print(f"[{task.id}] Separation complete: {res}")
                            
                            if task.error is None and getattr(task, "karaoke_type", "") == "basic":
                                # RENDER HERE! 
                                print(f"[{task.id}] Rendering now...")
                                pass
                            else:
                                # Handle other types
                                pass
                                
                    except Exception as e:
                        print(f"[{task.id}] Error during {task_type}: {e}")
            


if __name__ == "__main__":
    urls = [
        "https://youtube.com/watch?v=sdnDInjdWjw&list=RDsdnDInjdWjw&start_radio=1&pp=ygUIQmVybmFkeWGgBwE%3D",
        "https://youtube.com/watch?v=sdnDInjdWjw&list=RDsdnDInjdWjw&start_radio=1&pp=ygUIQmVybmFkeWGgBwE%3D",
    ] * 5
    
    BatchKaraoke({
        "id": i,
        "source": url
    } for i, url in enumerate(urls)).run()