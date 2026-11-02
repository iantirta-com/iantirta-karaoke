
from iantirta.media import IanTirtaAPI

from iantirta.karaoke.imedia_extensions.worker.karaoke_worker import (
    KaraokeWorker,
    _Sentinel,
)

__all__ = ["ClientKaraokeWorker"]


class ClientKaraokeWorker(KaraokeWorker):
    """
    Wrapper for client karaoke worker to api.

    Instead of using worker token, we are using
    user generated api token instead.
    """
    def __init__(
        self,
        worker_token : str | None = None,
        worker_id: int | None = None,
        **kwargs
    ):
        super().__init__(worker_token=worker_token, worker_id=worker_id, **kwargs)

        if self.access_token is None:
            raise ValueError('please defined ``access_token`` to use ClientKaraokeWroker')

        if not self.claim_task_domain:
            self.claim_task_domain = [('status', '=', 'waiting')]

    # --------
    # Helper api communication method
    # --------

    def make_json_request(
        self,
        method: str = 'GET',
        path: str = '',
        params: dict | None = None,
    ):
        return IanTirtaAPI.make_json_request(
            self,
            method=method,
            path=path,
            params=params,
        )

    # --------
    # Api communication method
    # --------

    def worker_claim_task(self, tasks):
        """
        Method that makes api call to the server.
        Indicating that the tasks will be processed by this client.

        This function exists because of how different
        api call to ``karaoke/tasks/get`` can be processed.
        it would not always be processed by the same worker instance
        or any other cases.

        :param list[dict[str, Any]] tasks:
        """
        return self.make_json_request(
            'POST',
            '/json/2/karaoke.karaoke/write',
            {
                'ids': [
                    task['id']
                    for task in tasks
                ],
                'vals': {
                    'status': 'processing',
                }
            }
        )

    def worker_start(self):
        self.logger.info("Starting client karaoke worker")

    def worker_stop(self):
        self.logger.info("Stopping client karaoke worker")

    def claim_tasks(self, domain = _Sentinel, limit = _Sentinel):
        """
        Method that makes api call to the server
        for getting task that needed to be processed.

        :param Optional[list[tuple]] domain:
        :param Optional[int] limit:
            if ``None``, would get everything all at once.
            if ``notset``, would use this instance limit.
        :return: list of dictionary represanting karaoke tasks from the server.
        :rtype: list[dict[str, Any]]
        """
        if domain is _Sentinel:
            domain = self.claim_task_domain
        
        if limit is _Sentinel:
            limit = self.process_task_limit

        karaoke_ids = self.make_json_request(
            'POST',
            '/json/2/karaoke.karaoke/search',
            {
                'domain': domain,
                'limit': limit,
            }
        )

        if not karaoke_ids:
            return []

        return self.make_json_request(
            'POST',
            '/json/2/karaoke.karaoke/read',
            {
                'ids': karaoke_ids,
                'fields': [
                    'title',
                    'artist',
                    'duration',
                    'lyric',
                    'status',
                    'source_url',
                    'karaoke_type'
                ]
            }
        )

    def post_tasks(self, tasks):
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
            'POST',
            '/json/2/karaoke.karaoke/client_post_tasks',
            {'tasks': [
                task.to_dict()
                for task in tasks
            ]},
        )
