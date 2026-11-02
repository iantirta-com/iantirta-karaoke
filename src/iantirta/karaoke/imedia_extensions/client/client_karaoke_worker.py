
from iantirta.media import IanTirtaAPI

from iantirta.karaoke.imedia_extensions.worker.karaoke_worker import (
    KaraokeWorker,
    _Sentinel,
)


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

        # Override it back
        self.make_json_request = IanTirtaAPI.make_json_request

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
