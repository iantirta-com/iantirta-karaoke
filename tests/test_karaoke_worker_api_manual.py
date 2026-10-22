import os

import pytest

from iantirta.karaoke.core.main import KaraokeTask

pytestmark = pytest.mark.manual


@pytest.fixture
def karaoke_worker():
    from iantirta.karaoke.api.karaoke_iantirta_worker import KaraokeWorker

    return KaraokeWorker(
        api_url="http://localhost:8069",
        api_token=os.environ['IANTIRTA_WORKER_API_TOKEN'],
        worker_id=1,
        storage_options={
            'provider': 'hf_dataset',
            'access_token': None,
            'repo_id': None,
        }
    )


def test_worker_all(karaoke_worker):
    res_worker_start = karaoke_worker.worker_start()
    res_claim_task = karaoke_worker.claim_task()
    res_worker_claim_task = karaoke_worker.worker_claim_task(res_claim_task)
    
    tasks = [KaraokeTask.from_dict(**task) for task in res_claim_task]

    if len(tasks) > 0:
        tasks[0].download_uris = {
            "count": 6,
            "files": [
                "5aea0582e6ab3ca7cbb1768427e46f136f9943c69272f13f01d7769e84e7ab7c/c3be996616d715d71aa9e91ba64857974096ae719be532cae806a32b28b5d96f",
                "5aea0582e6ab3ca7cbb1768427e46f136f9943c69272f13f01d7769e84e7ab7c/5d18219e69a48950d9942d7e99e3351b28eb25dfb7cbd70679f4eadb845cbcd7",
                "5aea0582e6ab3ca7cbb1768427e46f136f9943c69272f13f01d7769e84e7ab7c/982d1dabbafd49a9f0280f6cbfe5fcb93c000c94793e8f7416fd69b5c7bc1910",
                "5aea0582e6ab3ca7cbb1768427e46f136f9943c69272f13f01d7769e84e7ab7c/035131d57f0a3ae7ea2055c30c340484c543754b07ea64c83e4565c9e3c38ca4",
                "5aea0582e6ab3ca7cbb1768427e46f136f9943c69272f13f01d7769e84e7ab7c/653d2e0f8b531082f479e83b170cb28aa36bcb88b85f1f43b00285bd5ddb0b1c",
                "5aea0582e6ab3ca7cbb1768427e46f136f9943c69272f13f01d7769e84e7ab7c/669dcb7faa16768732133325d2af29e850b50a893022d24081f162ed814c3ab6"
            ],
            "provider": "hf_dataset",
            "size": 29823831,
            "suffix": ".mkv"
        }

    karaoke_worker.task_buffer = tasks
    res_flush_to_server = karaoke_worker._flush_task_to_server()
    res_worker_stop = karaoke_worker.worker_stop()

    assert res_worker_start == True
    assert isinstance(res_claim_task, list)
    assert res_worker_claim_task == True
    assert res_worker_stop == True

    from rich import inspect as i
    i(tasks)

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(['-v', '-s', __file__]))
