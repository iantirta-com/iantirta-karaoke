import pytest
import os

from iantirta.karaoke.core.main import KaraokeTask

pytestmark = pytest.mark.manual


@pytest.fixture
def karaoke_worker():
    from iantirta.karaoke.api.karaoke_iantirta_worker import KaraokeWorker

    return KaraokeWorker(
        api_url="http://localhost:8069",
        api_token=os.environ['IANTIRTA_WORKER_API_TOKEN'],
        worker_id=9,
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

    karaoke_worker.task_buffer = tasks
    res_flush_to_server = karaoke_worker._flush_task_to_server()
    res_worker_stop = karaoke_worker.worker_stop()

    assert res_worker_start == True
    assert isinstance(res_claim_task, list)
    assert res_worker_claim_task == True
    assert res_worker_stop == True

    from rich import inspect as i
    i(res_flush_to_server)

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(['-v', '-s', __file__]))
