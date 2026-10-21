from pathlib import Path
from typing import TYPE_CHECKING
import logging

import pytest
from unittest.mock import patch

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from iantirta.karaoke.api.karaoke_iantirta_worker import KaraokeWorker
    from iantirta.karaoke.core.main import KaraokeTask


def test_karaoke_worker_init(karaoke_worker: 'KaraokeWorker'):
    assert karaoke_worker
    assert karaoke_worker.api is not None
    assert karaoke_worker.storage_api is not None
    assert karaoke_worker.storage_provider == "hf_dataset"
    assert karaoke_worker.task_buffer == []
    assert karaoke_worker.max_batch_size > 0


@pytest.mark.parametrize(
    ("method_name", "endpoint", "payload"),
    [
        (
            "claim_task",
            "/karaoke/tasks/get",
            {
                "domain": [("status", "=", "waiting")],
                "limit": 200,
            },
        ),
        (
            "worker_start",
            "/karaoke/worker/start",
            {},
        ),
        (
            "worker_stop",
            "/karaoke/worker/stop",
            {"log_entries": []},
        ),
    ],
)
def test_worker_api_requests(
    karaoke_worker: 'KaraokeWorker',
    method_name: str,
    endpoint: str,
    payload: dict,
    mock_response: "MagicMock"
) -> None:
    mock_response.json.return_value = {
        'result': 'ok',
    }
    with patch.object(karaoke_worker.api, "make_json_request", return_value='ok') as api_request:

        result = getattr(karaoke_worker, method_name)()

        assert result == "ok"
        api_request.assert_called_once_with(
            endpoint,
            payload,
            logger=karaoke_worker.logger,
        )


def test_claim_task_custom_args(karaoke_worker: 'KaraokeWorker', mock_response: "MagicMock", mock_base_http):
    mock_response.json.return_value = {
        'result': 'Any',
    }
    domain = [("status", "=", "ready")]
    with patch.object(karaoke_worker.api, "make_json_request") as api_request:
        result = karaoke_worker.claim_task(domain=domain, limit=7)
        api_request.assert_called_once_with(
            "/karaoke/tasks/get",
            {"domain": domain, "limit": 7},
            logger=karaoke_worker.logger,
        )


def test_worker_claim_task(karaoke_worker: 'KaraokeWorker', mock_response: "MagicMock", tasks):
    mock_response.json.return_value = {
        'result': 'Any',
    }
    karaoke_worker.worker_claim_task(tasks)


def test_worker_start(karaoke_worker: 'KaraokeWorker', mock_response: "MagicMock"):
    mock_response.json.return_value = {
        'result': 'Any',
    }
    karaoke_worker.worker_start()


def test_worker_stop(karaoke_worker: 'KaraokeWorker', mock_response: "MagicMock"):
    mock_response.json.return_value = {
        'result': 'Any',
    }
    karaoke_worker.worker_stop()


def test_upload_and_test_should_buffer(
    karaoke_worker: 'KaraokeWorker',
    karaoke_tasks: 'KaraokeTask',
    tmp_path: Path,
    mock_base_http,
):
    karaoke_worker.max_batch_size = len(karaoke_tasks) + 1

    for task in karaoke_tasks:
        tmp_karaoke_path = tmp_path / f'task_{task.id}.txt'
        tmp_karaoke_path.write_text(f"Dummy data {task.id}")
        
        task.karaoke_path = tmp_karaoke_path
        task.artist = f'artist_{task.id}'
        karaoke_worker._upload_and_post(task)

    assert len(karaoke_worker.task_buffer) == len(karaoke_tasks)
    for task in karaoke_worker.task_buffer:
        assert task.download_uris['suffix'] == '.txt'
        assert task.download_uris['provider'] is not None

    mock_base_http.assert_not_called()


def test_flush_empty_buffer_does_nothing(
    karaoke_worker: 'KaraokeWorker',
    mock_hf_storage,
    mock_base_http,
) -> None:
    
    karaoke_worker.flush()

    mock_hf_storage.assert_not_called()
    mock_base_http.assert_not_called()


def test_flush_uploads_before_posting(
    karaoke_worker: 'KaraokeWorker',
    karaoke_tasks: list['KaraokeTask'],
    tmp_path: Path,
    mock_hf_storage,
    mock_base_http,
    mock_response
) -> None:
    karaoke_worker.max_batch_size = len(karaoke_tasks) + 1

    events: list[str] = []

    def upload_folder(**kwargs) -> None:
        events.append("storage")

    def post_tasks(*args, **kwargs) -> dict:
        events.append("server")
        mock_response.json.return_value = {
            'result': 'Any',
        }
        return mock_response

    mock_hf_storage.side_effect = upload_folder
    mock_base_http.side_effect = post_tasks

    for task in karaoke_tasks:
        tmp_karaoke_path = tmp_path / f'task_{task.id}.txt'
        tmp_karaoke_path.write_text(f"Dummy data {task.id}")
        
        task.karaoke_path = tmp_karaoke_path
        task.artist = f'artist_{task.id}'
        karaoke_worker._upload_and_post(task)

    assert len(karaoke_worker.task_buffer) == len(karaoke_tasks)

    karaoke_worker.flush()

    assert events == ["storage", "server"]
    assert karaoke_worker.task_buffer == []


def test_flushes_on_batch_size(
    karaoke_worker: 'KaraokeWorker',
    karaoke_tasks: list['KaraokeTask'],
    tmp_path: Path,
    mock_hf_storage,
    mock_base_http,
    mock_response
) -> None:
    batch_size = 2
    karaoke_worker.max_batch_size = batch_size

    with patch.object(karaoke_worker.api, "make_json_request") as api_request:

        for task in karaoke_tasks:
            tmp_karaoke_path = tmp_path / f'task_{task.id}.txt'
            tmp_karaoke_path.write_text(f"Dummy data {task.id}")
            
            task.karaoke_path = tmp_karaoke_path
            task.artist = f'artist_{task.id}'
            karaoke_worker._upload_and_post(task)

        # Explicitly flush any remaining partial batch.
        karaoke_worker.flush()

        expected_batches = (
            len(karaoke_tasks) + batch_size - 1
        ) // batch_size

        assert karaoke_worker.task_buffer == []
        assert mock_hf_storage.call_count == expected_batches
        assert api_request.call_count == expected_batches

        requests = api_request.call_args_list
        assert all(
            args.args[0] == "karaoke/task/update"
            for args in requests
        )

        posted_tasks = [
            task
            for request in requests
            for task in request.args[1]["tasks"]
        ]

        assert sorted(task["id"] for task in posted_tasks) == sorted(
            task.id for task in karaoke_tasks
        )


def test_should_flush_on_count(
    karaoke_worker: 'KaraokeWorker',
    karaoke_tasks: 'KaraokeTask',
    tmp_path: Path,
    mock_response: "MagicMock",
):
    mock_response.json.return_value = {
        'result': 'Any',
    }

    karaoke_worker.max_batch_size = max(1, len(karaoke_tasks) // 2)
    task: KaraokeTask
    for task in karaoke_tasks:
        tmp_karaoke_path = tmp_path / f'task_{task.id}.txt'
        tmp_karaoke_path.write_text(f"Dummy data {task.id}")
        
        task.karaoke_path = tmp_karaoke_path
        task.artist = f'artist_{task.id}'
        karaoke_worker._upload_and_post(task)

    assert not karaoke_worker.task_buffer
    assert karaoke_worker.storage_api.upload_folder.call_count == len(karaoke_tasks) / karaoke_worker.max_batch_size
    assert karaoke_worker.api.request.call_count == len(karaoke_tasks) / karaoke_worker.max_batch_size


def test_should_flush_on_count_with_storage_error(
    karaoke_worker: 'KaraokeWorker',
    karaoke_tasks: 'KaraokeTask',
    tmp_path: Path,
    mock_response: "MagicMock",
    mock_hf_storage,
    mock_base_http
):
    # mock_response.json.return_value = {
    #     "error": {
    #         'code': 404,
    #         'message': "task not found",
    #         'data': {
    #             'name': 'error name',
    #             'message': 'error message',
    #             'debug': 'Traceback'
    #         }
    #     }
    # }
    mock_response.json.return_value = {
        'result': 'Any',
    }

    mock_hf_storage.side_effect = Exception("Storage API Upload Failed")

    karaoke_worker.storage_api.staging_dir = tmp_path / 'staging_karaoke'

    karaoke_worker.max_batch_size = max(1, len(karaoke_tasks) // 2)
    task: KaraokeTask
    for task in karaoke_tasks:
        tmp_karaoke_path = tmp_path / f'task_{task.id}.txt'
        tmp_karaoke_path.write_text(f"Dummy data {task.id}")
        
        task.karaoke_path = tmp_karaoke_path
        task.artist = f'artist_{task.id}'
        karaoke_worker._upload_and_post(task)

    assert not karaoke_worker.task_buffer
    assert mock_hf_storage.call_count == len(karaoke_tasks) / karaoke_worker.max_batch_size
    assert mock_base_http.call_count == len(karaoke_tasks) / karaoke_worker.max_batch_size


def test_should_flush_on_count_with_server_error(
    karaoke_worker: 'KaraokeWorker',
    karaoke_tasks: 'KaraokeTask',
    tmp_path: Path,
    mock_response: "MagicMock",
    mock_hf_storage,
    mock_base_http
):
    mock_response.json.return_value = {
        "error": {
            'code': 404,
            'message': "task not found",
            'data': {
                'name': 'error name',
                'message': 'error message',
                'debug': 'Traceback'
            }
        }
    }

    karaoke_worker.storage_api.staging_dir = tmp_path / 'staging_karaoke'

    karaoke_worker.max_batch_size = max(1, len(karaoke_tasks) // 2)
    task: KaraokeTask
    for task in karaoke_tasks:
        tmp_karaoke_path = tmp_path / f'task_{task.id}.txt'
        tmp_karaoke_path.write_text(f"Dummy data {task.id}")
        
        task.karaoke_path = tmp_karaoke_path
        task.artist = f'artist_{task.id}'
        karaoke_worker._upload_and_post(task)

    assert not karaoke_worker.task_buffer
    assert mock_hf_storage.call_count == len(karaoke_tasks) / karaoke_worker.max_batch_size
    assert mock_base_http.call_count == len(karaoke_tasks) / karaoke_worker.max_batch_size


def test_flush_storage_failure_marks_tasks_as_errors(
    karaoke_worker: 'KaraokeWorker',
    karaoke_tasks: list['KaraokeTask'],
    tmp_path: Path,
    mock_hf_storage,
    mock_base_http,
    mock_response,
    caplog: pytest.LogCaptureFixture,
) -> None:
    karaoke_worker.max_batch_size = len(karaoke_tasks) + 1

    with patch.object(karaoke_worker.api, "make_json_request") as api_request:
        for task in karaoke_tasks:
            tmp_karaoke_path = tmp_path / f'task_{task.id}.txt'
            tmp_karaoke_path.write_text(f"Dummy data {task.id}")
            
            task.karaoke_path = tmp_karaoke_path
            task.artist = f'artist_{task.id}'
            karaoke_worker._upload_and_post(task)

        mock_hf_storage.side_effect = OSError("Storage API Upload Failed")

        with caplog.at_level(logging.ERROR):
            karaoke_worker.flush()

        assert karaoke_worker.task_buffer == []
        assert any(
            "Critical error while flushing storage" in record.message
            for record in caplog.records
        )

        # The server must still receive the task updates.
        api_request.assert_called_once()
        request = api_request.call_args

        assert request.args[0] == "karaoke/task/update"

        posted_tasks = request.args[1]["tasks"]
        assert len(posted_tasks) == len(karaoke_tasks)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(['-v', '-s', __file__]))
