import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call, patch

import pytest

if TYPE_CHECKING:
    from iantirta.karaoke.core.main import KaraokeTask


@pytest.fixture
def mock_response() -> MagicMock:
    response = MagicMock()
    response.headers = {}
    response.url = "http://localhost:8069"
    response.status_code = 200
    response.json.return_value = {} 
    return response


@pytest.fixture
def mock_base_http(mock_response):
    """
    Patches BaseHTTP.request and keeps it active.
    Using 'yield' ensures the patch stays alive for the entire test.
    """
    from iantirta.media import BaseHTTP
    
    with patch.object(BaseHTTP, 'request', return_value=mock_response) as mock_req:
        yield mock_req


@pytest.fixture
def mock_storage_result() -> MagicMock:
    result = MagicMock()
    return result


@pytest.fixture
def mock_gdrive_storage(mock_storage_result):
    from iantirta.media import DriveStorage

    with patch.object(DriveStorage, 'upload_folder', return_value=mock_storage_result) as mock_req:
        yield mock_req


@pytest.fixture
def mock_hf_storage(mock_storage_result):
    from iantirta.media import HFDatasetStorage

    with patch.object(HFDatasetStorage, 'upload_folder', return_value=mock_storage_result) as mock_req:
        yield mock_req


@pytest.fixture
def iantirta_api(mock_base_http):
    from iantirta.karaoke.api.karaoke_iantirta_api import IantirtaAPI

    return IantirtaAPI(
        api_url="http://localhost:8069",
        api_token=None,
        worker_id=None,
    )


@pytest.fixture
def karaoke_worker(mock_base_http, mock_hf_storage):
    from iantirta.karaoke.api.karaoke_iantirta_worker import KaraokeWorker

    return KaraokeWorker(
        api_url="http://localhost:8069",
        api_token=None,
        worker_id=None,
        storage_options={
            'provider': 'hf_dataset',
            'access_token': None,
            'repo_id': None,
        }
    )


@pytest.fixture
def tasks() -> list[dict]:
    return [
        {
            'id': 1,
            'source': "Any",
        },
        {
            'id': 2,
            'source': "Any",
        }
    ]


@pytest.fixture
def karaoke_tasks(tasks) -> list['KaraokeTask']:
    from iantirta.karaoke.core.main import KaraokeTask
    return [KaraokeTask.from_dict(**task) for task in tasks]
