from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from iantirta.karaoke.api.karaoke_iantirta_api import IantirtaAPI


def test_iantirta_api_init(iantirta_api: "IantirtaAPI"):
    assert iantirta_api


def test_make_payload(iantirta_api: 'IantirtaAPI'):
    payload = {
        'task': {'any task'}
    }
    normalized_payload = iantirta_api._make_payload(payload)
    assert normalized_payload['params']['task'] == payload['task']


def test_raise_for_status_404(iantirta_api: "IantirtaAPI", mock_response: "MagicMock"):
    mock_response.status_code = 404
    mock_response.json.return_value = {
        "error": {
            'code': 404,
            'message': "task not found",
            'data': {
                'name': 'error name',
                'message': 'error message',
                'debug': 'Traceback'
            }
        }}
    with pytest.raises(OSError, match='Iantirta API Error'):
        iantirta_api.make_json_request('/karaoke/tasks/get', {})


def test_make_json_response(iantirta_api: "IantirtaAPI", mock_response: "MagicMock"):
    mock_response.json.return_value = {
        'result': [1, 2, 3, 4, 5],
    }
    result = iantirta_api.make_json_request('/karaoke/tasks/get', {})
    assert result == [1, 2, 3, 4, 5]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(['-v', '-s', __file__]))
