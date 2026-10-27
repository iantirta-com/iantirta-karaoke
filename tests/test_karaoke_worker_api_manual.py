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
    if len(tasks) > 1:
        tasks[1].download_uris = [
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/41cb69b333f1c3c54fe1a65cc2a30497d5d8919accd833fc1444213579407390?download=true",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/66aab1f2102d8e222ee38ed2edeba204d7e6c612a590417a2f9e5d157167822d?download=true",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/2042ac24592edf6dcc06ada388546b36d7f646831f05f8727f614bcb3347e4d5?download=true",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/e55749bc720244a9f5a744bd56398363ce682a19332bc5a6e121d61aa289a2d3?download=true",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/fe827300e3ad726db42e00489715db7ad9457b002561bc97ef62db49c8d62472?download=true",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/111ac08d4ad67352295b50132398b169802de358d3f15481a69a9c6cec4ecf01?download=true",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/8e117887f2bd248e3476a8a64ca7efffa670427f70ec653c16fa3eb3e755b491?download=true",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/69f0d36cd47175a8a903c777d25945771e2078318094d029559ef7b4633e3788?download=true",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/5f37c42dad06ed4a92b0fbb748c2c90ebcf6048e2263ecf202440632a29dd3fe?download=true",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/74906edfe2cca45793099f6f61419a5d6f953c80a68e5e48aedb61013259b763?download=true",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/7e76d4144c1858fbbdbb9317f430d921bce9b7cccc0ee8ed802dc87b9404aba3?download=true",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/c78131fa732bbecaf4c547b8e189c23356e391337f1a3ad381f193fd71954184?download=true",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/b614abc9b15582c24eca6ef59303a87d90d777d881267736f16df23484c51a6f?download=true",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/71d68550e64a6521d096fff5b59564ab071057b84d06f6876d78afbf535893a1?download=true",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/566e7dcab55366747c0c9e5999ab130268308c9e5ea19baed46db3a9b1a6c791?download=true"
        ]

    if len(tasks) > 2:
        tasks[2].download_uris = {
            "count": 9,
            "files": [
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/52d4c0ea4262cd7ea04688e5e0b3e42e257c6829a063a21de4f14125de15ddf5",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/f1664a82205a61924d13f158a127809d8f96d9aa03b85718e725e7c1cff61e3d",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/fb9cbd08275a0b8592b11a069669d6115c5ba2fc50a060ebd6dfba0a1a89f2af",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/6ffd8424e37f33265ba81fcd3a92b92962aaef30f9f9976b71d9abe8f5a7304c",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/229c9107a75550e4e9dbda182a1242fa685c8eebb44cdda236fe15f3d76ea0b5",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/a83b3aaa86fe5c6d7030cf3ce465aba5ef97f7bbe3108dddae2d45f92cfeecdd",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/7819d64fc57ad6481e0b74ae73fa42d51d15f1c2e45654492f419166192926fe",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/7f14ad1bd8a3cf968478391f1c06dc8de8a7ddeb990b39d9125631bf0eed6383",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/f379cac6657bd23155c9ec21dd415cc6e539bb4df282132df11c13d15de44db3"
            ],
            "provider": "hf_dataset",
            "size": 45730471,
            "suffix": ".mkv"
        }

    if len(tasks) > 3:
        tasks[3].download_uris = [
            "https://drive.google.com/uc?id=1dxKT1VO2ix-8aTlCaX4jOp0AaukA-BKH&export=download"
        ]

    if len(tasks) > 4:
        tasks[4].download_uris = {
            "count": 15,
            "files": [
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/a5/a58faea13f2d5d2d0a9e3443a814bfff30d0ad7de3c45cacc65ba0c128fcfbba",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/3f/3f5528cb691298d4b851baf9e72bf357bd489d5a4f1c60b1bdc08216097184ac",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/a5/a58cb1b1d728b8945952bbb5d855a5f14103ee3725f50530d697469ec6d90128",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/bd/bd25c57a377336d394c3229a9f1ff219a13bfcfbe0db33f6b90dd9b0fefcce22",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/bf/bf42fba8b7924249668ce9408a8d16874905229112e72f179cf4982d4684c319",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/e4/e468941a73f18c5424f07dcd4cbfa3d50e13b580a68f50fc2dc045b3ae24044c",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/01/01180bb40c5cf95f4f273c7c431fc50d0a6979c67e02b780257e5e3fa5f971bd",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/a9/a9da37a96c61bcbd3aff7271a380dcdc857e6e43164506c44e3819496a8784ea",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/07/07e1d498680c861e5de69695bb1d7a0337234b8492cf0f358a8a4a6e6e7a9873",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/66/66682cee9e19a0b7a781991c483ae8f446e1f0ac15e774a6379b406c25eb319e",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/3c/3cbc56addb57828148df3b056f86f7f786e18534dbe712bb0950ad5ad121a96b",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/91/9167cfb6140772d8a1837fb56afc140710853f8109ac07bd261c0ffb59a4eefa",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/e2/e2f2ed7b495fe0b88df7b15b2aa042cc21645196240d8aa4d8a020b990c46c39",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/18/18e782e3bd3e144a5e346faa92951be62031e8a56421a0fba69154dd3cd1a9ac",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/c8/c88b782f382d91521afc1db6d0aa8c150dbdf2af581f58412b67450a0558eeeb"
            ],
            "provider": "hf_dataset",
            "size": 76014260,
            "suffix": ".mkv"
        }

    if len(tasks) > 5:
        tasks[5].download_uris = {
            "count": 4,
            "files": [
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/2a/2a89319ec8aea6b1bcef2758ce4229d1a368cb6ef89a3e1f4723cce6c511bf69",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/cc/ccf49f866971a2919db10dabb5f5b802409afcaa3f7417cb033037cb5a845c26",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/56/569b1fc38d68dfecbcefcf760e5cc0d000cca9f15f1c897f1225280fb38531ac",
            "https://huggingface.co/datasets/yancovenant/karaoke-archive/resolve/main/a0e04c70804a9bb53a86e6a7a4dd2ebfc34ede00378214e7a859bf7fb0834622/ae/aef26219e260ea0faabee0f5ee8ce72ff80aa559c9f178bfb31d04f072eac02f"
            ],
            "provider": "hf_dataset",
            "size": 19360179,
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
