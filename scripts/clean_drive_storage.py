import os
from urllib.parse import parse_qs, urlparse

import requests
from googleapiclient.errors import HttpError

from iantirta.karaoke.api.karaoke_iantirta_worker import KaraokeWorker

iantirta_api_token = os.environ['IANTIRTA_KARAOKE_API_TOKEN']
base_domain = 'https://karaoke.iantirta.com/json/2'


def get_karaoke_tasks():
    headers = {
        'Authorization': f'bearer {iantirta_api_token}'
    }
    params = {
        'domain': [
            ('download_url', '!=', False)
        ]
    }
    res = requests.post(f'{base_domain}/karaoke.karaoke/search', json=params, headers=headers)
    res.raise_for_status()
    ids = res.json()

    res_read = requests.post(f'{base_domain}/karaoke.karaoke/read', headers=headers, json={
        'ids': ids,
        'fields': ['title', 'download_url', 'download_uris', 'artist']
    })
    res_read.raise_for_status()
    tasks = res_read.json()
    return tasks


def clean_storage(api: KaraokeWorker, tasks: list[dict]):
    storage_api = api.storage_api
    print("Connected to Google Drive API.")

    buffer_tasks = []
    BATCH_SIZE = 50

    def _flush():
        if not buffer_tasks:
            return
            
        print(f"Updating {len(buffer_tasks)} tasks in server (setting download_url = False)...")
        try:
            api.api.make_json_request(
                'karaoke/task/update',
                {'tasks': buffer_tasks},
            )
        except Exception as e:
            print(f"Failed to update database: {e}")
        finally:
            buffer_tasks.clear()

    deleted_count = 0

    for task in tasks:
        task_id = task['id']
        download_url = task['download_url']
        parsed = urlparse(download_url)
        query_params = parse_qs(parsed.query)
        file_id = query_params.get('id', [None])[0]

        if file_id:
            try:
                # 2. Hard delete the file from Google Drive (bypasses trash)
                storage_api.service.files().delete(fileId=file_id).execute()
                print(f"Deleted Drive file {file_id} for Task {task_id}")
                deleted_count += 1
            except HttpError as e:
                # If Google throws a 404, the file is already gone. That's totally fine.
                if e.resp.status == 404:
                    print(f"File {file_id} already deleted (404) for Task {task_id}.")
                else:
                    print(f"Google Drive API error for Task {task_id}: {e}")
            except Exception as e:
                print(f"Unexpected error deleting Task {task_id}: {e}")
        else:
            print(f"Warning: Could not extract file ID from URL: {download_url}")

        # 3. Buffer the task to be updated in Odoo (even if the file was already a 404,
        #    we still want to wipe the phantom URL from the database).
        buffer_tasks.append({
            'id': task_id,
            'download_url': False
        })
        
        if len(buffer_tasks) >= BATCH_SIZE:
            _flush()

    _flush()
    print(f"Cleanup complete! Freed up space by deleting {deleted_count} files from Drive.")


if __name__ == '__main__':
    api = KaraokeWorker(
        # api_url="https://karaoke.iantirta.com",
        api_url='https://localhost:8069',
        api_token=os.environ['IANTIRTA_WORKER_API_TOKEN'],
        worker_id=1,
        storage_options={
            'provider': 'gdrive',
            'access_token': os.environ['IANTIRTA_DRIVE_USER_TOKEN'],
            'root_folder_id': os.environ['IANTIRTA_DRIVE_ROOT_FOLDER_ID'],
        },
        process_task_limit=None,
        max_gpu_pool=1,
        claim_task_domain=[
            ('status', '==', 'waiting'),
            ('download_uris', '==', False,),
            ('download_url', '!=', False)
        ]
    )
    res = api.api.make_json_request(
        'karaoke/task/update',
        {'tasks': [
            {
                'id': 12952,
            }
        ]}
    )
    print(res)
    # tasks = get_karaoke_tasks()
    # # clean_storage(api, tasks)
    # print(len(tasks))
    # for task in tasks:
    #     task['download_url'] = False
    #     assert task['download_uris']
    #     task['log_entries'] = []
    # res = api.api.make_json_request(
    #     'karaoke/task/update',
    #     {'tasks': tasks},
    # )
    # print(res)