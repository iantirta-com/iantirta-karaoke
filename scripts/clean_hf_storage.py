import os
from urllib.parse import parse_qs, unquote, urlparse

import requests
from googleapiclient.errors import HttpError
from huggingface_hub import (
    CommitOperationCopy,
    CommitOperationDelete,
)

from iantirta.karaoke.api.karaoke_iantirta_worker import KaraokeWorker

iantirta_api_token = os.environ['IANTIRTA_KARAOKE_API_TOKEN']
base_domain = 'https://karaoke.iantirta.com/json/2'


def get_karaoke_tasks():
    headers = {
        'Authorization': f'bearer {iantirta_api_token}'
    }
    params = {
        'domain': [
            ('status', '!=', 'completed'),
            ('download_uris', '!=', False)
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

def check_hf_storage_and_clean_if_not_found(api, tasks):
    print(f"Cleaning if not found for: {len(tasks)} Task(s)")

    buffer_tasks = []
    BATCH_SIZE = 50

    def _flush():
        if not buffer_tasks:
            return
            
        print(f"Updating {len(buffer_tasks)} tasks in server (setting download_uris = False)...")
        try:
            api.api.make_json_request(
                'karaoke/task/update',
                {'tasks': buffer_tasks},
            )
        except Exception as e:
            print(f"Failed to update database: {e}")
        finally:
            buffer_tasks.clear()

    # We uses this as to make it 1 single api call instead of file_exist multiple times.
    all_files_in_repo = set(api.storage_api._api.list_repo_files(
        repo_id=api.storage_api.repo_id,
        repo_type='dataset'
    ))
    print(len(all_files_in_repo))
    deleted_count = 0
    malformed = 0
    for task in tasks:
        try:
            task_id = task['id']
            title = task['title']
            metadata = task['download_uris']
            files = metadata['files']
            
            if not isinstance(files, list):
                malformed += 1
                print(f"Files is not a list {files}")
                continue

            missing = 0
            for path in files:
                parsed = urlparse(path)
                decoded = unquote(parsed.path).lstrip("/")
                if decoded.startswith('datasets'):
                    decoded = decoded[len(f'datasets/{api.storage_api.repo_id.lstrip("/")}/resolve/main/'):]

                parts = decoded.split('/')
                if len(parts) < 3:
                    print("Parts is less then 3", parts)

                if decoded not in all_files_in_repo:
                    missing += 1
                    # print(
                    #     "Not found for url",
                    #     f'https://huggingface.co/datasets/{api.storage_api.repo_id.lstrip("/")}/resolve/main/{decoded.lstrip("/")}'
                    # )
            
            # print("List", files)
            if missing:
                print(f"Task {task_id} | {title} is missing from hf, removing...")
    #             task['download_uris'] = False
    #             task['status'] = 'waiting'
                deleted_count += 1
    #             buffer_tasks.append(task)
    #         else:
    #             continue
        except Exception as e:
            print(f"Error while checking task {task_id}| Error: {e}")
    #     else:
    #         if len(buffer_tasks) >= BATCH_SIZE:
    #             _flush()

    # _flush()
    print(f"Cleanup complete! deleting {deleted_count} files from HF.")


def migrate_new_prefix_hf_storage(
    api: KaraokeWorker,
    *,
    files=None,
    dry_run: bool = False
):
    if not files:
        files = set(api.storage_api._api.list_repo_files(
            repo_id=api.storage_api.repo_id,
            repo_type='dataset'
        ))
    legacy_files = sorted(
        path for path in files
        if len(parts := path.split("/")) == 2
        and len(parts[0]) == 64
        and len(parts[1]) == 64
        and all(c in "0123456789abcdef" for c in parts[0].lower())
        and all(c in "0123456789abcdef" for c in parts[1].lower())
    )

    moves = []
    conflicts = []
    for source in legacy_files:
        parts = source.split('/')
        if len(parts) < 2:
            conflicts.append((source, "filename shorter than 2 characters"))
            continue

        artist, chunk_name = parts
        new_destination = f"{artist}/{chunk_name[:2]}/{chunk_name}"

        if new_destination in files:
            # Never overwrite or delete an existing destination blindly.
            conflicts.append((source, new_destination))
            continue

        moves.append((source, new_destination))

    print(f"Repository files: {len(files):,}")
    print(f"Legacy files found: {len(legacy_files):,}")
    print(f"Files eligible to move: {len(moves):,}")
    print(f"Conflicts/skipped: {len(conflicts):,}")

    for source, destination in conflicts[:20]:
        print(f"SKIP: {source} -> {destination}")

    if dry_run:
        for source, destination in moves[:20]:
            print(f"PLAN: {source} -> {destination}")
        print("Dry run only. No changes made.")
        return

    migrated = 0
    for start in range(0, len(moves), BATCH_SIZE):
        batch = moves[start:start + BATCH_SIZE]
        operations = []
        for source, destination in batch:
            operations.extend([
                CommitOperationCopy(
                    src_path_in_repo=source,
                    path_in_repo=destination,
                ),
                CommitOperationDelete(
                    path_in_repo=source,
                ),
            ])
        
        api.storage_api._api.create_commit(
            repo_id=api.storage_api.repo_id,
            repo_type='dataset',
            token=api.storage_api.access_token,
            operations=operations,
            commit_message=(
                f"Migrate files to two-character shard directories "
                f"({start + 1}-{start + len(batch)} of {len(moves)})"
            ),
        )
        migrated += len(batch)
        print(f"Migrated {migrated:,}/{len(moves):,}")

    print(f"Migration complete: {migrated:,} files moved.")

def delete_old_without_prefix_hf_storage(
    api: KaraokeWorker,
    *,
    files=None,
    dry_run: bool = False
):
    if not files:
        files = set(api.storage_api._api.list_repo_files(
            repo_id=api.storage_api.repo_id,
            repo_type='dataset'
        ))
    legacy_files = sorted(
        path for path in files
        if len(parts := path.split("/")) == 2
        and len(parts[0]) == 64
        and len(parts[1]) == 64
        and all(c in "0123456789abcdef" for c in parts[0].lower())
        and all(c in "0123456789abcdef" for c in parts[1].lower())
    )

    conflicts = []
    to_delete = []
    for source in legacy_files:
        parts = source.split('/')
        if len(parts) < 2:
            conflicts.append((source, "filename shorter than 2 characters"))
            continue

        artist, chunk_name = parts
        new_destination = f"{artist}/{chunk_name[:2]}/{chunk_name}"

        if new_destination in files:
            # Never overwrite or delete an existing destination blindly.
            to_delete.append((source, new_destination))
            continue

    print(f"Repository files: {len(files):,}")
    print(f"Legacy files found: {len(legacy_files):,}")
    print(f"Files to delete: {len(to_delete):,}")
    print(f"Conflicts/skipped: {len(conflicts):,}")

    if dry_run:
        for source, destination in to_delete[:20]:
            print(f"PLAN: {source} -> {destination}")
        print("Dry run only. No changes made.")
        return

BATCH_SIZE = 100
if __name__ == "__main__":
    api = KaraokeWorker(
        api_url="https://karaoke.iantirta.com",
        api_token=os.environ['IANTIRTA_WORKER_API_TOKEN'],
        worker_id=1,
        storage_options={
            'provider': 'hf_dataset',
            'access_token': os.environ['IANTIRTA_HF_TOKEN'],
            'repo_id': os.environ['IANTIRTA_HF_DATASET_REPO_ID'],
        },
    )
    files = set(api.storage_api._api.list_repo_files(
        repo_id=api.storage_api.repo_id,
        repo_type='dataset'
    ))
    migrate_new_prefix_hf_storage(api, files=files, dry_run=False)
    # tasks = get_karaoke_tasks()
    # print(len(tasks))
    # print(tasks)
    # check_hf_storage_and_clean_if_not_found(api, tasks)
