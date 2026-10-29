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
            #('status', '!=', 'completed'),
            ('download_uris', '!=', False),
            #('download_url', '!=', False)
        ]
    }
    res = requests.post(f'{base_domain}/karaoke.karaoke/search', json=params, headers=headers)
    res.raise_for_status()
    ids = res.json()

    res_read = requests.post(f'{base_domain}/karaoke.karaoke/read', headers=headers, json={
        'ids': ids,
        'fields': ['status', 'title', 'download_url', 'download_uris', 'artist']
    })
    res_read.raise_for_status()
    tasks = res_read.json()
    return tasks


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

    deleted = 0
    for start in range(0, len(to_delete), BATCH_SIZE):
        batch = to_delete[start:start + BATCH_SIZE]
        operations = []
        for source, destination in batch:
            operations.extend([
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
                f"Deleted files because of duplicating with new prefix "
                f"({start + 1}-{start + len(batch)} of {len(to_delete)})"
            ),
        )
        deleted += len(batch)
        print(f"Deleted {deleted:,}/{len(to_delete):,}")

    print(f"Deleted complete: {deleted:,} files delete.")


def update_server_karaoke_to_use_prefix_and_clean_up_if_not_exists(
    api: KaraokeWorker,
    *,
    files=None,
    dry_run: bool = False,
):
    tasks = get_karaoke_tasks()
    print('Karaoke Tasks found:', len(tasks))

    conflicts = []
    missing: dict[dict, list] = []
    to_update: dict[dict, list] = []
    for task in tasks:
        try:
            metadata = task['download_uris']

            if not isinstance(metadata, dict):
                conflicts.append((task, 'metadata not a dict'))
                continue

            task_files = metadata['files']
            if not isinstance(task_files, list):
                conflicts.append((task, 'files is not a list'))
                continue

            size = metadata['size']
            count = metadata['count']
            if size < (1 * 1024 * 1024):
                if count == 1:
                    missing.append((task, task_files))
                    continue
                if size == 0:
                    # on old metadata count still havent added yet.
                    # need a better way to handle or update.
                    pass
                else:
                    conflicts.append((
                        task,
                        'file size is too small: '
                        f'{size} and file count: {count}\n'
                        f'Files: {task_files}'
                    ))
                    continue

            has_change = False
            is_missing = False
            new_files = []
            for file in task_files:
                if not file.startswith('http'):
                    conflicts.append((task, 'path doesnt start with http'))
                    break

                parsed = urlparse(file)
                decoded = unquote(parsed.path).lstrip("/")
                if not decoded.startswith('datasets'):
                    conflicts.append((task, 'path is not a huggingface repo dataset'))
                    break

                path = decoded[len(f'datasets/{api.storage_api.repo_id.lstrip("/")}/resolve/main/'):]
                parts = path.split('/')
                new_path = path
                if len(parts) <= 3:
                    if len(parts) == 2:
                        artist, chunk_name = parts
                        new_path = f'{artist}/{chunk_name[:2]}/{chunk_name}'
                        new_files.append(new_path)
                        has_change = True
                    elif len(parts) == 3:
                        new_files.append(path)
                    else:
                        raise OSError('path parts is less than 2')
                else:
                    raise OSError('path parts is more than 3')

                if new_path not in files:
                    is_missing = True
                    missing.append((task, task_files))
                    break

            if has_change and not is_missing:
                assert len(new_files) == len(task_files), (
                    'new files length and recorded one is different'
                    f'Old: {len(task_files)}, New: {len(new_files)}'
                )
                to_update.append((task, new_files))
      
        except Exception as e:
            conflicts.append((task, e))

    print('Conflicts found:', len(conflicts))
    print('Missing found:', len(missing))
    print('To Update found:', len(to_update))
    
    for task, reason in conflicts[:20]:
        print(f"SKIP: {task['id']} -> {reason}")

    if dry_run:
        for task, files in missing[:20]:
            print(f"Clear: {task['id']} -> {task['status']}")
        for task, new_files in to_update[:20]:
            print(f"Updating {task['id']} -> {task['status']}")
        print("Dry run only. No changes made.")
        return

    buffer_tasks = []
    flushed = 0

    def _flush():
        if not buffer_tasks:
            return

        print(f"Updating {len(buffer_tasks)} tasks in server...")
        try:
            api.api.make_json_request(
                'karaoke/task/update',
                {'tasks': buffer_tasks},
            )
        # except Exception as e:
        #     print(f"Failed to update database: {e}")
        #     raise
        finally:
            nonlocal flushed
            flushed += len(buffer_tasks)
            buffer_tasks.clear()

    for task, new_files in to_update:
        updated_files = []
        for file in new_files:
            if not file.startswith('http'):
                file = api.storage_api.download_endpoint + file
            updated_files.append(file)
        task['download_uris']['files'] = updated_files
        task['status'] = 'completed'
        task['log_entries'] = []  # Required by old api
        buffer_tasks.append(task)

    _flush()

    for task, files in missing:
        task['status'] = 'waiting'
        task['download_uris'] = False
        task['log_entries'] = [] # Required for old api
        buffer_tasks.append(task)

    _flush()

    print(
        f"Update complete: {flushed:,} flushed. "
        f"To Update: {len(to_update)}. "
        f"Cleared: {len(missing)}."
    )

        

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
    delete_old_without_prefix_hf_storage(api, files=files, dry_run=False)
    update_server_karaoke_to_use_prefix_and_clean_up_if_not_exists(
        api,
        files=files,
        dry_run=False
    )
    # tasks = get_karaoke_tasks()
    # print(len(tasks))
    # print(tasks)
    # check_hf_storage_and_clean_if_not_found(api, tasks)
