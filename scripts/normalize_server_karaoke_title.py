import os
import requests
import re
import unicodedata

from iantirta.karaoke.api.karaoke_iantirta_worker import KaraokeWorker

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_WHITESPACE = re.compile(r"\s+")
_REPEATED_DASHES = re.compile(r"-+")

_RESERVED_WINDOWS_NAMES = re.compile(
    r"^(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?$",
    re.IGNORECASE,
)

_MAX_FILENAME_UTF16_UNITS = 240


iantirta_api_token = os.environ['IANTIRTA_KARAOKE_API_TOKEN']
base_domain = 'https://karaoke.iantirta.com/json/2'


def get_karaoke_tasks():
    headers = {
        'Authorization': f'bearer {iantirta_api_token}'
    }
    params = {
        'domain': [
            ('title', '!=', False),
        ]
    }
    res = requests.post(f'{base_domain}/karaoke.karaoke/search', json=params, headers=headers)
    res.raise_for_status()
    ids = res.json()

    res_read = requests.post(f'{base_domain}/karaoke.karaoke/read', headers=headers, json={
        'ids': ids,
        'fields': ['status', 'title', 'artist', 'download_url', 'download_uris']
    })
    res_read.raise_for_status()
    tasks = res_read.json()
    return tasks


def _normalize_title(title: str) -> str:
    """Normalize a title so it can safely be used as a filename across platforms."""
    if not isinstance(title, str):
        raise TypeError("title must be a string")

    title = unicodedata.normalize("NFC", title)

    # Replace characters forbidden in Windows filenames or used as separators.
    title = _INVALID_FILENAME_CHARS.sub("-", title)

    # Normalize whitespace and replacement characters.
    title = _WHITESPACE.sub(" ", title)
    title = _REPEATED_DASHES.sub("-", title)

    # Windows rejects filenames ending in spaces or periods.
    title = title.strip().rstrip(". ")

    if not title or title in {".", ".."}:
        title = "untitled"

    # Windows reserves these names even when an extension is present.
    if _RESERVED_WINDOWS_NAMES.fullmatch(title):
        title = f"_{title}"

    # Limit UTF-16 code units, which is more appropriate for Windows than
    # Python's len() when filenames contain supplementary Unicode characters.
    result = []
    units = 0

    for char in title:
        char_units = len(char.encode("utf-16-le")) // 2
        if units + char_units > _MAX_FILENAME_UTF16_UNITS:
            break
        result.append(char)
        units += char_units

    title = "".join(result).rstrip(". ")

    return title or "untitled"
    
def normalize_title(
    api: KaraokeWorker,
    *,
    dry_run: bool = False
):
    tasks = get_karaoke_tasks()
    print('Karaoke Tasks found:', len(tasks))

    new_titles = []
    for task in tasks:
        old_title = task["title"]
        new_title = _normalize_title(old_title)
    
        if new_title != old_title:
            new_titles.append((task, new_title))

    print('To Title Update found:', len(new_titles))

    if dry_run:
        for task, new_title in new_titles[:20]:
            print(f"Plan: {task['title']} -> {new_title}")
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
        finally:
            nonlocal flushed
            flushed += len(buffer_tasks)
            buffer_tasks.clear()

    for task, new_title in new_titles:
        task['title'] = new_title
        task['log_entries'] = []
        buffer_tasks.append(task)

    _flush()

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
    normalize_title(api, dry_run=False)
