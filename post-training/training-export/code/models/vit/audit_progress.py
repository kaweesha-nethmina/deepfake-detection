"""Bounded image auditing and visible progress for a remote dataset mount."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import os
from pathlib import Path
import threading
import time


class Progress:
    def __init__(self, label, total=None, interval=10):
        self.label, self.total, self.interval = label, total, interval
        self.count = 0
        self.detail = ""
        self.started = time.monotonic()
        self.stopped = threading.Event()

    def message(self):
        elapsed = time.monotonic() - self.started
        count = f"{self.count:,}" + (f"/{self.total:,}" if self.total is not None else "")
        return f"[{self.label}] completed {count}; elapsed {elapsed:.0f}s. {self.detail}"

    def heartbeat(self):
        while not self.stopped.wait(self.interval):
            print(self.message(), flush=True)


@contextmanager
def stage(label, total=None, interval=10):
    state = Progress(label, total, interval)
    print(f"[{label}] starting", flush=True)
    thread = threading.Thread(target=state.heartbeat, daemon=True)
    thread.start()
    try:
        yield state
    except BaseException:
        print(f"[{label}] interrupted or failed; no successful completion claimed", flush=True)
        raise
    else:
        print(state.message(), flush=True)
        print(f"[{label}] complete", flush=True)
    finally:
        state.stopped.set()
        thread.join(timeout=1)


def discover_images(root, extensions):
    """Do not issue a separate is_file/stat call for every image candidate.

    Actual image opens/decoding remain authoritative; a broken or non-image
    entry with an image extension fails auditing rather than being skipped.
    """
    root = Path(root)
    files = []
    with stage("Discover image paths") as progress:
        for folder in ("Real", "Fake"):
            stack = [root / folder]
            while stack:
                directory = stack.pop()
                progress.detail = f"Listing {directory}; unchanged count means waiting on directory access."
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if Path(entry.name).suffix.lower() in extensions:
                            files.append(Path(entry.path).relative_to(root).as_posix())
                            progress.count = len(files)
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
        progress.detail = "Sorting filenames deterministically."
        files.sort()
    return files


def bounded_map(function, items, workers, progress):
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("AUDIT_WORKERS must be a positive integer")
    executor = ThreadPoolExecutor(max_workers=workers)
    pending = {}
    iterator = iter(enumerate(items))
    results = {}

    def fill():
        while len(pending) < workers * 2:
            try:
                index, item = next(iterator)
            except StopIteration:
                break
            pending[executor.submit(function, item)] = index

    try:
        fill()
        while pending:
            done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                results[index] = future.result()
                progress.count += 1
            progress.detail = (f"At most {workers * 2} queued/running tasks; "
                               "unchanged count means workers have not completed another file.")
            fill()
    except BaseException:
        for future in pending:
            future.cancel()
        # Python cannot cancel a thread blocked inside an OS filesystem read.
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return [results[index] for index in sorted(results)]


def inspect_inventory(root, files, inspect_image, workers):
    with stage("Read, decode and hash images", total=len(files)) as progress:
        return bounded_map(inspect_image, ((root, name) for name in files), workers, progress)


def verify_inventory(root, inventory, sha256_file, workers):
    def verify(row):
        if sha256_file(root / row["filepath"]) != row["sha256"]:
            raise ValueError(f"Dataset content changed: {row['filepath']}")
        return None
    with stage("Verify existing snapshot", total=len(inventory)) as progress:
        bounded_map(verify, inventory, workers, progress)


def timed_split_assignment(assign_splits, inventory, seed):
    with stage("Group duplicates and assign splits") as progress:
        progress.detail = "CPU-only step; completed count is reported on finish, with separate grouping messages."
        result = assign_splits(inventory, seed)
        progress.count = len(inventory)
        return result
