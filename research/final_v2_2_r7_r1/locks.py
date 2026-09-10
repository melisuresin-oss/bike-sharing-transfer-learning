"""OS-managed cross-process lifetime locks for R7 on Windows and POSIX.

The lock byte, not the metadata file, is authoritative. Metadata is advisory
and is written only while the process owns the lock.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

class LockUnavailable(RuntimeError):
    pass

class LifetimeLock:
    def __init__(self, lock_path: Path, metadata_path: Path, metadata: dict[str, Any]):
        self.lock_path=lock_path; self.metadata_path=metadata_path; self.metadata=metadata; self.handle=None
    def acquire(self):
        self.lock_path.parent.mkdir(parents=True,exist_ok=True); self.metadata_path.parent.mkdir(parents=True,exist_ok=True)
        handle=self.lock_path.open("a+b",buffering=0)
        if self.lock_path.stat().st_size == 0:
            handle.write(b"\0"); handle.flush(); os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except (OSError,BlockingIOError) as exc:
            handle.close(); owner=None
            try: owner=json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except Exception: pass
            raise LockUnavailable(f"exclusive lock is already held: {self.lock_path}; owner={owner!r}") from exc
        self.handle=handle
        payload=json.dumps(self.metadata,indent=2,ensure_ascii=False,allow_nan=False).encode("utf-8")+b"\n"
        pending=self.metadata_path.with_name(self.metadata_path.name+f".pending-{os.getpid()}")
        with pending.open("wb") as out:
            out.write(payload); out.flush(); os.fsync(out.fileno())
        os.replace(pending,self.metadata_path)
        return self
    def release(self):
        if self.handle is None: return
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(),fcntl.LOCK_UN)
        finally:
            self.handle.close(); self.handle=None
    @property
    def held(self): return self.handle is not None
    def __enter__(self): return self.acquire()
    def __exit__(self,exc_type,exc,tb): self.release()

def probe(lock_path: Path) -> bool:
    probe_meta=lock_path.with_name(lock_path.name+".probe-owner.json")
    lock=LifetimeLock(lock_path,probe_meta,{"purpose":"non-authoritative-status-probe","pid":os.getpid()})
    try: lock.acquire()
    except LockUnavailable: return True
    finally:
        lock.release()
        try: probe_meta.unlink()
        except FileNotFoundError: pass
    return False