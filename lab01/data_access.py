"""Explicit bounded HTTPS downloads; importing this module performs no I/O."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen
import json


def download_https(url, destination, max_bytes=50_000_000):
    """Download a selected file and record origin and digest; never fetch whole archives implicitly."""
    if urlsplit(url).scheme != "https":
        raise ValueError("Use an HTTPS file URL")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    digest = sha256()
    total = 0
    created_temporary = False
    try:
        with urlopen(url, timeout=30) as response, temporary.open("xb") as out:
            created_temporary = True
            if urlsplit(response.url).scheme != "https":
                raise ValueError("Redirect left HTTPS")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > max_bytes:
                raise ValueError("File exceeds configured transfer limit")
            while chunk := response.read(1024*1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("File exceeds configured transfer limit")
                out.write(chunk)
                digest.update(chunk)
        temporary.rename(destination)
    except Exception:
        if created_temporary and temporary.exists():
            temporary.unlink()
        raise
    record = {"url": url, "bytes": total, "sha256": digest.hexdigest(),
              "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
    destination.with_name(destination.name + ".source.json").write_text(json.dumps(record, indent=2))
    return record
