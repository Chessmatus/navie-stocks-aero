"""Fetch selected experimental frames by ZIP byte ranges; never download the full ZIP."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import struct
import subprocess
import zlib

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'data' / 'turbine_1L'
URL = 'https://dataverse.no/api/access/datafile/:persistentId?persistentId=doi:10.18710/SX0ZQT/FWZUIM'
README_URL = 'https://dataverse.no/api/access/datafile/:persistentId?persistentId=doi:10.18710/SX0ZQT/NAFCKS'


def get_bytes(url, path, byte_range=None, limit=30_000_000):
    args = ['curl', '--fail', '--silent', '--show-error', '--location', '--retry', '2',
            '--max-time', '120', '--max-filesize', str(limit), '--output', str(path),
            '--write-out', '%{http_code}']
    if byte_range is not None:
        args += ['--range', byte_range]
    response = subprocess.run(args + [url], check=True, capture_output=True, text=True)
    expected = '206' if byte_range is not None else '200'
    if response.stdout.strip() != expected:
        raise RuntimeError(f'Expected HTTP {expected}, received {response.stdout.strip()}')


def read_index():
    path = DEST / 'archive_index.json'
    if path.exists():
        return json.loads(path.read_text())
    tail = DEST / 'zip_tail.bin'
    get_bytes(URL, tail, '-65557', 100_000)
    data = tail.read_bytes()
    pos = data.rfind(b'PK\x06\x06')
    if pos < 0:
        raise ValueError('Expected ZIP64 end of central directory')
    header = struct.unpack_from('<4sQ2H2L4Q', data, pos)
    size, offset = header[-2:]
    central = DEST / 'zip_central.bin'
    get_bytes(URL, central, f'{offset}-{offset+size-1}', 2_000_000)
    data = central.read_bytes()
    entries = []
    pos = 0
    while data[pos:pos+4] == b'PK\x01\x02':
        h = struct.unpack_from('<4s6H3L5H2L', data, pos)
        _, made, ver, flags, method, tm, date, crc, cs, us, nl, xl, cl, disk, ia, ea, off = h
        name = data[pos+46:pos+46+nl].decode('utf8')
        extra = data[pos+46+nl:pos+46+nl+xl]
        k = 0
        while k+4 <= len(extra):
            tag, length = struct.unpack_from('<HH', extra, k)
            payload = extra[k+4:k+4+length]
            if tag == 1:
                q = 0
                if us == 0xffffffff:
                    us = struct.unpack_from('<Q', payload, q)[0]; q += 8
                if cs == 0xffffffff:
                    cs = struct.unpack_from('<Q', payload, q)[0]; q += 8
                if off == 0xffffffff:
                    off = struct.unpack_from('<Q', payload, q)[0]
            k += 4+length
        entries.append(dict(name=name, method=method, crc32=crc,
                            compressed_size=cs, size=us, offset=off))
        pos += 46+nl+xl+cl
    if pos != size:
        raise ValueError('Central-directory length mismatch')
    path.write_text(json.dumps(entries, indent=2))
    return entries


def fetch_entry(entry):
    name = Path(entry['name']).name
    target = DEST / name
    if target.exists():
        content = target.read_bytes()
        if len(content) == entry['size'] and zlib.crc32(content) == entry['crc32']:
            return record(entry, target, content, cached=True)
        raise ValueError(f'Existing {name} fails ZIP CRC/size verification')
    start = entry['offset']
    length = entry['compressed_size'] + 4096
    if length > 30_000_000:
        raise ValueError('Entry exceeds per-request limit')
    temporary = DEST / (name + '.range')
    try:
        get_bytes(URL, temporary, f'{start}-{start+length-1}')
        data = temporary.read_bytes()
        if data[:4] != b'PK\x03\x04':
            raise ValueError('Missing local ZIP header')
        nl, xl = struct.unpack_from('<HH', data, 26)
        local_name = data[30:30+nl].decode('utf8')
        if local_name != entry['name']:
            raise ValueError('ZIP entry name mismatch')
        offset = 30+nl+xl
        content = zlib.decompress(data[offset:offset+entry['compressed_size']], -15)
        if len(content) != entry['size'] or zlib.crc32(content) != entry['crc32']:
            raise ValueError('Uncompressed ZIP CRC/size mismatch')
        target.write_bytes(content)
    finally:
        temporary.unlink(missing_ok=True)
    return record(entry, target, content, cached=False)


def record(entry, target, content, cached):
    return {'file': str(target.relative_to(ROOT)), 'archive_entry': entry['name'],
            'compressed_bytes': entry['compressed_size'], 'bytes': len(content),
            'crc32': entry['crc32'], 'sha256': hashlib.sha256(content).hexdigest(),
            'byte_offset': entry['offset'], 'cached': cached}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, nargs='+', default=[2732])
    args = parser.parse_args()
    if len(args.frames) > 64 or any(not 1 <= n <= 5457 for n in args.frames):
        raise ValueError('Select at most 64 valid frame numbers')
    DEST.mkdir(parents=True, exist_ok=True)
    index = read_index()
    names = {f'B{n:05}.txt' for n in args.frames} | {'rotor_angle.txt', 'scale_quantities.txt'}
    selected = [e for e in index if Path(e['name']).name in names]
    if len(selected) != len(names):
        raise ValueError('Requested files not found in archive index')
    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for item in pool.map(fetch_entry, selected):
            records.append(item)
            print(f"Verified {item['file']}: {item['bytes']/1e6:.1f} MB", flush=True)
    if not (DEST / '00_README.txt').exists():
        get_bytes(README_URL, DEST / '00_README.txt', limit=100_000)
    manifest_path = DEST / 'subset_provenance.json'
    previous = json.loads(manifest_path.read_text())['files'] if manifest_path.exists() else []
    merged = {r['file']: r for r in previous + records}
    manifest_path.write_text(json.dumps({'dataset_doi': '10.18710/SX0ZQT',
        'case': '1L', 'archive': '1WT_rand_100_2D.zip', 'source_url': URL,
        'extraction': 'HTTP byte ranges + raw DEFLATE; every entry checked against ZIP CRC32',
        'files': sorted(merged.values(), key=lambda r:r['file'])}, indent=2))


if __name__ == '__main__':
    main()
