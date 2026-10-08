"""Import the result packet printed by this task's Colab export cell."""
import argparse
import base64
import hashlib
import json
import lzma
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("session", type=Path)
args = parser.parse_args()
text = args.session.read_text(encoding="utf-8")
packets = re.findall(r"LAB21_TRANSFER_V1:([A-Za-z0-9+/=]+)", text)
hashes = re.findall(r"LAB21_SHA256:([a-f0-9]{64})", text)
if not packets or not hashes:
    raise SystemExit("No completed Colab export packet in the task transcript")
packed = base64.b64decode(packets[-1], validate=True)
assert hashlib.sha256(packed).hexdigest() == hashes[-1], "Packet checksum mismatch"
files = json.loads(lzma.decompress(packed))
out = Path(__file__).resolve().parents[1] / "results"
out.mkdir(exist_ok=True)
for name, content in files.items():
    assert Path(name).name == name and Path(name).suffix in {".json", ".csv", ".txt"}
    (out / name).write_bytes(content.encode("utf-8"))
print(f"Imported {len(files)} artifacts; packet checksum verified")
print("\n".join(sorted(files)))
