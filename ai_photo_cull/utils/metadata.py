import subprocess
from pathlib import Path


def set_xmp_flag(path: Path, rating=None, label=None, pick=None, reject=None):
    cmd = ["exiftool", "-overwrite_original"]
    if rating is not None:
        cmd.append(f"-XMP:Rating={rating}")
    if label is not None:
        cmd.append(f"-XMP:Label={label}")
    if pick is not None:
        cmd.append(f"-XMP:Pick={pick}")
    if reject is not None:
        cmd.append(f"-XMP:Reject={reject}")
    cmd.append(str(path))
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
