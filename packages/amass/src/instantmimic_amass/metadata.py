"""Convert the trusted PHC occlusion metadata pickle to a portable NPZ."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path

from .phc import save_phc_occlusion_metadata_npz

# Recorded from the metadata used by the research preprocessing pipeline.
# This is a local provenance pin, not a checksum published by PHC.
RECORDED_SHA256 = "2c336550d452e5769bc12d199c285bb3717af651f3691ee18ac6b82f2de0a3d8"
PHC_REVISION = "96e92bb83cd91a4ad162e155ea3c1e50d5001dc7"
DRIVE_FILE_ID = "1uzFkT2s_zVdnAohPWHOLFcyRDq372Fmc"


def convert_metadata(
    source: Path, destination: Path, *, expected_sha256=RECORDED_SHA256
):
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"PHC metadata checksum mismatch: expected {expected_sha256}, got {actual}"
        )
    import joblib

    payload = joblib.load(source)
    if not isinstance(payload, Mapping):
        raise ValueError("PHC metadata must be a mapping")
    save_phc_occlusion_metadata_npz(payload, destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--expected-sha256",
        default=RECORDED_SHA256,
        help="Only override after independently verifying trusted upstream bytes",
    )
    args = parser.parse_args(argv)
    convert_metadata(args.input, args.output, expected_sha256=args.expected_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
