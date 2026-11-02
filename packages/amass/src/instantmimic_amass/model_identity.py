"""Portable identity for the complete compiled target model."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def compiled_model_sha256(model: mujoco.MjModel) -> str:
    """Hash MuJoCo's MJB representation, including compiled includes and assets.

    Unlike the root XML file hash, this observes changed nested includes,
    meshes, height fields and textures. Include the MuJoCo version so model
    format or compiler changes cannot reuse an earlier processing result.
    Call once per processing job and reuse the digest in its artifacts.
    """
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    digest = hashlib.sha256(mujoco.__version__.encode("ascii"))
    digest.update(buffer)
    return digest.hexdigest()


def model_file_sha256(path: Path) -> str:
    return compiled_model_sha256(mujoco.MjModel.from_xml_path(str(path)))


def has_external_model_files(path: Path) -> bool:
    """Whether a root XML hash alone cannot establish its model dependencies."""
    return any(element.get("file") for element in ET.parse(path).iter())
