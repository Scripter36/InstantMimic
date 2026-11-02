"""Parser for skeleton XML files."""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from pymotion.rotations import quat

Y_UP_TO_Z_UP = quat.to_matrix(
    quat.mul(
        quat.from_angle_axis(np.array([np.pi / 2]), np.array([0, 0, 1])),
        quat.from_angle_axis(np.array([np.pi / 2]), np.array([1, 0, 0])),
    )
).astype(np.float32)


def convert_y_up_to_z_up_position(pos):
    """Convert position from Y-up to Z-up coordinate system."""
    return Y_UP_TO_Z_UP @ pos


def convert_y_up_to_z_up_rotation(rot_matrix):
    """Convert rotation matrix from Y-up to Z-up coordinate system."""
    return Y_UP_TO_Z_UP @ rot_matrix @ Y_UP_TO_Z_UP.T


def rotmat_to_quat_wxyz(rot_matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion in [w, x, y, z] order."""
    rot_matrix = np.asarray(rot_matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(rot_matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rot_matrix[2, 1] - rot_matrix[1, 2]) / scale
        y = (rot_matrix[0, 2] - rot_matrix[2, 0]) / scale
        z = (rot_matrix[1, 0] - rot_matrix[0, 1]) / scale
    elif rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
        scale = (
            math.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2])
            * 2.0
        )
        w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / scale
        z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / scale
    elif rot_matrix[1, 1] > rot_matrix[2, 2]:
        scale = (
            math.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2])
            * 2.0
        )
        w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / scale
        x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / scale
    else:
        scale = (
            math.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1])
            * 2.0
        )
        w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / scale
        x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / scale
        y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / scale
        z = 0.25 * scale
    out = np.array([w, x, y, z], dtype=float)
    out /= np.linalg.norm(out) + 1e-12
    return out


@dataclass
class JointInfo:
    name: str
    type: str  # Free, Ball, Revolute, Weld
    axis: str | None  # X, Y, Z, or None for Free
    lower: np.ndarray | None
    upper: np.ndarray | None
    rotation: np.ndarray
    translation: np.ndarray
    bvh_name: str | None = None
    smpl_jidx: int | None = None


@dataclass
class BodyInfo:
    name: str
    type: str  # Box, Capsule, Sphere
    mass: float
    size: np.ndarray | None  # for Box
    radius: float | None  # for Capsule, Sphere
    height: float | None  # for Capsule
    rotation: np.ndarray
    translation: np.ndarray
    contact: bool
    color: np.ndarray
    mesh: str | None


@dataclass
class NodeInfo:
    name: str
    parent_name: str
    body: BodyInfo
    joint: JointInfo
    is_endeffector: bool = False


class SkeletonParser:
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.nodes: list[NodeInfo] = []
        self.skeleton_name = ""

    def parse(self) -> dict[str, list[NodeInfo]]:
        """Parse skeleton XML file."""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        self.skeleton_name = root.attrib["name"]

        for node in root:
            node_info = self._parse_node(node)
            self.nodes.append(node_info)

        return {"name": self.skeleton_name, "nodes": self.nodes}

    def _parse_node(self, node: ET.Element) -> NodeInfo:
        """Parse a single node."""
        name = node.attrib["name"]
        parent = node.attrib["parent"]
        is_endeffector = node.attrib.get("endeffector", "False") == "True"

        # Parse body
        body_elem = node.find("Body")
        body = self._parse_body(name, body_elem)

        # Parse joint
        joint_elem = node.find("Joint")
        joint = self._parse_joint(name, joint_elem)

        return NodeInfo(name, parent, body, joint, is_endeffector)

    def _parse_body(self, name: str, elem: ET.Element) -> BodyInfo:
        """Parse body element."""
        body_type = elem.attrib["type"]
        mass = float(elem.attrib["mass"])
        contact = elem.attrib.get("contact", "Off") == "On"

        color_str = elem.attrib.get("color", "0.6 0.6 0.6 1.0")
        color = np.array([float(x) for x in color_str.split()])

        # Parse transformation
        trans_elem = elem.find("Transformation")
        rotation = np.array(
            [float(x) for x in trans_elem.attrib["linear"].split()]
        ).reshape(3, 3)
        translation = np.array(
            [float(x) for x in trans_elem.attrib["translation"].split()]
        )

        # convert to Z-up coordinate system
        rotation = convert_y_up_to_z_up_rotation(rotation)
        translation = convert_y_up_to_z_up_position(translation)

        # Parse geometry
        size = None
        radius = None
        height = None

        if body_type == "Box":
            size = np.array([float(x) for x in elem.attrib["size"].split()])
            # convert to Z-up coordinate system
            size = np.abs(convert_y_up_to_z_up_position(size))
        elif body_type == "Capsule":
            radius = float(elem.attrib["radius"])
            height = float(elem.attrib["height"])
        elif body_type == "Sphere":
            radius = float(elem.attrib["radius"])

        mesh_name = elem.attrib.get("obj")
        if mesh_name is not None and mesh_name.strip().lower() in {"", "none", "null"}:
            mesh_name = None

        return BodyInfo(
            name,
            body_type,
            mass,
            size,
            radius,
            height,
            rotation,
            translation,
            contact,
            color,
            mesh_name,
        )

    def _parse_joint(self, name: str, elem: ET.Element) -> JointInfo:
        """Parse joint element."""
        joint_type = elem.attrib["type"]

        # Parse transformation
        trans_elem = elem.find("Transformation")
        rotation = np.array(
            [float(x) for x in trans_elem.attrib["linear"].split()]
        ).reshape(3, 3)
        translation = np.array(
            [float(x) for x in trans_elem.attrib["translation"].split()]
        )

        # convert to Z-up coordinate system
        rotation = convert_y_up_to_z_up_rotation(rotation)
        translation = convert_y_up_to_z_up_position(translation)

        # Parse limits and axis
        axis = None
        lower = None
        upper = None

        if "axis" in elem.attrib:
            axis = np.array([float(x) for x in elem.attrib["axis"].split()])
            # convert to Z-up coordinate system
            axis = convert_y_up_to_z_up_position(axis)

        if "lower" in elem.attrib:
            lower_str = elem.attrib["lower"]
            lower = np.array([float(x) for x in lower_str.split()])
            # convert to Z-up coordinate system
            if lower.shape[0] == 3:
                lower = convert_y_up_to_z_up_position(lower)

        if "upper" in elem.attrib:
            upper_str = elem.attrib["upper"]
            upper = np.array([float(x) for x in upper_str.split()])
            # convert to Z-up coordinate system
            if upper.shape[0] == 3:
                upper = convert_y_up_to_z_up_position(upper)

        # find dominant axis, and if it is negative, flip the lower and upper limits
        axis_str = None
        if axis is not None:
            dominant_axis = np.argmax(np.abs(axis))
            if axis[dominant_axis] < 0:
                assert lower.shape[0] == upper.shape[0] == 1, (
                    "when axis is specified, lower and upper must be 1-dimensional"
                )
                for i in range(lower.shape[0]):
                    lower[i] = -lower[i]
                    upper[i] = -upper[i]
            axis_str = ["X", "Y", "Z"][dominant_axis]

        # find if there is a flipped lower and upper
        if lower is not None and upper is not None:
            for i in range(lower.shape[0]):
                if lower[i] > upper[i]:
                    lower[i], upper[i] = upper[i], lower[i]

        bvh_name = elem.attrib.get("bvh")
        smpl_jidx = (
            int(elem.attrib["smpl_jidx"]) if "smpl_jidx" in elem.attrib else None
        )

        return JointInfo(
            name,
            joint_type,
            axis_str,
            lower,
            upper,
            rotation,
            translation,
            bvh_name,
            smpl_jidx,
        )
