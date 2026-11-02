"""Parser for muscle XML files."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from .skeleton_xml import convert_y_up_to_z_up_position


@dataclass
class WaypointAttachmentInfo:
    """A weighted body-local attachment for one muscle waypoint."""

    body: str
    position: np.ndarray
    weight: float = 1.0


@dataclass
class WaypointInfo:
    """Waypoint information for muscle path."""

    body: str
    position: np.ndarray
    attachments: list[WaypointAttachmentInfo] = field(default_factory=list)


@dataclass
class MuscleUnitInfo:
    """Information for a single muscle unit."""

    name: str
    f0: float  # Maximum isometric force
    lm: float  # Optimal muscle fiber length
    lt: float  # Tendon slack length
    pen_angle: float  # Pennation angle
    lmax: float  # Maximum muscle length
    waypoints: list[WaypointInfo] = field(default_factory=list)


class MuscleParser:
    """Parser for muscle XML files."""

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.muscles: list[MuscleUnitInfo] = []

    def parse(self) -> list[MuscleUnitInfo]:
        """Parse muscle XML file."""
        self.muscles = []
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        for unit in root.findall("Unit"):
            for muscle_info in self._parse_unit_muscles(unit):
                # Only add muscles that have waypoints
                if len(muscle_info.waypoints) > 0:
                    self.muscles.append(muscle_info)

        return self.muscles

    def _parse_unit_muscles(self, unit: ET.Element) -> list[MuscleUnitInfo]:
        """Parse one body-fixed polyline per muscle."""
        if any(child.tag != "Waypoint" for child in unit):
            raise ValueError("Muscle metadata requires body-fixed Waypoint children.")
        return [self._parse_unit(unit)]

    def _parse_unit(self, unit: ET.Element) -> MuscleUnitInfo:
        """Parse a single muscle unit."""
        name = unit.attrib["name"]
        f0 = float(unit.attrib["f0"])
        lm = float(unit.attrib["lm"])
        lt = float(unit.attrib["lt"])
        pen_angle = float(unit.attrib["pen_angle"])
        lmax = float(unit.attrib["lmax"])

        waypoints = self._parse_waypoints(unit)

        return MuscleUnitInfo(
            name=name,
            f0=f0,
            lm=lm,
            lt=lt,
            pen_angle=pen_angle,
            lmax=lmax,
            waypoints=waypoints,
        )

    def _parse_waypoints(self, parent: ET.Element) -> list[WaypointInfo]:
        """Parse the fixed points of one muscle."""
        return [
            self._parse_waypoint(waypoint) for waypoint in parent.findall("Waypoint")
        ]

    def _parse_waypoint(self, waypoint: ET.Element) -> WaypointInfo:
        """Parse a single waypoint."""
        if set(waypoint.attrib) - {"body", "p"} or len(waypoint):
            raise ValueError(
                "Muscle waypoints accept only a body and a fixed position."
            )
        body = waypoint.attrib["body"]
        position_str = waypoint.attrib["p"].strip()
        position = np.array([float(x) for x in position_str.split()])
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Muscle waypoint positions require three finite values.")

        # Convert to Z-up coordinate system
        position = convert_y_up_to_z_up_position(position)

        return WaypointInfo(body=body, position=position)

    def get_muscle_by_name(self, name: str) -> MuscleUnitInfo | None:
        """Get a muscle unit by name."""
        for muscle in self.muscles:
            if muscle.name == name:
                return muscle
        return None

    def get_muscles_by_body(self, body_name: str) -> list[MuscleUnitInfo]:
        """Get all muscles that have waypoints on a specific body."""
        result = []
        for muscle in self.muscles:
            for waypoint in muscle.waypoints:
                if waypoint.body == body_name:
                    result.append(muscle)
                    break
        return result

    def get_muscle_names(self) -> list[str]:
        """Get all muscle names."""
        return [muscle.name for muscle in self.muscles]

    def get_all_body_names(self) -> set[str]:
        """Get all unique body names referenced in muscle waypoints."""
        bodies = set()
        for muscle in self.muscles:
            for waypoint in muscle.waypoints:
                bodies.add(waypoint.body)
        return bodies
