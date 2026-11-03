"""The Isaac Lab ``Isaac-Ant-Direct-v0`` benchmark, reimplemented on MuJoCo Warp.

The task id is kept so results are comparable against the original. The
implementation here is independent -- see THIRD_PARTY_NOTICES.md, which also
records that this project is not affiliated with or endorsed by NVIDIA or the
Isaac Lab project.
"""

from . import tasks

__all__ = ["tasks"]
