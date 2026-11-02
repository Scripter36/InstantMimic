# Third-party notices

`src/instantmimic_amass/smpl_kinematics.py` derives from MimicKit, Copyright 2025 Xue Bin Peng, under Apache-2.0: https://github.com/xbpeng/MimicKit.

The SMPL names and neutral offsets reproduce its `tools/smpl_to_mimickit/smpl_names.py` and `smpl_constants.py`. The NumPy kinematics helpers translate its rotation utilities. No SMPL body model or AMASS motion data is included.

PHC split and occlusion filtering follow https://github.com/ZhengyiLuo/PHC. PHC metadata is obtained separately by the user. Dependencies retain their own licenses.
