from dataclasses import dataclass
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import uint32, float64, array


@dataclass
class SimResetSignal(IdlStruct):
    episode_id: uint32
    nut_pos: array[float64, 3]  # filled by scene.py in reset_done (zeros in the request)
    target_pos: array[float64, 3]
    base_pos: array[float64, 3] # robot floating base position after settling (filled by scene.py)
    base_quat: array[float64, 4]    # robot floating base orientation (w, x, y, z)
    success: bool
