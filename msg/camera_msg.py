from dataclasses import dataclass
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import (
    uint8,
    uint32,
    uint64,
    sequence,
)

@dataclass
class CameraImage(IdlStruct):
    frame_id: uint64
    timestamp: uint64
    width: uint32
    height: uint32
    encoding: str
    data: sequence[uint8]