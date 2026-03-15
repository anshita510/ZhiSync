"""Device process classes implementing ZhiAware per-device inference."""

from .base_device import BaseDevice, DeviceResult
from .ecg_device import ECGDevice
from .breath_device import BreathDevice
from .motion_device import MotionDevice

__all__ = [
    "BaseDevice", "DeviceResult",
    "ECGDevice", "BreathDevice", "MotionDevice",
]
