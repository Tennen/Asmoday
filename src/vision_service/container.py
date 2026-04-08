from dataclasses import dataclass

from vision_service.runtime.manager import RuntimeManager
from vision_service.settings import Settings
from vision_service.vision.backend import VisionBackend


@dataclass(slots=True)
class ServiceContainer:
    settings: Settings
    backend: VisionBackend
    manager: RuntimeManager
