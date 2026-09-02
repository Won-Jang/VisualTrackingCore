# Central registry used to expose and construct available tracking adapters.

from adapters.apriltag_adapter import AprilTagAdapter
from adapters.aruco_adapter import ArUcoAdapter
from adapters.mediapipe_adapter import MediaPipeFaceAdapter
from adapters.opentrack_adapter import OpenTrackAdapter


# Add a new adapter here to make it available in the main UI.
ADAPTERS = {
    AprilTagAdapter.name: AprilTagAdapter,
    ArUcoAdapter.name: ArUcoAdapter,
    MediaPipeFaceAdapter.name: MediaPipeFaceAdapter,
    OpenTrackAdapter.name: OpenTrackAdapter,
}


def adapter_names() -> list[str]:
    # Return adapter display names in UI order.
    return list(ADAPTERS.keys())


def create_adapter(name: str, target_id: int | None = 0, **kwargs):
    # Instantiate an adapter by its display name.
    try:
        adapter_class = ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown adapter: {name}") from exc

    return adapter_class(target_id=target_id, **kwargs)
