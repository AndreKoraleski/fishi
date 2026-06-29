"""WoodScape semantic segmentation classes."""

CLASS_NAMES = [
    "void",
    "road",
    "lanemarks",
    "curb",
    "person",
    "rider",
    "vehicles",
    "bicycle",
    "motorcycle",
    "traffic_sign",
]
VOID_ID = 0

PROMPTS = {
    class_id: name.replace("_", " ")
    for class_id, name in enumerate(CLASS_NAMES)
    if class_id != VOID_ID
}
