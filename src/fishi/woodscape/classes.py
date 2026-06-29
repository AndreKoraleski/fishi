"""WoodScape semantic segmentation classes.

Names and order match WoodScape's official taxonomy (seg_annotation_info.json: class_names with
class_indexes 0-9). Id 0 is void and is excluded from the text prompts.
"""

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
CLASS_COUNT = len(CLASS_NAMES)
ID_TO_NAME = dict(enumerate(CLASS_NAMES))

PROMPTS = {
    class_id: name.replace("_", " ")
    for class_id, name in enumerate(CLASS_NAMES)
    if class_id != VOID_ID
}
