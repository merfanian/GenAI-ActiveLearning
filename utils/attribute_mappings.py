"""
Attribute and target label mapping for data augmentation.

This module provides dictionaries to convert numeric attribute values
and target labels into human-readable descriptions for prompt generation
and to map back generated labels to numeric codes.
"""

# Mapping of numeric attribute values to descriptive strings for image generation prompts
ATTRIBUTE_VALUE_MAPPING: dict[str, dict[int, str]] = {
    "age": {
        0: "child",
        1: "teenager",
        2: "young adult",
        3: "adult",
        4: "middle-aged",
        5: "senior",
        6: "elderly",
    },
    "ethnicity": {
        0: "white",
        1: "black",
        2: "asian",
        3: "indian",
        4: "other",
    },
}

# Definition of the target label (name and mapping of numeric codes to descriptions)
TARGET_LABEL_MAPPING: dict[str, object] = {
    "name": "gender",
    "mapping": {
        0: "male",
        1: "female",
    },
}