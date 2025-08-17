# """
# Attribute and target label mapping for data augmentation.
#
# This module provides dictionaries to convert numeric attribute values
# and target labels into human-readable descriptions for prompt generation
# and to map back generated labels to numeric codes.
# """
#
# Mapping of numeric attribute values to descriptive strings for image generation prompts

# attributes = {
#     "lettuce",
#     "potato",
#     "orange",
#     "lemon",
#     "garlic",
#     "cauliflower",
#     "beetroot",
#     "pear",
#     "grapes",
#     "carrot",
#     "tomato",
#     "raddish",
#     "mango",
#     "onion",
#     "pineapple",
#     "kiwi",
#     "bell pepper",
#     "corn",
#     "chilli pepper",
#     "banana",
#     "soy beans",
#     "sweetcorn",
#     "cucumber",
#     "ginger",
#     "spinach",
#     "paprika",
#     "apple",
#     "jalepeno",
#     "sweetpotato",
#     "pomegranate",
#     "eggplant",
#     "cabbage",
#     "watermelon",
#     "peas",
#     "turnip",
# }

attributes = { 
              "female", "male"
              }

ATTRIBUTE_VALUE_MAPPING: dict[str, dict[int, str]] = {
    # "age": {
    #     # 0: "0 to 6 years old",
    #     1: "7 to 13 years old",
    #     2: "14 to 23 years old",
    #     3: "24 to 45 years old",
    #     4: "46 to 65 years old",
    #     5: "66 to 100 years old",
    # },
    # "ethnicity": {
    #     0: "white",
    #     1: "black",
    #     2: "asian",
    #     3: "indian",
    #     4: "other",
    # },
    
    "label": {s: s for s in attributes},
    "Wearing_Hat" : {
        0: "not wearing a hat",
        1: "wearing a hat",
    },
    "Male": {
        0: "female",
        1: "male",
    }
}

# Definition of the target label (name and mapping of numeric codes to descriptions)
TARGET_LABEL_MAPPING: dict[str, object] = {
    # "name": "gender",
    # "mapping": {
    #     0: "female",
    #     1: "male",
    # },
    "name": "gender",
    "mapping": {s: s for s in attributes},
}

"""
# Attribute and target label mapping for data augmentation.
#
# This module provides dictionaries to convert numeric attribute values
# and target labels into human-readable descriptions for prompt generation
# and to map back generated labels to numeric codes.
# """

# Mapping of numeric attribute values to descriptive strings for image generation prompts
# ATTRIBUTE_VALUE_MAPPING: dict[str, dict[int, str]] = {
#     "blond": {
#         0: "does not have blond hair",
#         1: "has blond hair",
#     },
#     "male": {
#         0: "is male",
#         1: "is female",
#     },
# }
#
# # Definition of the target label (name and mapping of numeric codes to descriptions)
# TARGET_LABEL_MAPPING: dict[str, object] = {
#     "name": "eyeglasses",
#     "mapping": {
#         0: "with eyeglasses",
#         1: "without eyeglasses",
#     },
# }
