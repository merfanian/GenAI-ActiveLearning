# Mapping of attribute values to descriptive strings for image generation prompts.
# This provides a centralized place for standard, reusable mappings.

# For Adience gender classification
ADIENCE_GENDER_MAPPING = {
    "label": {
        "female": "female",
        "male": "male",
    },
    "ethnicity": {
        "middle eastern": "middle eastern",
        "white": "white",
        "black": "black",
        "asian": "asian",
        "indian": "indian",
        "latino hispanic": "latino hispanic"
    }
}
ADIENCE_GENDER_TARGET = {
    "name": "label",
    "mapping": {
        "female": "female",
        "male": "male",
    },
}

FFHQ_GENDER_MAPPING = {
    "gender": {
        "female": "female",
        "male": "male",
    },
    "ethnicity": {
        "middle eastern": "middle eastern",
        "white": "white",
        "black": "black",
        "asian": "asian",
        "indian": "indian",
        "latino hispanic": "latino hispanic"
    }
}
FFHQ_GENDER_TARGET = {
    "name": "gender",
    "mapping": {
        "female": "female",
        "male": "male",
    },
}

# For UTKFace race classification
UTKFACE_RACE_MAPPING = {
    "race": {
        0: "a White person",
        1: "a Black person",
        2: "an Asian person",
        3: "an Indian person",
        4: "an person of other races (such as Hispanic, Latino, Middle Eastern)",
    }
}
UTKFACE_RACE_TARGET = {
    "name": "race",
    "mapping": {
        0: "White",
        1: "Black",
        2: "Asian",
        3: "Indian",
        4: "Other",
    },
}

FRUITS_LABEL_MAPPING = {
    "label": {
        "lettuce": "lettuce",
        "potato": "potato",
        "orange": "orange",
        "lemon": "lemon",
        "garlic": "garlic",
        "cauliflower": "cauliflower",
        "beetroot": "beetroot",
        "pear": "pear",
        "grapes": "grapes",
        "carrot": "carrot",
        "tomato": "tomato",
        "raddish": "raddish",
        "mango": "mango",
        "onion": "onion",
        "pineapple": "pineapple",
        "kiwi": "kiwi",
        "bell pepper": "bell pepper",
        "corn": "corn",
        "chilli pepper": "chilli pepper",
        "banana": "banana",
        "soy beans": "soy beans",
        "sweetcorn": "sweetcorn",
        "cucumber": "cucumber",
        "ginger": "ginger",
        "spinach": "spinach",
        "paprika": "paprika",
        "apple": "apple",
        "jalepeno": "jalepeno",
        "sweetpotato": "sweetpotato",
        "pomegranate": "pomegranate",
        "eggplant": "eggplant",
        "cabbage": "cabbage",
        "watermelon": "watermelon",
        "peas": "peas",
        "turnip": "turnip",
    }
}
FRUITS_LABEL_TARGET = {
    "name": "label",
    "mapping": {
        "lettuce": "lettuce",
        "potato": "potato",
        "orange": "orange",
        "lemon": "lemon",
        "garlic": "garlic",
        "cauliflower": "cauliflower",
        "beetroot": "beetroot",
        "pear": "pear",
        "grapes": "grapes",
        "carrot": "carrot",
        "tomato": "tomato",
        "raddish": "raddish",
        "mango": "mango",
        "onion": "onion",
        "pineapple": "pineapple",
        "kiwi": "kiwi",
        "bell pepper": "bell pepper",
        "corn": "corn",
        "chilli pepper": "chilli pepper",
        "banana": "banana",
        "soy beans": "soy beans",
        "sweetcorn": "sweetcorn",
        "cucumber": "cucumber",
        "ginger": "ginger",
        "spinach": "spinach",
        "paprika": "paprika",
        "apple": "apple",
        "jalepeno": "jalepeno",
        "sweetpotato": "sweetpotato",
        "pomegranate": "pomegranate",
        "eggplant": "eggplant",
        "cabbage": "cabbage",
        "watermelon": "watermelon",
        "peas": "peas",
        "turnip": "turnip",
    },
}

ANIMALS_LABEL_MAPPING = {
    "label": {
        "butterfly": "butterfly",
        "cat": "cat",
        "chicken": "chicken",
        "cow": "cow",
        "dog": "dog",
        "elephant": "elephant",
        "horse": "horse",
        "ragno": "ragno",
        "sheep": "sheep",
        "squirrel": "squirrel",
    },
}

ANIMALS_LABEL_TARGET = {
    "name": "label",
    "mapping": {
        "butterfly": "butterfly",
        "cat": "cat",
        "chicken": "chicken",
        "cow": "cow",
        "dog": "dog",
        "elephant": "elephant",
        "horse": "horse",
        "ragno": "ragno",
        "sheep": "sheep",
        "squirrel": "squirrel",
    },
}

# Default fallbacks for backward compatibility
ATTRIBUTE_VALUE_MAPPING = ADIENCE_GENDER_MAPPING
TARGET_LABEL_MAPPING = ADIENCE_GENDER_TARGET
