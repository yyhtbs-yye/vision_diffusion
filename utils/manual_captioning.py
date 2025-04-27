import json
import os
import random

# Dictionaries for synonyms and descriptive adjectives
SYNONYMS = {
    "gender": {"male": ["man", "male", "gentleman"], "female": ["woman", "female", "lady"]},
    "age": {
        "young": ["youthful", "young", "juvenile"],
        "adult": ["adult", "grown", "mature"],
        "middle-aged": ["middle-aged", "seasoned", "midlife"],
        "senior": ["senior", "elderly", "older"]
    },
    "emotion": {
        "happiness": ["joyful", "cheerful", "happy", "delighted"],
        "neutral": ["neutral", "calm", "composed", "serene"],
        "sadness": ["sad", "melancholy", "sorrowful"],
        "anger": ["angry", "irate", "furious"],
        "surprise": ["surprised", "astonished", "shocked"],
        "fear": ["fearful", "afraid", "anxious"],
        "disgust": ["disgusted", "repulsed"],
        "contempt": ["contemptuous", "disdainful"]
    },
    "smile": {
        "smiling": ["smiling", "grinning", "beaming", "smirking"],
        "not smiling": ["not smiling", "serious", "stoic", "expressionless"]
    },
    "hair_color": {
        "brown": ["brown", "brunette", "chestnut"],
        "black": ["black", "ebony", "jet-black"],
        "blonde": ["blonde", "golden", "flaxen"],
        "red": ["red", "auburn", "ginger"],
        "gray": ["gray", "silver", "ashen"],
        "white": ["white", "platinum", "snowy"]
    },
    "glasses": {
        "wearing glasses": ["wearing glasses", "with spectacles", "in glasses"],
        "no glasses": ["without glasses", "bare-eyed"]
    }
}

DESCRIPTIVE_ADJECTIVES = {
    "hair": ["vibrant", "sleek", "lustrous", "flowing", "shiny"],
    "emotion": ["radiant", "subtle", "intense", "warm", "striking"]
}

# Sentence templates for varied structures
TEMPLATES = [
    "A {age} {gender} {smile}, {emotion} in expression, with {hair} hair and {glasses}.",
    "{gender} of {age} age, {smile} with a {emotion} demeanor, sporting {hair} hair and {glasses}.",
    "A {smile} {age} {gender} with {hair} hair, showing a {emotion} expression, and {glasses}.",
    "With {hair} hair, a {age} {gender} appears {emotion} and {smile}, {glasses}.",
    "A {emotion} {age} {gender} {smile}, characterized by {hair} hair and {glasses}.",
    "Radiating a {emotion} vibe, a {age} {gender} is {smile} with {hair} hair and {glasses}."
]

def bin_age(age):
    """Convert numeric age to categorical description."""
    if age < 18:
        return "young"
    elif 18 <= age < 40:
        return "adult"
    elif 40 <= age < 60:
        return "middle-aged"
    else:
        return "senior"

def get_dominant_emotion(emotion_dict):
    """Return the emotion with the highest confidence score."""
    return max(emotion_dict, key=emotion_dict.get)

def attributes_to_caption(json_data):
    """Convert JSON attributes to a varied, natural language caption."""

    if not json_data or len(json_data) == 0:
        return "No attributes found."

    attrs = json_data[0]["faceAttributes"]
    
    # Extract attributes
    gender = attrs["gender"].lower()
    age = bin_age(attrs["age"])
    emotion = get_dominant_emotion(attrs["emotion"])
    smile = "smiling" if attrs["smile"] > 0.5 else "not smiling"
    hair_color = attrs["hair"]["hairColor"][0]["color"] if attrs["hair"]["hairColor"] else "unknown"
    bald = attrs["hair"]["bald"] > 0.5
    glasses = "wearing glasses" if attrs["glasses"] != "NoGlasses" else "no glasses"
    facial_hair = ""
    if gender == "male" and any(attrs["facialHair"][key] > 0.5 for key in ["moustache", "beard", "sideburns"]):
        facial_hair_parts = []
        if attrs["facialHair"]["moustache"] > 0.5:
            facial_hair_parts.append("moustache")
        if attrs["facialHair"]["beard"] > 0.5:
            facial_hair_parts.append("beard")
        if attrs["facialHair"]["sideburns"] > 0.5:
            facial_hair_parts.append("sideburns")
        facial_hair = " and ".join(facial_hair_parts)

    # Randomly select synonyms
    gender = random.choice(SYNONYMS["gender"][gender])
    age = random.choice(SYNONYMS["age"][age])
    emotion = random.choice(SYNONYMS["emotion"][emotion])
    smile = random.choice(SYNONYMS["smile"][smile])
    glasses = random.choice(SYNONYMS["glasses"][glasses])

    hair = random.choice(SYNONYMS["hair_color"].get(hair_color, [hair_color]))

    # Add descriptive adjectives with 50% probability
    if not bald and random.random() > 0.5:
        hair = f"{random.choice(DESCRIPTIVE_ADJECTIVES['hair'])} {hair}"
    if random.random() > 0.5:
        emotion = f"{random.choice(DESCRIPTIVE_ADJECTIVES['emotion'])} {emotion}"

    # Select a random template
    template = random.choice(TEMPLATES)

    # Format caption
    caption = template.format(age=age, gender=gender, smile=smile, emotion=emotion, hair=hair, glasses=glasses)
    
    # Append facial hair if present
    if facial_hair:
        caption = caption.rstrip(".") + f", with a {facial_hair}."

    return caption

def process_json_files(json_dir, output_file):
    """Process all JSON files and save varied captions."""
    captions = []
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            with open(os.path.join(json_dir, json_file), "r") as f:
                data = json.load(f)
                caption = attributes_to_caption(data)
                image_id = json_file.replace(".json", "")
                captions.append({"image_id": image_id, "caption": caption})
    
    # Save captions to a JSON file
    with open(output_file, "w") as f:
        json.dump(captions, f, indent=4)
    print(f"Saved {len(captions)} captions to {output_file}")

# Example usage
json_dir = "datasets/ffhq/json"  # Update with your JSON directory
output_file = "ffhq_captions_varied.json"
process_json_files(json_dir, output_file)