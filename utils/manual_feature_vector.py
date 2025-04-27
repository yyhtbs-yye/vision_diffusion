import json
import os
import numpy as np

def collect_unique_values(json_dir):
    """Scan JSON files to collect unique values for categorical attributes using sets."""
    unique_genders = set()
    unique_emotions = set()
    unique_hair_colors = set()
    unique_glasses = set()
    unique_blur_levels = set()
    unique_exposure_levels = set()
    unique_noise_levels = set()
    unique_accessories = set()
    
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            with open(os.path.join(json_dir, json_file), "r") as f:
                data = json.load(f)
                if not data or len(data) == 0:
                    continue
                attrs = data[0]["faceAttributes"]
                
                # Gender
                unique_genders.add(attrs["gender"].lower())
                
                # Emotion (dominant)
                emotion_dict = attrs["emotion"]
                dominant_emotion = max(emotion_dict, key=emotion_dict.get)
                unique_emotions.add(dominant_emotion)
                
                # Hair Color (primary or bald)
                if attrs["hair"]["bald"] > 0.5:
                    unique_hair_colors.add("bald")
                elif attrs["hair"]["hairColor"]:
                    primary_color = attrs["hair"]["hairColor"][0]["color"]
                    unique_hair_colors.add(primary_color)
                
                # Glasses
                unique_glasses.add(attrs["glasses"])
                
                # Blur level
                if "blur" in attrs:
                    unique_blur_levels.add(attrs["blur"]["blurLevel"])
                
                # Exposure level
                if "exposure" in attrs:
                    unique_exposure_levels.add(attrs["exposure"]["exposureLevel"])
                
                # Noise level
                if "noise" in attrs:
                    unique_noise_levels.add(attrs["noise"]["noiseLevel"])
                
                # Accessories
                if "accessories" in attrs and attrs["accessories"]:
                    for accessory in attrs["accessories"]:
                        if "type" in accessory:
                            unique_accessories.add(accessory["type"])
    
    # Convert sets to sorted lists for consistent indexing
    return {
        "genders": sorted(unique_genders),
        "emotions": sorted(unique_emotions),
        "hair_colors": sorted(unique_hair_colors),
        "glasses": sorted(unique_glasses),
        "blur_levels": sorted(unique_blur_levels),
        "exposure_levels": sorted(unique_exposure_levels),
        "noise_levels": sorted(unique_noise_levels),
        "accessories": sorted(unique_accessories)
    }

def json_to_features(json_data, unique_values):
    """Convert JSON attributes to a structured feature dictionary with list-based one-hot encoded categorical variables."""
    if not json_data or len(json_data) == 0:
        return {}

    attrs = json_data[0]["faceAttributes"]
    
    # Create dictionary to store all features
    features = {}
    
    # Gender (list-based one-hot encoding)
    gender = attrs["gender"].lower()
    gender_onehot = [1.0 if g == gender else 0.0 for g in unique_values["genders"]]
    features["gender"] = gender_onehot
    
    # Age (normalized to 0-1 range)
    features["age"] = min(max(attrs["age"], 0), 100) / 100.0
    
    # Smile (already 0-1 range)
    features["smile"] = attrs["smile"]
    
    # Emotion (list-based one-hot encoding of dominant emotion)
    emotion_dict = attrs["emotion"]
    dominant_emotion = max(emotion_dict, key=emotion_dict.get)
    dominant_emotion_onehot = [1.0 if e == dominant_emotion else 0.0 for e in unique_values["emotions"]]
    features["dominant_emotion"] = dominant_emotion_onehot
    
    # Store all emotion scores directly (continuous 0-1)
    emotion_scores = []
    for emotion in sorted(emotion_dict.keys()):  # Sort to ensure consistent order
        emotion_scores.append(emotion_dict[emotion])
    features["emotion_scores"] = emotion_scores
    
    # Hair Color (list-based one-hot encoding)
    hair_color = None
    if attrs["hair"]["bald"] > 0.5:
        hair_color = "bald"
    elif attrs["hair"]["hairColor"]:
        hair_color = attrs["hair"]["hairColor"][0]["color"]
    
    hair_color_onehot = [1.0 if hc == hair_color else 0.0 for hc in unique_values["hair_colors"]]
    features["hair_color"] = hair_color_onehot
    
    # Hair baldness (continuous 0-1)
    features["hair_bald"] = attrs["hair"]["bald"]
    
    # Glasses (list-based one-hot encoding)
    glasses = attrs["glasses"]
    glasses_onehot = [1.0 if g == glasses else 0.0 for g in unique_values["glasses"]]
    features["glasses"] = glasses_onehot
    
    # Facial Hair (continuous 0-1 values as a list)
    facial_hair = attrs["facialHair"]
    features["facial_hair"] = [
        facial_hair["moustache"],
        facial_hair["beard"],
        facial_hair["sideburns"]
    ]
    
    # Head Pose (continuous, normalized to -1 to 1)
    if "headPose" in attrs:
        head_pose = attrs["headPose"]
        features["head_pose"] = [
            head_pose.get("pitch", 0) / 90.0,
            head_pose.get("roll", 0) / 90.0,
            head_pose.get("yaw", 0) / 90.0
        ]
    else:
        features["head_pose"] = [0.0, 0.0, 0.0]
    
    # Blur (list-based one-hot encoding for level, continuous for value)
    blur_level = ""
    blur_value = 0.0
    if "blur" in attrs:
        blur = attrs["blur"]
        blur_level = blur.get("blurLevel", "")
        blur_value = blur.get("value", 0)
    
    blur_level_onehot = [1.0 if bl == blur_level else 0.0 for bl in unique_values["blur_levels"]]
    features["blur_level"] = blur_level_onehot
    features["blur_value"] = blur_value
    
    # Exposure (list-based one-hot encoding for level, continuous for value)
    exposure_level = ""
    exposure_value = 0.0
    if "exposure" in attrs:
        exposure = attrs["exposure"]
        exposure_level = exposure.get("exposureLevel", "")
        exposure_value = exposure.get("value", 0)
    
    exposure_level_onehot = [1.0 if el == exposure_level else 0.0 for el in unique_values["exposure_levels"]]
    features["exposure_level"] = exposure_level_onehot
    features["exposure_value"] = exposure_value
    
    # Noise (list-based one-hot encoding for level, continuous for value)
    noise_level = ""
    noise_value = 0.0
    if "noise" in attrs:
        noise = attrs["noise"]
        noise_level = noise.get("noiseLevel", "")
        noise_value = noise.get("value", 0)
    
    noise_level_onehot = [1.0 if nl == noise_level else 0.0 for nl in unique_values["noise_levels"]]
    features["noise_level"] = noise_level_onehot
    features["noise_value"] = noise_value
    
    # Makeup (binary list)
    makeup_eye = 0.0
    makeup_lip = 0.0
    if "makeup" in attrs:
        makeup = attrs["makeup"]
        makeup_eye = 1.0 if makeup.get("eyeMakeup", False) else 0.0
        makeup_lip = 1.0 if makeup.get("lipMakeup", False) else 0.0
    
    features["makeup"] = [makeup_eye, makeup_lip]
    
    # Accessories (list-based one-hot encoding for each type)
    accessory_types = set()
    if "accessories" in attrs and attrs["accessories"]:
        for accessory in attrs["accessories"]:
            if "type" in accessory:
                accessory_types.add(accessory["type"])
    
    accessories_onehot = [1.0 if acc_type in accessory_types else 0.0 for acc_type in unique_values["accessories"]]
    features["accessories"] = accessories_onehot
    
    # Occlusion (binary list)
    occlusion_forehead = 0.0
    occlusion_eye = 0.0
    occlusion_mouth = 0.0
    if "occlusion" in attrs:
        occlusion = attrs["occlusion"]
        occlusion_forehead = 1.0 if occlusion.get("foreheadOccluded", False) else 0.0
        occlusion_eye = 1.0 if occlusion.get("eyeOccluded", False) else 0.0
        occlusion_mouth = 1.0 if occlusion.get("mouthOccluded", False) else 0.0
    
    features["occlusion"] = [occlusion_forehead, occlusion_eye, occlusion_mouth]
    
    return features

def process_json_files(json_dir, output_file):
    """Process JSON files to generate structured feature dictionaries with list-based one-hot encoding."""
    # Step 1: Collect unique values
    print("Collecting unique values...")
    unique_values = collect_unique_values(json_dir)
    print("Unique values found:")
    print(f"Genders: {unique_values['genders']}")
    print(f"Emotions: {unique_values['emotions']}")
    print(f"Hair Colors: {unique_values['hair_colors']}")
    print(f"Glasses: {unique_values['glasses']}")
    print(f"Blur Levels: {unique_values['blur_levels']}")
    print(f"Exposure Levels: {unique_values['exposure_levels']}")
    print(f"Noise Levels: {unique_values['noise_levels']}")
    print(f"Accessories: {unique_values['accessories']}")
    
    # Also save feature dimension information
    feature_dimensions = {
        "gender": len(unique_values["genders"]),
        "dominant_emotion": len(unique_values["emotions"]),
        "emotion_scores": len(unique_values["emotions"]),
        "hair_color": len(unique_values["hair_colors"]),
        "glasses": len(unique_values["glasses"]),
        "facial_hair": 3,  # moustache, beard, sideburns
        "head_pose": 3,    # pitch, roll, yaw
        "blur_level": len(unique_values["blur_levels"]),
        "exposure_level": len(unique_values["exposure_levels"]),
        "noise_level": len(unique_values["noise_levels"]),
        "makeup": 2,       # eye, lip
        "accessories": len(unique_values["accessories"]),
        "occlusion": 3     # forehead, eye, mouth
    }
    
    # Step 2: Generate feature dictionaries
    feature_data = []
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            with open(os.path.join(json_dir, json_file), "r") as f:
                data = json.load(f)
                features = json_to_features(data, unique_values)
                if features:
                    image_id = json_file.replace(".json", "")
                    feature_data.append({"image_id": image_id, "features": features})
    
    # Save feature data to a JSON file
    with open(output_file, "w") as f:
        json.dump(feature_data, f, indent=4)
    print(f"Saved {len(feature_data)} feature dictionaries to {output_file}")

    # Save the unique values dictionary and feature dimensions for future reference
    unique_values_file = output_file.replace(".json", "_metadata.json")
    with open(unique_values_file, "w") as f:
        metadata = {
            "unique_values": unique_values,
            "feature_dimensions": feature_dimensions,
            "categorical_features": [
                "gender", "dominant_emotion", "hair_color", "glasses", 
                "blur_level", "exposure_level", "noise_level", "accessories"
            ],
            "continuous_features": [
                "age", "smile", "emotion_scores", "hair_bald", "facial_hair",
                "head_pose", "blur_value", "exposure_value", "noise_value"
            ],
            "binary_features": ["makeup", "occlusion"]
        }
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {unique_values_file}")

# Example usage
json_dir = "datasets/ffhq/feature_json_files"  # Update with your JSON directory
output_file = "ffhq_disentangled_features.json"
process_json_files(json_dir, output_file)