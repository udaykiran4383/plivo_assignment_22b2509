import json
import random
import re
import string
from faker import Faker
from num2words import num2words
from tqdm import tqdm
import os

fake = Faker()

# Entity types to generate
ENTITY_TYPES = ["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE", "CITY", "LOCATION"]

def generate_clean_sample():
    """Generates a clean sample with entities using Faker."""
    text = ""
    entities = []
    
    # Templates for sentence generation to ensure variety
    templates = [
        "My name is {PERSON_NAME} and I live in {CITY}.",
        "Contact me at {EMAIL} or {PHONE}.",
        "I was born on {DATE} in {LOCATION}.",
        "My credit card number is {CREDIT_CARD}.",
        "Please send the package to {LOCATION}, {CITY}.",
        "Call {PERSON_NAME} at {PHONE} regarding the meeting on {DATE}.",
        "Is {EMAIL} your current email address?",
        "The event is at {LOCATION} on {DATE}.",
        "I need to update my card {CREDIT_CARD}.",
        "{PERSON_NAME} visited {CITY} last week."
    ]
    
    template = random.choice(templates)
    
    # Helper to track current position
    current_len = 0
    parts = []
    
    # Regex to find placeholders like {PERSON_NAME}
    pattern = re.compile(r"\{(\w+)\}")
    last_end = 0
    
    for match in pattern.finditer(template):
        # Add text before the placeholder
        pre_text = template[last_end:match.start()]
        parts.append(pre_text)
        current_len += len(pre_text)
        
        entity_type = match.group(1)
        entity_value = ""
        
        if entity_type == "PERSON_NAME":
            entity_value = fake.name()
        elif entity_type == "CITY":
            entity_value = fake.city()
        elif entity_type == "LOCATION":
            entity_value = fake.address()
        elif entity_type == "EMAIL":
            entity_value = fake.email()
        elif entity_type == "PHONE":
            entity_value = fake.phone_number()
        elif entity_type == "DATE":
            entity_value = fake.date()
        elif entity_type == "CREDIT_CARD":
            entity_value = fake.credit_card_number()
            
        parts.append(entity_value)
        entities.append({
            "start": current_len,
            "end": current_len + len(entity_value),
            "label": entity_type,
            "text": entity_value # Store original text for debugging/verification
        })
        current_len += len(entity_value)
        
        last_end = match.end()
        
    # Add remaining text
    parts.append(template[last_end:])
    
    full_text = "".join(parts)
    return full_text, entities

def inject_noise_and_realign(text, entities):
    """
    Injects STT noise and realigns entity spans.
    Noise:
    - Lowercase
    - Remove punctuation (except specific replacements)
    - @ -> " at "
    - . -> " dot " (for emails/urls mostly, but simple logic: replace all dots?)    # Requirement: "Special characters are written out (e.g., 'dot' instead of '.', 'at' instead of '@')"
    - Numbers -> words
    """
    
    noisy_text = ""
    new_entities = []
    
    # We process the text character by character (or chunk by chunk) to track alignment
    # But doing it char by char is hard with num2words expansion.
    # Better approach: Split text into tokens/segments, process each, and track the cumulative offset shift.
    
    # However, entities are character spans. 
    # Let's create a map from original char index to new char index.
    
    # 1. Lowercase is 1:1 mapping (mostly).
    # 2. Punctuation removal/replacement changes length.
    # 3. Number expansion changes length.
    
    # Strategy: Iterate through the original text. Build noisy text. 
    # Maintain a mapping: original_index -> new_index (start of the corresponding segment)
    
    mapping = {} # original_idx -> new_idx
    
    i = 0
    new_i = 0
    
    while i < len(text):
        char = text[i]
        mapping[i] = new_i
        
        replacement = char.lower()
        
        # Check for numbers
        if char.isdigit():
            # Grab the full number sequence
            num_str = ""
            j = i
            while j < len(text) and text[j].isdigit():
                num_str += text[j]
                j += 1
            
            # If we found a multi-digit number
            if len(num_str) > 0:
                # Decide whether to convert to words or keep as digits
                # Stress test has digits, so we need to expose the model to them.
                # Let's say 50% chance to keep as digits.
                if random.random() < 0.5:
                    replacement = num_str
                else:
                    try:
                        word_rep = num2words(int(num_str))
                        # Remove dashes and commas
                        word_rep = word_rep.replace("-", " ").replace(",", "")
                        replacement = word_rep
                    except:
                        replacement = num_str # Fallback
                
                # Update mapping for all digits in this number
                # All original digits map to the start of the replacement
                for k in range(len(num_str)):
                    mapping[i + k] = new_i 
                
                noisy_text += replacement
                new_i += len(replacement)
                i += len(num_str)
                continue

        # Check for special chars
        if char == "@":
            replacement = " at "
        elif char == ".":
            replacement = " dot "
        elif char in string.punctuation:
            # Randomly keep some punctuation (like hyphens) to match stress test?
            # Stress test has "6522-0922".
            if char == "-" and random.random() < 0.3:
                replacement = "-"
            else:
                replacement = " " 
        
        # Append
        noisy_text += replacement
        
        new_i += len(replacement)
        i += 1
        
    # Add a mapping for the end of the string
    mapping[len(text)] = new_i
    
    # Clean up double spaces in noisy text? 
    # If we do that, we need to adjust mapping again. 
    # Let's do a second pass or be careful.
    # Simple approach: Don't clean up spaces yet, or handle it carefully.
    # Requirement: "Noisy STT text... No punctuation... Lowercase only".
    # Extra spaces are probably fine or expected.
    
    # Now realign entities
    new_entity_list = []
    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        
        # Find new start
        # We need to find the mapping for 'start'. 
        # If 'start' points to a character that was removed (e.g. punctuation), we might need to look ahead?
        # But entities usually start on alphanumeric chars.
        
        new_start = mapping.get(start)
        new_end = mapping.get(end)
        
        # If the entity contained characters that expanded (like numbers), new_end should reflect that.
        # mapping[end] should point to the position *after* the last character of the entity.
        
        if new_start is not None and new_end is not None:
            # Verify the text matches roughly (ignoring case/punctuation)
            # original_snippet = text[start:end]
            # new_snippet = noisy_text[new_start:new_end]
            
            # Refine boundaries: trim whitespace from the result if any
            # STT usually doesn't have leading/trailing space for entities?
            # Actually, if we replaced "@" with " at ", the entity "user@email" -> "user at email".
            # The span should cover "user at email".
            
            new_entity_list.append({
                "start": new_start,
                "end": new_end,
                "label": ent["label"]
            })
            
    return noisy_text, new_entity_list

def generate_dataset(filename, num_samples):
    with open(filename, "w") as f:
        for i in tqdm(range(num_samples), desc=f"Generating {filename}"):
            clean_text, entities = generate_clean_sample()
            noisy_text, noisy_entities = inject_noise_and_realign(clean_text, entities)
            
            # Final cleanup of noisy text (collapse spaces)
            # If we collapse spaces, we break the mapping we just made.
            # So let's just leave it or be very careful. 
            # "user  at  email" is probably fine for a noisy STT simulation.
            # But let's try to normalize spaces and shift indices?
            # Too complex for now. Let's stick to the generated noisy text.
            
            record = {
                "id": f"{os.path.basename(filename).split('.')[0]}_{i}",
                "text": noisy_text,
                "entities": noisy_entities
            }
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    generate_dataset("data/train.jsonl", 800) # 500-1000
    generate_dataset("data/dev.jsonl", 200)   # 100-200
