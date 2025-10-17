import logging
import pandas as pd
from services.generation_clients import get_generation_client
from deep_translator import GoogleTranslator
import random
import nltk
from nltk.corpus import wordnet, stopwords
import re

# Download necessary NLTK data
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    wordnet.synsets('computer')
except LookupError:
    nltk.download('wordnet')


_llm_client = None
_translate_client = None
_stop_words = None


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = get_generation_client(modality="text")
    return _llm_client


def _get_translate_client():
    global _translate_client
    if _translate_client is None:
        _translate_client = GoogleTranslator(source='auto', target='de')
    return _translate_client


def _get_stop_words():
    global _stop_words
    if _stop_words is None:
        _stop_words = set(stopwords.words('english'))
    return _stop_words


def generate_text_samples(attributes: dict, num_samples: int, config: dict, example_texts: list[str] = None) -> list[dict]:
    """
    Generates new text samples using an LLM based on attributes and a configuration.
    If example_texts is provided, it generates samples similar to the examples.
    """
    client = _get_llm_client()
    generated_data = []
    
    label_index = list(attributes.values())[0]
    label_description = config["label_mapping"].get(label_index)
    text_column = config["text_column"]
    target_attribute = config["target_attribute"]
    
    if example_texts:
        examples_str = "\n".join(f"- {text}" for text in example_texts)
        prompt = config["prompt_template_examples"].format(
            num_samples=num_samples,
            label_description=label_description,
            examples_str=examples_str
        )
    else:
        prompt = config["prompt_template_no_examples"].format(
            num_samples=num_samples,
            label_description=label_description
        )

    logging.info(f"Generating {num_samples} text samples with prompt: '{prompt[:150]}'...")
    
    try:
        response = client.generate_text(prompt)
        headlines = response.strip().split('\n')
        
        for headline in headlines:
            if headline:
                # Clean up potential leading characters like '-' 
                cleaned_headline = headline.strip()
                if cleaned_headline.startswith('- '):
                    cleaned_headline = cleaned_headline[2:]
                
                generated_data.append({
                    text_column: cleaned_headline.strip(),
                    target_attribute: label_description 
                })
        
        if len(generated_data) > num_samples:
            generated_data = generated_data[:num_samples]

        logging.info(f"Successfully generated {len(generated_data)} samples.")
        return generated_data

    except Exception as e:
        logging.error(f"Failed to generate text samples: {e}")
        return []




def augment_with_backtranslation(
    texts: list[str], attributes: dict, num_samples: int, lang: str = 'de'
) -> list[dict]:
    """
    Augments text using back-translation.
    """
    client = _get_translate_client()
    augmented_texts = []
    
    samples_to_augment = random.choices(texts, k=num_samples)
    label = list(attributes.values())[0]

    logging.info(f"Performing back-translation for {num_samples} samples for label '{label}'...")

    for text in samples_to_augment:
        try:
            # Translate to the intermediate language
            client.target = lang
            intermediate_text = client.translate(text)
            
            # Translate back to English
            client.target = 'en'
            final_text = client.translate(intermediate_text)
            
            augmented_texts.append({
                "text": final_text,
                "label": label
            })
        except Exception as e:
            logging.warning(f"Back-translation failed for a sample: {e}")
            # If it fails, just add the original text back to keep the count
            augmented_texts.append({
                "text": text,
                "label": label
            })
            
    logging.info(f"Generated {len(augmented_texts)} samples via back-translation.")
    return augmented_texts

def augment_with_eda(
    texts: list[str], attributes: dict, num_samples: int, eda_strategy: str
) -> list[dict]:
    """
    Augments text using a specified EDA strategy.
    """
    augmented_texts = []
    samples_to_augment = random.choices(texts, k=num_samples)
    label = list(attributes.values())[0]

    logging.info(f"Performing EDA with strategy '{eda_strategy}' for {num_samples} samples for label '{label}'...")

    for text in samples_to_augment:
        augmented_text = text
        if eda_strategy == 'sr':
            augmented_text = _synonym_replacement(text)
        elif eda_strategy == 'ri':
            augmented_text = _random_insertion(text)
        elif eda_strategy == 'rs':
            augmented_text = _random_swap(text)
        elif eda_strategy == 'rd':
            augmented_text = _random_deletion(text)
        
        augmented_texts.append({
            "text": augmented_text,
            "label": label
        })

    logging.info(f"Generated {len(augmented_texts)} samples via EDA strategy '{eda_strategy}'.")
    return augmented_texts


def _synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in _get_stop_words()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = _get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def _random_insertion(sentence, n=1):
    words = sentence.split()
    for _ in range(n):
        new_synonym = _get_random_synonym(words)
        if new_synonym:
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, new_synonym)
    return ' '.join(words)


def _random_swap(sentence, n=1):
    words = sentence.split()
    if len(words) < 2:
        return sentence
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

def _random_deletion(sentence, p=0.1):
    words = sentence.split()
    if len(words) == 1:
        return sentence
    new_words = [word for word in words if random.uniform(0, 1) > p]
    if len(new_words) == 0:
        return random.choice(words)
    return ' '.join(new_words)


def _get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def _get_random_synonym(words):
    random_word = None
    while random_word is None or random_word in _get_stop_words():
        if not words:
            return None
        random_word = random.choice(words)
    synonyms = _get_synonyms(random_word)
    if synonyms:
        return random.choice(synonyms)
    return None
