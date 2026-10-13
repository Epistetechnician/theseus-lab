"""Procedural V36 corpus rules; instantiated prompts stay in external custody.

State slices: theseus-lab-sandbox-v1, theseus-pretrained-whole-block-transfer-v36,
global-experiment-assurance-lifecycle-v1.
"""
import hashlib
import random

import replacement_loss_v36_core as c


def records(family, seed, partition, count, entropy='preparation'):
    """No data/model-dependent fixture selection or retries."""
    if count < 1 or partition not in ('visible', 'adapt', 'donor', *c.STRATA):
        raise ValueError('fixture partition')
    rng = random.Random(c.seed_for(family, seed, partition, entropy))
    values = []
    for i in range(count):
        tag = hashlib.sha256(c.canonical([c.PROTOCOL, family, seed, partition, entropy, i])).hexdigest()[:12]
        a, b, d = [rng.randint(2, 97) for _ in range(3)]
        actor = rng.choice(('teacher', 'builder', 'farmer', 'sailor'))
        item = rng.choice(('books', 'seeds', 'maps', 'tools'))
        place = rng.choice(('garden', 'harbor', 'library', 'market'))
        if partition == 'input_composition':
            body = f'A {actor} brings {a} {item} to the {place}, gives away {b}, then receives {d}. Describe the remaining collection.'
        elif partition == 'interchange_composition':
            body = f'The {place} has {a} {item}. A {actor} takes {b} to another place and returns with {d}. Explain both transfers.'
        elif partition == 'adversarial':
            body = f'The note says "there are {b} {item}"; the inventory lists {a}. A {actor} repeats the note at the {place}. Separate the note from the inventory.'
        else:
            body = f'A {actor} carries {a} {item} into the {place}. Describe the delivery.'
        values.append({'id': tag, 'input_text': f'Record {tag}. {body}'})
    return values


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def encode(tokenizer, rows, seen_text, seen_tokens, limit=96):
    encoded = []
    for row in rows:
        h = text_hash(row['input_text'])
        ids = tokenizer(row['input_text'], add_special_tokens=True)['input_ids']
        token_hash = c.digest([int(x) for x in ids])
        if h in seen_text or token_hash in seen_tokens:
            raise ValueError('fixture overlaps frozen predecessor or another partition')
        if not 2 <= len(ids) <= limit:
            raise ValueError('tokenized fixture exceeds frozen bounds; no truncation/retry')
        seen_text.add(h)
        seen_tokens.add(token_hash)
        encoded.append({'id': row['id'], 'ids': [int(x) for x in ids], 'text_sha256': h})
    return encoded
