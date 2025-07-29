import numpy as np
def generate_sequence_shifted_second_half(y0, diff_candidates=None):
    if diff_candidates is None:
        diff_candidates = [i for i in range(-9, 10) if i != 0]

    attempts = 0
    while True:
        attempts += 1
        if attempts > 10000:
            raise RuntimeError("Too many failed attempts; may need to relax constraints.")

        # Step 1: Generate 15 unique diffs
        diffs = np.random.choice(diff_candidates, size=15, replace=False)

        # Step 2: First half from y0
        relative_first = np.insert(np.cumsum(diffs), 0, 0)
        first_half = y0 + relative_first
        if not np.all((first_half >= 1) & (first_half <= 999)):
            continue
        if len(set(first_half)) != 16:
            continue

        # Step 3: Reuse same diffs for second half
        relative_second = np.insert(np.cumsum(diffs), 0, 0)
        min_rel = np.min(relative_second)
        bias = np.max(first_half) + 1 - min_rel
        second_half = bias + relative_second
        if not np.all((second_half >= 1) & (second_half <= 999)):
            continue
        if len(set(second_half)) != 16:
            continue

        # Step 4: Full sequence and vertical flip
        full_sequence = np.concatenate([first_half, second_half])
        if len(set(full_sequence)) != 32:
            continue

        min_y, max_y = np.min(full_sequence), np.max(full_sequence)
        flipped = (min_y + max_y) - full_sequence
        if not np.all((flipped >= 1) & (flipped <= 999)):
            continue
        if len(set(flipped)) != 32:
            continue

        return tuple(full_sequence.tolist()), tuple(flipped.tolist())  # use tuples for hashing

def generate_vertical_flip_pairs(num_pairs=10000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    seen = set()
    originals, flipped = [], []

    while len(originals) < num_pairs:
        y0 = np.random.randint(301, 600)  # ensure y0 > 300
        seq, seq_flip = generate_sequence_shifted_second_half(y0)

        if seq in seen:
            continue
        seen.add(seq)
        originals.append(seq)
        flipped.append(seq_flip)

    return np.array(originals), np.array(flipped)

# Run and save
original_dataset, _ = generate_vertical_flip_pairs(num_pairs=10000, seed=42)
dataset_finalsame = np.zeros((2, 10000, 32))
dataset_finalsame[0, :, :] = original_dataset


