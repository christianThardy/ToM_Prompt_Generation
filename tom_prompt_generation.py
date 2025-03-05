import copy
import random
import warnings

import numpy as np
import torch as t
from transformers import AutoTokenizer


NAMES = [
    "Aaron",
    "Adam",
    "Alan",
    "Alex",
    "Alice",
    "Amy",
    "Anderson",
    "Andre",
    "Andrew",
    "Andy",
    "Anna",
    "Anthony",
    "Arthur",
    "Austin",
    "Blake",
    "Brandon",
    "Brian",
    "Carter",
    "Charles",
    "Charlie",
    "Christian",
    "Christopher",
    "Clark",
    "Cole",
    "Collins",
    "Connor",
    "Crew",
    "Crystal",
    "Daniel",
    "David",
    "Dean",
    "Edward",
    "Elizabeth",
    "Emily",
    "Eric",
    "Eva",
    "Ford",
    "Frank",
    "George",
    "Georgia",
    "Graham",
    "Grant",
    "Henry",
    "Ian",
    "Jack",
    "Jacob",
    "Jake",
    "James",
    "Jamie",
    "Jane",
    "Jason",
    "Jay",
    "Jennifer",
    "Jeremy",
    "Jessica",
    "John",
    "Jonathan",
    "Jordan",
    "Joseph",
    "Joshua",
    "Justin",
    "Kate",
    "Kelly",
    "Kevin",
    "Kyle",
    "Laura",
    "Leon",
    "Lewis",
    "Lisa",
    "Louis",
    "Luke",
    "Madison",
    "Marco",
    "Marcus",
    "Maria",
    "Mark",
    "Martin",
    "Mary",
    "Matthew",
    "Max",
    "Michael",
    "Michelle",
    "Morgan",
    "Patrick",
    "Paul",
    "Peter",
    "Prince",
    "Rachel",
    "Richard",
    "River",
    "Robert",
    "Roman",
    "Rose",
    "Ruby",
    "Russell",
    "Ryan",
    "Sarah",
    "Scott",
    "Sean",
    "Simon",
    "Stephen",
    "Steven",
    "Sullivan",
    "Taylor",
    "Thomas",
    "Tyler",
    "Victoria",
    "Warren",
    "William",
]

ABC_TOM_TEMPLATES = [
    # Original template
    "In the [PLACE] there are [A], [B], [C], a [OBJECT1], a [OBJECT2], and a [OBJECT3]. [A] takes the [OBJECT1] and puts it on the [OBJECT2]. [A] leaves the [PLACE]. While [A] is away, [B] and [C] move the [OBJECT1] to the [OBJECT3]. [A] returns and thinks the [OBJECT1] is on the [TARGET]",
    # New variations
    "Friends [A], [B] and [C] were in the [PLACE]. After [A] placed the [OBJECT1] on the [OBJECT2], they went to [LOCATION]. Together, [B] and [C] decided to move the [OBJECT1] to the [OBJECT3]. When [A] came back, they thought the [OBJECT1] was on the [TARGET]",
    "When [A], [B] and [C] were spending time in the [PLACE], [A] put their [OBJECT1] on the [OBJECT2]. During [A]'s trip to [LOCATION], [B] and [C] relocated the [OBJECT1] to the [OBJECT3]. Upon return, [A] assumed the [OBJECT1] was still on the [TARGET]",
    "At the [PLACE], [A] showed [B] and [C] how they organized their [OBJECT1] by placing it on the [OBJECT2]. While [A] was away at [LOCATION], [B] and [C] transferred the [OBJECT1] to the [OBJECT3]. Later, [A] believed the [OBJECT1] would be on the [TARGET]",
    "Yesterday, [A], [B] and [C] met at the [PLACE]. First thing, [A] set their [OBJECT1] on the [OBJECT2]. During [A]'s absence at [LOCATION], [B] and [C] shifted the [OBJECT1] to the [OBJECT3]. Back at the [PLACE], [A] expected to find the [OBJECT1] on the [TARGET]",
    "The morning started with [A], [B] and [C] gathering in the [PLACE]. Once [A] positioned the [OBJECT1] on the [OBJECT2], they left for [LOCATION]. [B] and [C] worked together to transport the [OBJECT1] to the [OBJECT3]. Returning later, [A] was certain the [OBJECT1] would be on the [TARGET]"
]

BAC_TOM_TEMPLATES = [
    template.replace("[B]", "[A]", 1).replace("[A]", "[B]", 1) for template in ABC_TOM_TEMPLATES
]

# Basic structure of belief task
BABA_TOM_TEMPLATES = [
    "In the [PLACE] there are [B] and [A], a [OBJECT1], a [OBJECT2], and a [OBJECT3]. [A] takes the [OBJECT1] and puts it on the [OBJECT2]. [A] leaves for [LOCATION]. [B] moves the [OBJECT1] to the [OBJECT3]. [A] returns and thinks the [OBJECT1] is on the [TARGET]",
    "The [PLACE] contains [B] and [A], with a [OBJECT1], a [OBJECT2], and a [OBJECT3]. [A] places the [OBJECT1] on the [OBJECT2]. [A] goes to [LOCATION]. [B] moves the [OBJECT1] to the [OBJECT3]. [A] comes back thinking the [OBJECT1] is on the [TARGET]",
]

# Additional temporal markers, more detailed descriptions, extra content about actions
BABA_LONG_TOM_TEMPLATES = [
    "Early in the morning, in the [PLACE] there are [B] and [A], a [OBJECT1], a [OBJECT2], and a [OBJECT3]. After careful consideration, [A] takes the [OBJECT1] and puts it on the [OBJECT2]. [A] leaves for [LOCATION]. While [A] is away for several hours, [B] moves the [OBJECT1] to the [OBJECT3]. When [A] finally returns, they think the [OBJECT1] is on the [TARGET]",
]

# Belief state mentioned later in sequence, additional details about observation with emphasis on time passing, more explicit return and realization
BABA_LATE_TOM_TEMPLATES = [
    "In the [PLACE] there are [B] and [A], a [OBJECT1], a [OBJECT2], and a [OBJECT3]. [A] takes the [OBJECT1] and puts it on the [OBJECT2]. [A] leaves for [LOCATION]. After examining the situation, [B] moves the [OBJECT1] to the [OBJECT3]. Much later when [A] returns and surveys the [PLACE], they think the [OBJECT1] is on the [TARGET]",
]

# Belief state mentioned earlier in sequence, more condensed structure using conjunctions to link events, maintains false belief focus
BABA_EARLY_TOM_TEMPLATES = [
    "In the [PLACE] with [B] and [A], there's a [OBJECT1], a [OBJECT2], and a [OBJECT3]. [A] puts the [OBJECT1] on the [OBJECT2] and leaves for [LOCATION], while [B] moves the [OBJECT1] to the [OBJECT3], though [A] still thinks the [OBJECT1] is on the [TARGET]",
]

ABBA_TOM_TEMPLATES = BABA_TOM_TEMPLATES[:]
ABBA_LATE_TOM_TEMPLATES = BABA_LATE_TOM_TEMPLATES[:]
ABBA_EARLY_TOM_TEMPLATES = BABA_EARLY_TOM_TEMPLATES[:]

for TEMPLATES in [ABBA_TOM_TEMPLATES, ABBA_LATE_TOM_TEMPLATES, ABBA_EARLY_TOM_TEMPLATES]:
    for i in range(len(TEMPLATES)):
        first_clause = True
        for j in range(1, len(TEMPLATES[i]) - 1):
            if TEMPLATES[i][j - 1 : j + 2] == "[B]" and first_clause:
                TEMPLATES[i] = TEMPLATES[i][:j] + "A" + TEMPLATES[i][j + 1 :]
            elif TEMPLATES[i][j - 1 : j + 2] == "[A]" and first_clause:
                first_clause = False
                TEMPLATES[i] = TEMPLATES[i][:j] + "B" + TEMPLATES[i][j + 1 :]

# Define ToM-specific nouns
TOM_PLACES = [
    "room",
    "office",
    "kitchen",
    "garage",
    "basement",
    "attic",
    "classroom",
    "laboratory",
]

TOM_LOCATIONS = [
    "store",
    "school",
    "work",
    "market",
    "library",
    "park",
    "gym",
]

# Modified ToM object definitions to better reflect semantic categories
# Semanticity of objects wrt to the template dataset sentence matters alot
TOM_OBJECTS = {
    "OBJECT1": ["cat"],  # Fixed to just "book" as the tracked object
    "OBJECT2": [          # Initial locations (surfaces/furniture)
        "basket",
        "box",
        "shelf",
        "table",
        "chair",
        "desk",
        "cabinet",
        "counter"
    ],
    "OBJECT3": [          # Final locations (same category as OBJECT2)
        "basket",
        "box",
        "shelf",
        "table",
        "chair",
        "desk",
        "cabinet",
        "counter"
    ]
}

TOM_OBJECTS_1 = [
        "basket",
        "box",
        "shelf",
        "table",
        "chair",
        "desk",
        "cabinet",
        "counter"
]

def gen_tom_prompt_uniform(
    templates: list[str],
    names: list[str],
    nouns_dict: dict,
    N: int,
    symmetric: bool = False,
    prefixes=None,
    abc: bool = False
) -> list[dict]:
    """Generate ToM prompts uniformly"""
    nb_gen = 0
    tom_prompts = []

    while nb_gen < N:
        temp = random.choice(templates)
        temp_id = templates.index(temp)

        # Select names
        if abc:
            name_1, name_2, name_3 = random.sample(names, 3)
        else:
            name_1, name_2 = random.sample(names, 2)

        # Select objects ensuring no duplicates for surfaces
        object1 = random.choice(TOM_OBJECTS["OBJECT1"])
        object2 = random.choice(TOM_OBJECTS["OBJECT2"])
        object3 = random.choice([x for x in TOM_OBJECTS["OBJECT3"] if x != object2])

        # Build prompt
        prompt = temp
        prompt = prompt.replace("[PLACE]", random.choice(TOM_PLACES))
        prompt = prompt.replace("[LOCATION]", random.choice(TOM_LOCATIONS))
        prompt = prompt.replace("[OBJECT1]", object1)
        prompt = prompt.replace("[OBJECT2]", object2)
        prompt = prompt.replace("[OBJECT3]", object3)
        prompt = prompt.replace("[TARGET]", object2)  # Target is always initial location

        if prefixes is not None:
            L = random.randint(30, 40)
            pref = ".".join(random.choice(prefixes).split(".")[:L])
            pref += "<|endoftext|>"
        else:
            pref = ""

        prompt1 = prompt.replace("[A]", name_1).replace("[B]", name_2)

        if abc:
            prompt1 = prompt1.replace("[C]", name_3)

        prompt1 = pref + prompt1

        # Create prompt dictionary with consistent key names
        tom_prompt = {
            "text": prompt1,
            "A": name_1,  # belief holder
            "B": name_2,  # state changer
            "TEMPLATE_IDX": temp_id,
            "OBJECT1": object1,
            "OBJECT2": object2,
            "OBJECT3": object3,
            "initial_loc": object2,
            "final_loc": object3
        }

        if abc:
            tom_prompt["C"] = name_3

        tom_prompts.append(tom_prompt)
        nb_gen += 1

        # Generate symmetric version if requested
        if symmetric and nb_gen < N:
            prompt2 = prompt.replace("[A]", name_2).replace("[B]", name_1)
            prompt2 = pref + prompt2

            tom_prompts.append({
                "text": prompt2,
                "A": name_2,
                "B": name_1,
                "TEMPLATE_IDX": temp_id,
                "OBJECT1": object1,
                "OBJECT2": object2,
                "OBJECT3": object3,
                "initial_loc": object2,
                "final_loc": object3
            })

            nb_gen += 1

    return tom_prompts


def gen_tom_flipped_prompts(
    prompts: list[dict],
    templates_by_prompt: list[str],
    seed: int
) -> list[dict]:
    """
    Generate flipped versions of prompts by swapping locations while keeping the fixed object,
    and tracking of token sequences.
    """
    random.seed(seed)
    np.random.seed(seed)

    new_prompts = []

    for prompt in prompts:
        new_prompt = copy.deepcopy(prompt)
        text = prompt["text"]

        # Keep the same OBJECT1 (book)
        new_obj1 = prompt["OBJECT1"]

        # Get original locations
        orig_obj2 = prompt["OBJECT2"]
        orig_obj3 = prompt["OBJECT3"]

        # Try multiple times to find valid new locations
        max_attempts = 10
        valid_locations_found = False

        for _ in range(max_attempts):
            # Select new locations ensuring no duplicates
            available_obj2 = [x for x in TOM_OBJECTS["OBJECT2"]
                            if x != orig_obj2 and x != orig_obj3]
            if not available_obj2:
                continue
            new_obj2 = random.choice(available_obj2)

            available_obj3 = [x for x in TOM_OBJECTS["OBJECT3"]
                            if x != orig_obj3 and x != new_obj2]
            if not available_obj3:
                continue
            new_obj3 = random.choice(available_obj3)

            # Create new text with replacements
            new_text = text

            # Replace locations in reverse order (longer words first to avoid partial matches)
            locations = [(orig_obj2, new_obj2), (orig_obj3, new_obj3)]
            locations.sort(key=lambda x: len(x[0]), reverse=True)

            for old_loc, new_loc in locations:
                new_text = new_text.replace(f" {old_loc}", f" {new_loc}")
                new_text = new_text.replace(f"{old_loc},", f"{new_loc},")
                new_text = new_text.replace(f"{old_loc}.", f"{new_loc}.")

            # Verify the replacement worked correctly
            expected_count_obj2 = text.count(orig_obj2)
            expected_count_obj3 = text.count(orig_obj3)

            actual_count_obj2 = new_text.count(new_obj2)
            actual_count_obj3 = new_text.count(new_obj3)

            if (expected_count_obj2 == actual_count_obj2 and
                expected_count_obj3 == actual_count_obj3):
                valid_locations_found = True
                break

        if not valid_locations_found:
            raise ValueError(f"Could not find valid location replacements after {max_attempts} attempts")

        # Update prompt dictionary
        new_prompt["text"] = new_text
        new_prompt["OBJECT1"] = new_obj1  # Same as original
        new_prompt["OBJECT2"] = new_obj2
        new_prompt["OBJECT3"] = new_obj3
        new_prompt["initial_loc"] = new_obj2
        new_prompt["final_loc"] = new_obj3

        new_prompts.append(new_prompt)

    return new_prompts


def validate_tom_prompts(prompts, tokenizer):
    """Validate ToM prompts with pattern matching and location tracking"""
    for i, prompt in enumerate(prompts):
        text = prompt["text"]
        initial_loc = prompt["OBJECT2"]
        final_loc = prompt["OBJECT3"]
        movable_obj = prompt["OBJECT1"]

        # Check for unique objects
        if len({movable_obj, initial_loc, final_loc}) != 3:
            raise ValueError(f"Prompt {i} contains duplicate objects: {movable_obj}, {initial_loc}, {final_loc}")

        # Verify object mentions
        if text.count(movable_obj) < 3:
            raise ValueError(f"Prompt {i} doesn't mention '{movable_obj}' enough times")

        # Check location mentions with better context
        initial_patterns = [
            f" {initial_loc},",
            f" {initial_loc}.",
            f"the {initial_loc}"
        ]

        final_patterns = [
            f" {final_loc},",
            f" {final_loc}.",
            f"the {final_loc}"
        ]

        initial_mentions = sum(text.count(pat) for pat in initial_patterns)
        final_mentions = sum(text.count(pat) for pat in final_patterns)

        if initial_mentions < 2:
            raise ValueError(f"Prompt {i}: Initial location '{initial_loc}' not mentioned enough times")
        if final_mentions < 1:
            raise ValueError(f"Prompt {i}: Final location '{final_loc}' not mentioned enough times")

        # Verify basic structure
        required_phrases = [
            " takes ", " puts ", " moves ", " thinks ",
            " leaves ", " returns "
        ]

        for phrase in required_phrases:
            if phrase not in text:
                raise ValueError(f"Prompt {i} missing required phrase: '{phrase}'")

    return True


# Updated places and locations for more variety
TOM_PLACES.extend([
    "study", "library", "workshop", "studio",
    "hall", "lobby", "lounge", "office space"
])

TOM_LOCATIONS.extend([
    "cafe", "meeting", "lunch", "break room",
    "conference", "appointment", "garden"
])


def gen_tom_flipped_prompts_objects(
    prompts: list[dict],
    templates_by_prompt: list[str],
    seed: int
) -> list[dict]:
    """
    Generate flipped versions of prompts by swapping objects instead of names.
    Maintains the same agents but changes the physical arrangement of objects.
    """
    random.seed(seed)
    np.random.seed(seed)

    new_prompts = []

    for prompt, template in zip(prompts, templates_by_prompt):
        new_prompt = copy.deepcopy(prompt)

        # Get current objects
        curr_obj1 = prompt["OBJECT1"]  # The movable object (e.g., pen, book)
        curr_obj2 = prompt["OBJECT2"]  # Initial location
        curr_obj3 = prompt["OBJECT3"]  # Final location

        # Select new objects ensuring no duplicates
        available_obj1 = [x for x in TOM_OBJECTS["OBJECT1"] if x != curr_obj1]
        new_obj1 = random.choice(available_obj1)

        available_obj2 = [x for x in TOM_OBJECTS["OBJECT2"] if x != curr_obj2]
        new_obj2 = random.choice(available_obj2)

        available_obj3 = [x for x in TOM_OBJECTS["OBJECT3"] if x != curr_obj3 and x != new_obj2]
        new_obj3 = random.choice(available_obj3)

        # Update prompt text
        new_text = prompt["text"]
        new_text = new_text.replace(curr_obj1, new_obj1)
        new_text = new_text.replace(curr_obj2, new_obj2)
        new_text = new_text.replace(curr_obj3, new_obj3)

        # Update prompt dictionary
        new_prompt["text"] = new_text
        new_prompt["OBJECT1"] = new_obj1
        new_prompt["OBJECT2"] = new_obj2
        new_prompt["OBJECT3"] = new_obj3
        new_prompt["initial_loc"] = new_obj2
        new_prompt["final_loc"] = new_obj3

        new_prompts.append(new_prompt)

    return new_prompts


def get_location_idxs(prompts, tokenizer, prepend_bos=False):
    """Get indices for locations in ToM task with improved token handling"""
    location_idx_dict = {
        "believed_loc": [],  # Where agent thinks object is (initial location)
        "actual_loc": [],    # Where object actually is (final location)
        "object": []         # The object being moved
    }

    def find_last_token_index(text, target_word):
        """Find the last occurrence of a word in tokenized text"""
        # Get the full text tokens
        tokens = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()

        # Get target word tokens with space prefix
        target_tokens = tokenizer(f" {target_word}", add_special_tokens=False)["input_ids"]
        target_token = target_tokens[0]  # We'll look for the first token

        # Find last occurrence
        last_idx = -1
        for i, token in enumerate(tokens):
            if token == target_token:
                last_idx = i

        if last_idx == -1:
            raise ValueError(f"Token for word '{target_word}' not found in text")

        return last_idx


    def find_first_token_index(text, target_word):
        """Find the first occurrence of a word in tokenized text"""
        tokens = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
        target_tokens = tokenizer(f" {target_word}", add_special_tokens=False)["input_ids"]
        target_token = target_tokens[0]

        for i, token in enumerate(tokens):
            if token == target_token:
                return i

        raise ValueError(f"Token for word '{target_word}' not found in text")

    for prompt in prompts:
        text = prompt["text"]

        try:
            # Find indices with better token handling
            believed_idx = find_last_token_index(text, prompt["OBJECT2"])
            actual_idx = find_last_token_index(text, prompt["OBJECT3"])
            object_idx = find_first_token_index(text, prompt["OBJECT1"])

            # Add debugging info
            if len(location_idx_dict["believed_loc"]) < 5:  # Only show first 5 for brevity
                print(f"\nProcessing prompt: {text}")
                print(f"Believed location '{prompt['OBJECT2']}' found at index {believed_idx}")
                print(f"Actual location '{prompt['OBJECT3']}' found at index {actual_idx}")
                print(f"Object '{prompt['OBJECT1']}' found at index {object_idx}")

            # Store indices
            location_idx_dict["believed_loc"].append(believed_idx)
            location_idx_dict["actual_loc"].append(actual_idx)
            location_idx_dict["object"].append(object_idx)

        except Exception as e:
            print(f"\nError processing prompt: {text}")
            print(f"Error details: {str(e)}")
            raise

    # Add offset for BOS token if needed
    offset = int(prepend_bos)
    return [
        t.tensor([idx + offset for idx in location_idx_dict[key]])
        for key in ["believed_loc", "actual_loc", "object"]
    ]


def get_object_idxs(prompts, tokenizer):
    """Get indices of the objects and their locations in the text"""
    obj_idxs = []
    initial_loc_idxs = []
    final_loc_idxs = []

    for prompt in prompts:
        toks = tokenizer.tokenize(prompt["text"])

        # Get object indices
        obj = prompt["OBJECT1"]  # e.g., "cat"
        obj_idxs.append(toks.index(tokenizer.tokenize(" " + obj)[0]))

        # Get initial location
        init_loc = prompt["OBJECT2"]  # e.g., "basket"
        initial_loc_idxs.append(toks.index(tokenizer.tokenize(" " + init_loc)[0]))

        # Get final location
        final_loc = prompt["OBJECT3"]  # e.g., "box"
        final_loc_idxs.append(toks.index(tokenizer.tokenize(" " + final_loc)[0]))

    return t.tensor(obj_idxs), t.tensor(initial_loc_idxs), t.tensor(final_loc_idxs)


def get_end_idxs(toks, tokenizer, name_tok_len=1, prepend_bos=False):
    relevant_idx = int(prepend_bos)
    pad_token_id = tokenizer.pad_token_id

    end_idxs_raw = []
    for i in range(toks.shape[0]):
        seq = toks[i]
        # Search for pad token only in tokens starting at index 1 to avoid the BOS token.
        pad_positions = (seq[1:] == pad_token_id).nonzero()
        if pad_positions.numel() == 0:
            end_idx = seq.shape[0]
        else:
            # Add 1 to account for the fact that we searched in seq[1:]
            end_idx = pad_positions[0].item() + 1
        end_idxs_raw.append(end_idx)
    end_idxs = t.tensor(end_idxs_raw)
    end_idxs = end_idxs - 1 - name_tok_len

    for i in range(toks.shape[0]):
        seq = toks[i]
        idx = end_idxs[i].item()
        # Check that the token immediately after the computed index is not zero,
        # and that either the sequence ends right after or the following token is the pad token.
        assert seq[idx + 1] != 0 and (
            toks.shape[1] == idx + 2 or seq[idx + 2] == pad_token_id
        ), (seq, end_idxs[i], seq.shape, "the END idxs aren't properly formatted")
    return end_idxs


def get_belief_state_idx(prompts, tokenizer):
    """Get index where belief state is mentioned"""
    belief_idxs = []

    belief_markers = ["thinks", "believes", "assumes", "expects"]

    for prompt in prompts:
        toks = tokenizer.tokenize(prompt["text"])

        # Find the belief marker word
        idx = None
        for marker in belief_markers:
            try:
                idx = toks.index(tokenizer.tokenize(" " + marker)[0])
                break
            except ValueError:
                continue

        if idx is None:
            raise ValueError(f"No belief marker found in prompt: {prompt['text']}")

        belief_idxs.append(idx)

    return t.tensor(belief_idxs)


def get_tom_idx_dict(tom_prompts, tokenizer, prepend_bos=False, toks=None):
    """Get dictionary of important indices for ToM task"""

    # Get location indices with improved token handling
    believed_loc_idxs, actual_loc_idxs, object_idxs = get_location_idxs(
        tom_prompts,
        tokenizer,
        prepend_bos=prepend_bos
    )

    # Get belief state index
    belief_idxs = get_belief_state_idx(tom_prompts, tokenizer)
    if prepend_bos:
        belief_idxs = belief_idxs + 1

    # Get end indices
    if toks is not None:
        end_idxs = get_end_idxs(
            toks,
            tokenizer,
            name_tok_len=1,
            prepend_bos=prepend_bos,
        )
    else:
        # If no toks provided, calculate from max sequence length
        max_len = max(len(tokenizer.encode(p["text"])) for p in tom_prompts)
        end_idxs = t.tensor([max_len - 1] * len(tom_prompts))

    return {
        "believed_loc": believed_loc_idxs,
        "actual_loc": actual_loc_idxs,
        "object": object_idxs,
        "belief": belief_idxs,
        "end": end_idxs,
        "start": t.zeros_like(end_idxs),
        "believed_loc-1": believed_loc_idxs - 1,
        "believed_loc+1": believed_loc_idxs + 1,
        "actual_loc-1": actual_loc_idxs - 1,
        "actual_loc+1": actual_loc_idxs + 1,
    }


class ToMDataset:
    def __init__(
        self,
        prompt_type: str | list[str],
        N=500,
        tokenizer=None,
        prompts=None,
        symmetric=False,
        prefixes=None,
        nb_templates=None,
        prepend_bos=False,
        manual_word_idx=None,
        has_been_flipped: bool = False,
        seed=0,
        device="cuda",
    ):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.device = device

        # Warning handling
        if not (
            N == 1
            or not prepend_bos
            or (tokenizer is not None and tokenizer.bos_token_id == tokenizer.eos_token_id)
        ):
            warnings.warn("Word indices may be calculated incorrectly with this formatting")

        self.has_been_flipped = has_been_flipped

        # Validate inputs
        assert not (symmetric and prompt_type == "ABC"), "ABC templates cannot be symmetric in ToM task"
        assert (prompts is not None) or (not symmetric) or (N % 2 == 0), f"N must be even for symmetric={symmetric}"

        self.prompt_type = prompt_type

        # Initialize templates first
        if nb_templates is None:
            nb_templates = len(BABA_TOM_TEMPLATES)

        # Template selection based on prompt_type
        if prompt_type == "ABBA":
            self.templates = ABBA_TOM_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "BABA":
            self.templates = BABA_TOM_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "mixed":
            self.templates = (
                BABA_TOM_TEMPLATES[: nb_templates // 2].copy()
                + ABBA_TOM_TEMPLATES[: nb_templates // 2].copy()
            )
            random.shuffle(self.templates)
        elif prompt_type == "ABC":
            self.templates = ABC_TOM_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "BAC":
            self.templates = BAC_TOM_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "ABC mixed":
            self.templates = (
                ABC_TOM_TEMPLATES[: nb_templates // 2].copy()
                + BAC_TOM_TEMPLATES[: nb_templates // 2].copy()
            )
            random.shuffle(self.templates)
        elif isinstance(prompt_type, list):
            self.templates = prompt_type
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        self.prefixes = prefixes

        # Generate or use provided prompts
        if prompts is None:
            self.tom_prompts = gen_tom_prompt_uniform(
                self.templates,
                NAMES,
                nouns_dict={
                    "[PLACE]": TOM_PLACES,
                    "[LOCATION]": TOM_LOCATIONS,
                    "[OBJECT1]": TOM_OBJECTS["OBJECT1"],
                    "[OBJECT2]": TOM_OBJECTS["OBJECT2"],
                    "[OBJECT3]": TOM_OBJECTS["OBJECT3"]
                },
                N=N,
                symmetric=symmetric,
                prefixes=self.prefixes,
                abc=(prompt_type in ["ABC", "ABC mixed", "BAC"]),
            )
        else:
            assert N == len(prompts), f"N ({N}) must match number of prompts ({len(prompts)})"
            self.tom_prompts = prompts

        # Initialize groups
        all_ids = [prompt["TEMPLATE_IDX"] for prompt in self.tom_prompts]
        all_ids_ar = np.array(all_ids)
        self.groups = []
        for id in list(set(all_ids)):
            self.groups.append(np.where(all_ids_ar == id)[0])

        # Store sentences and determine template types
        self.sentences = [prompt["text"] for prompt in self.tom_prompts]

        # Update template determination to use A/B keys
        self.templates_by_prompt = []
        for i in range(N):
            if self.sentences[i].index(self.tom_prompts[i]["A"]) < self.sentences[i].index(
                self.tom_prompts[i]["B"]
            ):
                self.templates_by_prompt.append("ABBA")
            else:
                self.templates_by_prompt.append("BABA")

        # Process texts
        texts = [
            (self.tokenizer.bos_token if prepend_bos else "") + prompt["text"]
            for prompt in self.tom_prompts
        ]
        tokenizer_output = self.tokenizer(texts, padding=True, return_tensors='pt')
        self.toks = tokenizer_output['input_ids'].to(self.device)
        self.attention_mask = tokenizer_output['attention_mask'].to(self.device)


        # Get word indices
        self.word_idx = get_tom_idx_dict(
            self.tom_prompts,
            self.tokenizer,
            prepend_bos=prepend_bos,
            toks=self.toks,
        )

        if manual_word_idx is not None:
            self.word_idx = manual_word_idx

        self.prepend_bos = prepend_bos
        self.N = N

        # Calculate max length
        self.max_len = max(
            [len(self.tokenizer(prompt["text"]).input_ids) for prompt in self.tom_prompts]
        )

        # Store tokenized IDs for believed location and actual location - FIX HERE
        self.believed_location_tokenIDs = [
            self.tokenizer.encode(" " + prompt["OBJECT2"])[1]  # Use index 1 instead of 0
            for prompt in self.tom_prompts
        ]
        self.actual_location_tokenIDs = [
            self.tokenizer.encode(" " + prompt["OBJECT3"])[1]  # Use index 1 instead of 0
            for prompt in self.tom_prompts
        ]

        # For debugging
        print("\nTokenization verification:")
        for i in range(min(5, len(self.tom_prompts))):
            print(f"\nExample {i}:")
            print(f"Prompt: {self.tom_prompts[i]['text']}")
            print(f"Believed location: {self.tom_prompts[i]['OBJECT2']}")
            print(f"Believed token ID: {self.believed_location_tokenIDs[i]}")
            print(f"Decoded: {self.tokenizer.decode([self.believed_location_tokenIDs[i]])}")
            print(f"Actual location: {self.tom_prompts[i]['OBJECT3']}")
            print(f"Actual token ID: {self.actual_location_tokenIDs[i]}")
            print(f"Decoded: {self.tokenizer.decode([self.actual_location_tokenIDs[i]])}")

        # Store tokenized prompts
        self.tokenized_prompts = []
        for i in range(self.N):
            self.tokenized_prompts.append(
                "|".join([self.tokenizer.decode(tok) for tok in self.toks[i]])
            )

        self.device = device
        self.to(device)

    # Add this method to the class
    def __len__(self):
        """Returns the number of samples in the dataset"""
        return self.N

    def to(self, device):
        """Move the dataset tensors to the specified device"""
        self.toks = self.toks.to(device)
        return self

    def gen_flipped_prompts(self, flip=None):
        """
        Generate flipped prompts by only changing locations while keeping the fixed object.
        Ignores the flip argument since we're only changing locations.
        """
        if self.has_been_flipped:
            warnings.warn(
                "This dataset has already been flipped. Apply flips in one step to avoid errors."
            )

        # Use same seed logic
        seed = self.seed + 1

        # Generate flipped prompts with only location changes
        flipped_prompts = gen_tom_flipped_prompts(
            self.tom_prompts,
            self.templates_by_prompt,
            seed
        )

        # Create new dataset with flipped prompts
        flipped_tom_dataset = ToMDataset(
            prompt_type=self.prompt_type,
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=flipped_prompts,
            prefixes=self.prefixes,
            prepend_bos=self.prepend_bos,
            manual_word_idx=self.word_idx,
            has_been_flipped=True,
            seed=seed,
            device=self.device
        )

        return flipped_tom_dataset

    def copy(self):
        """Create a deep copy of the dataset"""
        copy_tom_dataset = ToMDataset(
            prompt_type=self.prompt_type,
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=copy.deepcopy(self.tom_prompts),
            prefixes=copy.deepcopy(self.prefixes) if self.prefixes is not None else None,
            prepend_bos=self.prepend_bos,
            seed=self.seed,
            device=self.device
        )
        return copy_tom_dataset
