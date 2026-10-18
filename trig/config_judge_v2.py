"""Judge prompts, version 2 (2026 refresh). Not used by the legacy pipeline; selected with
protocol `v2` in scripts/diag/judge_protocols.py so it can be compared with the paper's prompts.

Design: one anchored five-level scale for all dimensions; each dimension states what it asks,
what to check, and what does NOT count (to decouple it from other dimensions); image-quality
dimensions are judged from the image alone; realism means plausibility within the intended style.
"""

SCALE = (
    "Rate on this scale and answer with exactly one word:\n"
    "excellent - nothing to fault on this dimension.\n"
    "good - minor flaws that do not affect the main content.\n"
    "medium - noticeable problems, but the dimension is basically met.\n"
    "bad - major problems on this dimension.\n"
    "terrible - the dimension is essentially not met."
)

SYSTEM_IMAGE_ONLY = (
    "You are a strict evaluator of AI-generated images. You will see one image and must judge ONE dimension only. "
    "You are deliberately not shown the text prompt: judge purely from what is visible. "
    "Qualities outside this dimension must not change your rating.\n\n"
    "Dimension: {name}\n{criteria}\n\n" + SCALE +
    "\nYour answer must be exactly one of: excellent, good, medium, bad, terrible."
)

SYSTEM_WITH_PROMPT = (
    "You are a strict evaluator of AI-generated images. You will see the text prompt used to generate an image and the image itself, "
    "and must judge ONE dimension only. First identify which parts of the prompt concern this dimension; then check only those against the image. "
    "Qualities outside this dimension (overall beauty, realism, unrelated details) must not change your rating.\n\n"
    "Dimension: {name}\n{criteria}\n\n" + SCALE +
    "\nYour answer must be exactly one of: excellent, good, medium, bad, terrible."
)

# dims judged from the image alone (no prompt shown)
IMAGE_ONLY = {"IQ-R", "IQ-O", "IQ-A"}

DIMENSIONS = {
    "IQ-R": dict(name="Realism (plausibility within the intended style)", criteria=(
        "Ask: within the visual style the image is clearly rendered in (photo, painting, 3D, cartoon ...), is everything physically and structurally coherent?\n"
        "Check: anatomy and object structure, proportions, perspective and depth, lighting and shadow consistency, material behaviour, contact and support between objects, "
        "no melted, duplicated, fused or floating parts, no garbled text-like artifacts.\n"
        "Does NOT count: whether the style is photographic (a coherent watercolour can be excellent), whether the content is imaginative, whether it is beautiful, whether it matches any prompt.")),
    "IQ-O": dict(name="Originality (departure from the default rendering)", criteria=(
        "Ask: compared with the most typical, generic way this kind of scene is usually depicted, does the image show a distinct creative choice?\n"
        "Check: unusual but purposeful composition, viewpoint, colour or lighting idea, element combination, visual metaphor or interpretation; evidence of a specific idea rather than a stock-image default.\n"
        "Does NOT count: technical quality, beauty, realism, prompt adherence. Randomness or errors are not originality: a strange artifact is a flaw, not a creative choice.")),
    "IQ-A": dict(name="Aesthetics (visual appeal of the image as a picture)", criteria=(
        "Ask: judged as a picture, how visually appealing and well-composed is it?\n"
        "Check: composition and balance, colour harmony and contrast, use of light, focus and clarity of the subject, overall finish and visual impact.\n"
        "Does NOT count: whether it follows any prompt, whether it is realistic or original, whether the content is pleasant or unpleasant as subject matter.")),
    "TA-C": dict(name="Content alignment (requested elements are present and correct)", criteria=(
        "Ask: are the objects, subjects and attributes requested in the prompt present and correct?\n"
        "Check: each required entity exists; counts; attributes such as colour, size, material, clothing, state; nothing requested is missing or replaced; no prominent unrequested element contradicts the prompt.\n"
        "Does NOT count: spatial or logical relations between elements (judged under relation alignment), artistic style, image quality, beauty.")),
    "TA-R": dict(name="Relation alignment (relations between elements are as requested)", criteria=(
        "Ask: are the relationships stated in the prompt realised correctly?\n"
        "Check: spatial relations (left/right, on, under, behind, inside, facing), interactions and actions between entities (holding, chasing, looking at), comparative relations (larger than, in front of), ordering and grouping.\n"
        "Does NOT count: whether every object is present (judged under content alignment), style, image quality. If a relation cannot be judged because an entity is missing, rate the relation as not met.")),
    "TA-S": dict(name="Style alignment (visual style matches the prompt)", criteria=(
        "Ask: does the image render in the artistic or visual style the prompt specifies?\n"
        "Check: medium (oil, watercolour, pixel art, photograph, 3D render ...), named artist or movement characteristics, era or genre look, colour palette and mood words, rendering technique; the style should be applied consistently across the whole image.\n"
        "Does NOT count: whether the depicted objects are correct, whether the image is beautiful, whether the style itself is realistic.")),
    "D-K": dict(name="Knowledge (specialised content is depicted factually)", criteria=(
        "Ask: the prompt relies on specialised, technical, scientific, historical or cultural knowledge. Is that knowledge depicted correctly?\n"
        "Check: identify the knowledge claims in the prompt (how a mechanism works, what a species, artifact, symbol, uniform, instrument, landmark or process actually looks like, correct structure and parts); verify each against the image; wrong facts, impossible mechanisms, mixed-up referents and invented details count against the image.\n"
        "Does NOT count: generic prompt adherence that needs no special knowledge, beauty, style, realism of rendering.")),
}

DIMS = tuple(DIMENSIONS)                       # the 7 benchmark dimensions (2026 refresh)
EXCLUDED_DIMS = ("R-T", "R-B", "D-A")          # paper dimensions no longer evaluated or trained on


def pair_allowed(pair, exclude=EXCLUDED_DIMS):
    """pair: "A_B" or (A, B)."""
    dims = pair.split("_") if isinstance(pair, str) else pair
    return not any(d in exclude for d in dims)


def system_message(dim):
    d = DIMENSIONS[dim]
    tmpl = SYSTEM_IMAGE_ONLY if dim in IMAGE_ONLY else SYSTEM_WITH_PROMPT
    return tmpl.format(name=d["name"], criteria=d["criteria"])


def user_text(dim, prompt):
    if dim in IMAGE_ONLY:
        return "Rate this image on the dimension above."
    return f"Prompt used to generate this image:\n{prompt}\n\nRate the image on the dimension above."
