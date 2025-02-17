# replace with your own API key
API_KEY = "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"

# system msg and dim prompt for gpt logit metric, modify the scales to change the scoring granularity
gpt_logit_system_msg = '''
You are an evaluation assistant, I will give an AI generated image and a description (i.e. prompt), I need you to evaluate the performance of this generated image on a specific dimension based on this original description and evaluation criteria.
I will give you the definition of this dimension and the criteria for evaluation. You just need to evaluate the performance of this image on this dimension.
The information about the dimension is as follows:
{}
You should evaluate the image by a scale from: excellent, good, medium, bad, terrible. You must give me one of these first words as your evaluation.
'''

gpt_logit_dimension_msg = {
    'IQ-R': 'Realism: Similarity between the generated images and those in the real world.',
    'IQ-O': 'Originality: Novelty and uniqueness in the generated images.',
    'IQ-A': '',
    'TA-C': 'Alignment of the image’s main objects and scenes with those specified in the prompt',
    'TA-R': 'Relation Alignment: Alignment of the image’s spatial and semantic logical relationships between human and objects with those specified in the prompt',
    'TA-S': '',
    'D-M': '',
    'D-K': '',
    'D-A': '',
    'R-T': '',
    'R-B': '',
    'R-E': '',
}

## for i2i
gpt_logit_system_msg_i2i = '''
You are an evaluation assistant, I will give an AI generated image and a description (i.e. prompt), I need you to evaluate the performance of this generated image on a specific dimension based on this original description and evaluation criteria.
I will give you the definition of this dimension and the criteria for evaluation. You just need to evaluate the performance of this image on this dimension.
The information about the dimension is as follows:
{}
You should evaluate the image by a scale from: excellent, good, medium, bad, terrible. You must give me one of these first words as your evaluation.
'''

# vqascore from https://github.com/linzhiqiu/t2v_metrics
default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = 'Yes'

# for image generation
DIM_DICT = {
    "IQ-R": ["IQ-O", "IQ-A", "TA-C", "TA-R", "TA-S", "D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "IQ-O": ["IQ-A", "TA-C", "TA-R", "TA-S", "D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "IQ-A": ["TA-C", "TA-R", "TA-S", "D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "TA-C": ["TA-R", "TA-S", "D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "TA-R": ["TA-S", "D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "TA-S": ["D-M", "D-K", "D-A", "R-T", "R-B", "R-E"],
    "D-M":  ["D-K", "D-A", "R-T", "R-B", "R-E"],
    "D-K":  ["D-A", "R-T", "R-B", "R-E"],
    "D-A":  ["R-T", "R-B", "R-E"],
    "R-T":  ["R-B", "R-E"],
    "R-B":  ["R-E"]
}

DIM_DICT_WITHOUT_M_E = {
    "IQ-R": ["IQ-O", "IQ-A", "TA-C", "TA-R", "TA-S", "D-K", "D-A", "R-T", "R-B"],
    "IQ-O": ["IQ-A", "TA-C", "TA-R", "TA-S", "D-K", "D-A", "R-T", "R-B"],
    "IQ-A": ["TA-C", "TA-R", "TA-S", "D-K", "D-A", "R-T", "R-B"],
    "TA-C": ["TA-R", "TA-S",  "D-K", "D-A", "R-T", "R-B"],
    "TA-R": ["TA-S", "D-K", "D-A", "R-T", "R-B"],
    "TA-S": ["D-K", "D-A", "R-T", "R-B"],
    "D-K":  ["D-A", "R-T", "R-B"],
    "D-A":  ["R-T", "R-B"],
    "R-T":  ["R-B"],
}

OD_NEGATIVE_PROMPT = "monochrome, greyscale, low-res, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"

