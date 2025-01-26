
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

default_answer = 'Yes'
default_answer_set = {'excellent': 1.0, 'good': 0.5, 'medium': 0, 'bad': -0.5, 'terrible': -1.0}
