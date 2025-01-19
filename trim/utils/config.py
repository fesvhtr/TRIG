
gpt_logit_system_msg = '''
You are an evaluation assistant, I will give an AI generated image and a description (i.e. prompt), I need you to evaluate the performance of this generated image on a specific dimension based on this original description and evaluation criteria.
I will give you the definition of this dimension and the criteria for evaluation. You just need to evaluate the performance of this image on this dimension.
The information about the dimension is as follows:
{}
You should evaluate the image by a scale from: excellent, good, medium, bad, terrible.
'''

gpt_logit_dimension_msg = {
    'IQ-R': '',
    'IQ-O': '',
    'IQ-A': '',
    'TA-C': '',
    'TA-R': '',
    'TA-S': '',
    'D-M': '',
    'D-K': '',
    'D-A': '',
    'R-T': '',
    'R-B': '',
    'R-E': '',
}

default_answer_template = 'Yes'
