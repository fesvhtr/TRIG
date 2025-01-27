# TRIG
Trade-offs and Relationships in IMage Generation



## Setup
### Installation
```bash
conda create -n trig python=3.10 -y
conda activate trig
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Inference
Text-to-Image Generation
```bash
python text_to_image.py --model OmniGen
```


