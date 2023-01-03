Source codes of KAGS

Requirements:
- Python 3.6
- pytorch = 1.3.0
- Other python packages: nltk, pycocotools, pyyaml, easydict, datasets

Data preparation:
For bottom-up feature, you can refer the instructions in https://github.com/peteanderson80/bottom-up-attention.
For fc or conv features, we directly download from https://vist-arel.s3.amazonaws.com/resnet_features.zip.
For commonsense knowledge, we utilize the ConceptNet API (https://github.com/commonsense/conceptnet5/wiki/API) for each concept word.

Training:
python train.py

Testing:
python test.py

Note that our experiments are conducted on a single 32GB TESLA V100 GPU.

Tengpeng Li, Hanli Wang, Bin He, Chang Wen Chen, Knowledge-enriched Attention Network with Group-wise Semantic for Visual Storytelling, 
IEEE Transactions on Pattern Analysis and Machine Intelligence, accepted, 2023.