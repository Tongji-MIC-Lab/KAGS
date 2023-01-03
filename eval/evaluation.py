from __future__ import division,print_function
from vist_eval.album_eval import AlbumEvaluator
import json
import sys
reload(sys)
sys.setdefaultencoding('utf8')
eval=AlbumEvaluator()
import numpy as np
reference = json.load(open('./test_reference.json'))
root='./predict_test_21'
txt_root='./scores.txt'
##
f1=open(txt_root, 'a+')
f1.close()
## 
predictions = {}
id=0
with open(root) as f:
            for line in f:
                vid, seq = line.strip().split('\t')
                id=id+1
                if vid not in predictions:
                    seq=seq[2:]
                    predictions[vid] = [seq]

eval.evaluate(reference, predictions, txt_root)
