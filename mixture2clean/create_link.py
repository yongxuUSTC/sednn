import numpy as np
import random
import csv
import os
import gzip

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


te_speech_dir = "/vol/vssp/msos/qk/workspaces/speech_enhancement/timit_wavs/test"
subte_speech_dir = "/vol/vssp/msos/qk/workspaces/speech_enhancement/timit_wavs/subtest"
create_folder(subte_speech_dir)

csv_gz_path = os.path.join("splits", "timit_subtest.csv.gz")
with gzip.open(csv_gz_path, 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    lis = list(reader)
    
for li in lis:
    na = li[0]
    cmd = "ln -s %s/%s %s/%s" % (te_speech_dir, na, subte_speech_dir, na)
    os.system(cmd)