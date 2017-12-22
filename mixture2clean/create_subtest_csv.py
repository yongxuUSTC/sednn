import os
import csv
import argparse
import numpy as np
import csv
import gzip

tr_speech_dir = "/vol/vssp/msos/qk/workspaces/speech_enhancement/timit_wavs/train"
te_speech_dir = "/vol/vssp/msos/qk/workspaces/speech_enhancement/timit_wavs/test"

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def create_train_csv():
    # Write out csv. 
    out_path = os.path.join("splits", "timit_train.csv.gz")
    create_folder(os.path.dirname(out_path))
    
    names = os.listdir(tr_speech_dir)
    f = gzip.open(out_path, 'w')
    for na in names:
        f.write(na)
        f.write('\n')
    f.close()
        

def create_subtest_csv():
    names = os.listdir(te_speech_dir)
    
    dict = {}   # key: people, value: list of names
    for na in names:
        people = na.split('_')[2]
        if people in dict.keys():
            dict[people].append(na)
        else:
            dict[people] = [na]
            
    # Selecte one sentence per people. 
    rs = np.random.RandomState(0)
    selected_names = []
    for key in dict.keys():
        selected_na = rs.choice(dict[key], size=1, replace=False)[0]
        selected_names.append(selected_na)
    
    # Write out csv. 
    out_path = os.path.join("splits", "timit_subtest.csv.gz")
    create_folder(os.path.dirname(out_path))
    
    f = gzip.open(out_path, 'w')
    for na in selected_names:
        f.write(na)
        f.write('\n')
    f.close()
    

if __name__ == '__main__':
    create_train_csv()
    create_subtest_csv()