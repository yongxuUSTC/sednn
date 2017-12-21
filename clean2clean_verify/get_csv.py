import os
import glob

path        ="/vol/vssp/AP_datasets/audio/dcase2016/timit_enh/wav/train/"
train_files = glob.glob(os.path.join(path,'TRAIN*.WAV'))
test_path  ="/vol/vssp/AP_datasets/audio/dcase2016/timit_enh/wav/train/"
test_files = glob.glob(os.path.join(test_path,'TEST*.WAV'))

csvf=open("file_list.csv","w")
id=0
for f in train_files:
   f=f.lstrip('/vol/vssp/AP_datasets/audio/dcase2016/timit_enh/wav/train/')
   f=f.rstrip('.WAV')
   csvf.write("%d,%s,0\n"%(id,f))
   id+=1

for f in test_files:
   f=f.lstrip('/vol/vssp/AP_datasets/audio/dcase2016/timit_enh/wav/train/')
   f=f.rstrip('.WAV')
   csvf.write("%d,%s,1\n"%(id,f))
   id+=1

csvf.close()
