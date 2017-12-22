'''
SUMMARY:  config file
AUTHOR:   YONG XU
Created:  2017.12.20
Modified: 
--------------------------------------
'''
ori_dev_root ='/vol/vssp/msos/yx/source_separation/enhSpec2wav' ### running folder

dev_wav_fd = '/vol/vssp/AP_datasets/audio/dcase2016/timit_enh/wav'  ### fold where all wavs are there

# temporary data folder
scrap_fd = "/vol/vssp/msos/yx/source_separation/enhSpec2wav/fea"    ### generated log-magnitude-fea parent folder
dev_fe_mel_fd = scrap_fd + '/fe/log_mag_spec'    ### generated log-magnitude-fea folder

enh_fea_wav_fd = "/vol/vssp/msos/yx/source_separation/enhSpec2wav/enh_wavs" ### enhanced fea and wav

model_fd = "/vol/vssp/msos/yx/source_separation/enhSpec2wav/fea/md" ### trained models folder

dev_cv_csv_path = ori_dev_root + '/file_list.csv'  ### file list "number id,file name,train (0) or test (1) flag", including train and test files

fs = 16000.  # sample rate
win = 512. #32ms ###win size
