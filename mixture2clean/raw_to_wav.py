import os
import argparse
import numpy as np

import prepare_data as pp_data
import config as cfg

def noise_raw_to_wav(args):
    noise_dir = "/vol/vssp/datasets/audio/dcase2016/timit_enh/timit_enhancement/noise"
    workspace = args.workspace
    fs = cfg.sample_rate

    # Output directory. 
    out_dir = os.path.join(workspace, "nosie_wavs")
    pp_data.create_folder(out_dir)
    
    # Convert files from raw to wav. 
    names = os.listdir(noise_dir)
    for na in names:
        print(na)
        bare_na = os.path.splitext(na)[0]
        noise_path = os.path.join(noise_dir, na)
        out_path = os.path.join(out_dir, "%s.wav" % bare_na)
        
        # Read raw. 
        raw_audio = np.memmap(noise_path, dtype='h', mode='r')
        raw_audio = raw_audio.astype(np.float32) / 32768.
        
        # Write out wav. 
        pp_data.write_audio(out_path, raw_audio, fs)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    
    parser.add_argument('--workspace', type=str, required=True)
    
    args = parser.parse_args()
    
    noise_raw_to_wav(args)