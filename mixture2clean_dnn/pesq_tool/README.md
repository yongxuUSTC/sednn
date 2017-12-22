## PESQ tool for speech enhancement evaluation. 

Download the PESQ source code from: https://www.itu.int/rec/T-REC-P.862-200102-I/en

Then compile:

$ gcc -o PESQ *.c -lm

Then use:

./pesq clean_speech.wav enhanced_speech.wav +16000

