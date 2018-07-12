from pydub import AudioSegment
from pydub.utils import which
import os, sys, csv, re

AudioSegment.converter = which("ffmpeg")

os.chdir('/home/sayan/duke_internship/sound-analyzer/') # change to the file directory

for root, dirs, files in os.walk(sys.argv[1]):
	#os.mkdir('output')
	#with open(root+'.csv', 'w',newline='') as f:
	for file in files:
		path = os.path.join(root,file)
		src = path
		neAudio = AudioSegment.from_wav(path)
		playlist_length = len(neAudio) / (1000)
		for t in range(0,int(playlist_length),1):
			t1 = t*1000
			t2 = (t+1)*1000
			newAudio = AudioSegment.from_wav(path)
			newAudio = newAudio[t1:t2]
			newAudio.export(root+'/output/'+file[:-4]+str(int(t))+'.wav', format="wav")