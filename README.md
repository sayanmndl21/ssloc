# ssloc
A cheap alternative to Sound Source Localization using single microphone

## Requirements
I have used python3 and python3.6 for this project
All requirements for pc are written in the `requirements.txt` file
In case of raspberrypi, use `requirementsrasppi.txt` to install dependencies as some packages dont work on raspberry pi
Use: 
	
	 $ pip3 install -r requirements.txt
	

## Recording Audio 
If you want to train the model using SVM, record audio with `recordaudio.py`. Be sure to label data properly.
Labelling the data is cruicial for training.
Use:
	
	 $ python recordaudio.py [height(number)] [distance(number)] [index(int)] [duration(sec){optional}]
	
 Audio analysis blows out if the recordings are used as it is. So, I we have to normalise tha data first. Use:
	
	 $ python npytowav.py [folder/directory]
	
 For realtime analysis, I have used 1 sec data and extracted features from it. You can use `splitaudio.py` split the audio data.
	
	 $ python splitaudio.py [folder/directory]
	
  Finally use `writetocsv.py` to extract features from the dataset you want. I have made sure to get labels from the file names as it collects data. So a table is generated with its ID, height, distance, indexand features
	
	 $ python writetocsv.py [folder/directory] [noisefilename.wav]
	
  Here noise file name is a raw noise recording you have to get using the first two lines. It is advised to get atleast 15 seconds so that noise spectral features are properly calculated.
  
  ##
