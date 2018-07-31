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
  
## Training
  
When all the data is processed, you would get a csv file containing the ID, height, Ditance, Features etc. You can either use an IDLE(I have used spyder3) or bash to run the sript.
Change the metafile path to the csv file which you just created and  run the script to get your model. You can use pickle to save the model:
	
	 metafile = '[path to csv file].csv' #load dataset
	 
Save using:
	
	joblib.dump(svm, 'input/[name of dump file].pkl')

You can change nu value in `svm = OneVsRestClassifier(NuSVC(nu=.2, kernel='poly', decision_function_shape='ovr'))` or other parameters to check for different training errors
Once done, you can now use this model for detection and localisation.
## Real-Time Attribute Prediction

I have written several scripts for realtime detection. Although I have tested with Apex220 and Behringer ECM8000, the model should be okay with any mic. You can run the scripts `main.py`, `test.py`, `test1.py`, `test2.py`, `test3.py`, `test6.py`, `test7.py`, `test8.py`, `test9.py`. All are similar scripts with different models loaded but `test8.py` is an exception in a sense that it contains noise reduction algorithm. Just run the script to get prediction. You can uncomment api calls and add your own server settings to get api logs in your server and push notifications.
	
	 $ python3 test8.py
	 
The best performing script for now is test8.py, which does pretty good prediction almost all the time. Be sure to take atleast 15 sec background noise recordings before you start predicting.{the code is included in the script so, you dont have to record seperately} 
