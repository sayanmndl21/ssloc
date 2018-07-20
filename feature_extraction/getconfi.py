import numpy as np
import pandas as pd

class logdata:
    def __init__(self, size):
        self.size = size
        self.df = pd.DataFrame(data = None, 
                               columns = ['Timestamp','Label','Occurance', 'Confidence'],
                              )
        self.df1 = pd.DataFrame(data = None, 
                               columns = ['Timestamp','Actuallabel','Labelpredicted_S','Labelpredicted_R'],
                              )
        

    def insertdf(self, x, timestamp):
        self.x = x
        self.times = timestamp
        # default values
        self.occurance = 1
        self.confidence = 100

        self.df = self.df.append(pd.Series({
            'Timestamp': self.times, 
            'Label': self.x, 
            'Occurance': self.occurance, 
            'Confidence': self.confidence
        }), ignore_index=True)
        

        self.df.sort_index(inplace=True, ascending=False)
        self.del_row()

        # Calculate the confidence and occurances of labels
        if self.df.shape[0] > 1:
            self.occurance = self.get_occurance()
            self.confidence = self.get_confidence(self.occurance)

            self.df['Occurance'] = self.df.Label.apply(lambda x: self.occurance[x])
            self.df['Confidence'] = self.df.Label.apply(lambda x: self.confidence[x])


        
        return self.df

    
    def logdf(self, user_x, x1,x2, file,i):
        
        self.df1 = self.df1.append(pd.Series({
            'Timestamp': self.times,
            'Actuallabel':user_x,
            'Labelpredicted_S': x1, 
            'Labelpredicted_R': x2, 
        }), ignore_index=True)
        

        self.df1.sort_index(inplace=True, ascending=False)
        if self.df.shape[0] > int(i):
            self.df1.to_csv(file+".csv", sep='\t', encoding='utf-8')
        #iter+= 1

        return self.df1     

        


    def dfempty(self):
        return self.df.empty

    def get_occurance(self):
        # group by label and count
        occ = self.df.groupby('Label').Timestamp.count().rename('Occurance').astype(int)
        return occ

    def get_confidence(self, occurance):
        conf = ((occurance / sum(occurance)).rename('Confidence') * 100).astype('float64')
        return conf

    def del_row(self):
        if self.df.shape[0] > int(self.size):
            self.df = self.df.head(self.size)

    def get_result(self):
        return self.df.loc[self.df['Confidence'].idxmax()]