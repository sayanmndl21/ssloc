import numpy as np
import pandas as pd

class logdata:
    def __init__(self, size):
        self.size = size
        self.df = pd.DataFrame(data = None, 
                               columns = ['Timestamp','Label','Occurance', 'Confidence'],
                              )

    def insertdf(self, x, timestamp):
        # default values
        occurance = 1
        confidence = 100

        self.df = self.df.append(pd.Series({
            'Timestamp': timestamp, 
            'Label': x, 
            'Occurance': occurance, 
            'Confidence': confidence
        }), ignore_index=True)

        self.df.sort_index(inplace=True, ascending=False)
        self.del_row()

        # Calculate the confidence and occurances of labels
        if self.df.shape[0] > 1:
            occurance = self.get_occurance()
            confidence = self.get_confidence(occurance)

            self.df['Occurance'] = self.df.Label.apply(lambda x: occurance[x])
            self.df['Confidence'] = self.df.Label.apply(lambda x: confidence[x])

        return self.df

    def get_occurance(self):
        # group by label and count
        occ = self.df.groupby('Label').Timestamp.count().rename('Occurance').astype(int)
        return occ

    def get_confidence(self, occurance):
        conf = ((occurance / sum(occurance)).rename('Confidence') * 100).astype(int)
        return conf

    def del_row(self):
        if self.df.shape[0] > int(self.size):
            self.df = self.df.head(self.size)

    def get_result(self):
        return self.df.loc[self.df['Confidence'].idxmax()]