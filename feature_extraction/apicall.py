import requests
 
class apicalls(object):
    def __init__(self, api_url, apikey):
        self.url = api_url
        self.key = apikey

    
    def sendtoken(self, record):
        self.x = record['Label']
        self.Label = self.getLabel(int(self.x))
        self.timestamp =record['Timestamp']
        self.confidence = record['Confidence']
        #self.confidenceLabel = self.getConfidence(int(self.confidence))
        self.log = {"type": "Drone",
        "distance" : self.Label,
        "confidence": self.confidence,
        "location": "Drone Detector A",
        "time": self.timestamp
        }
        self.r =  requests.post(self.url, data = self.log)
        return self.r.text
        
    
    def getLabel(self,x):
        if x == 0:
            self.label = "far"
        elif x == 1:
            self.label = "midrange"
        elif x == 2:
            self.label = "near"
        elif x == 3:
            self.label = "very_far"
        elif x == 4:
            self.label = "very_near"
        return self.label
    
#    def getConfidence(self,y):
#        if y < 50:
#            self.confidencelabel = "Low at \n"+str(y)+"%"
#        elif y >= 50 and y < 85:
#            self.confidencelabel = "Medium at \n"+str(y)+"%"
#        elif y >= 85:
#            self.confidencelabel = "High at \n"+str(y)+"%"
#        return self.confidencelabel


 


