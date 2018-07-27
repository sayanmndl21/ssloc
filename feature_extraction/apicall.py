import requests
import json
 
class apicalls(object):
    def __init__(self, api_url, apikey, push_url, pushkey):
        self.url = api_url
        self.key = apikey
        self.pushurl = push_url
        self.pushKey = pushkey

    
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
    
    def sendtoken1(self, record):
        self.x = record['Label']
        self.Label1 = self.getLabel1(int(self.x))
        self.timestamp =record['Timestamp']
        self.confidence = record['Confidence']
        #self.confidenceLabel = self.getConfidence(int(self.confidence))
        self.log = {"type": "Drone",
        "distance" : self.Label1,
        "confidence": self.confidence,
        "location": "Drone Detector A",
        "time": self.timestamp
        }
        self.r =  requests.post(self.url, data = self.log)
        return self.r.text
    
    def sendtoken2(self, record):
        self.x = record['Label']
        self.Label2 = self.getLabel2(int(self.x))
        self.timestamp =record['Timestamp']
        self.confidence = record['Confidence']
        #self.confidenceLabel = self.getConfidence(int(self.confidence))
        self.log = {"type": "Drone",
        "distance" : self.Label2,
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

    def getLabel1(self,x):
        if x == 0:
            self.label = "far"
        elif x == 1:
            self.label = "midrange"
        elif x == 2:
            self.label = "near"
        elif x == 3 or x == 4:
            self.label = "very_far"
        elif x == 5:
            self.label = "very_near"
        return self.label

    def getLabel2(self,x):
        if x == 0:
            self.label = "far"
        elif x == 1:
            self.label = "midrange"
        elif x == 2:
            self.label = "near"
        elif x == 3:
            self.label = "vnear"
        return self.label
    
    def push_notify(self):
        self.header = {"Content-Type": "application/json; charset=utf-8",
        "Authorization": "Basic NDMyMTM5MjctMzYxZC00OTM3LTkxODEtYjljNDY5OTdmNGE0"}
        self.payload = {"app_id": "2ebe188c-34d4-423f-8c7f-21bd0483fc95",
        "contents": {"en": "Drone Detected!!"},
	    "template_id": "658d2118-ea02-4902-88e0-b708fa2e4fcd",
        "included_segments": ["All"]}
        self.req = requests.post(self.pushurl,headers = self.header,data = json.dumps(self.payload))
        return self.req.text

    
#    def getConfidence(self,y):
#        if y < 50:
#            self.confidencelabel = "Low at \n"+str(y)+"%"
#        elif y >= 50 and y < 85:
#            self.confidencelabel = "Medium at \n"+str(y)+"%"
#        elif y >= 85:
#            self.confidencelabel = "High at \n"+str(y)+"%"
#        return self.confidencelabel


 


