import  pymongo
import time, datetime

class dronelog():
    def __init__(self):
        self.timestamp = dronerec['Timestamp']
        self.x = dronerec['Label']
        self.Label = self.get_label(self.x)
        self.confidence = dronerec['Confidence']
        #example of dataset to release to api
        self.DataSet = [{"ID": 00000000, "Label": self.Label, "Timestamp": self.timestamp, "Confidence": self.confidence }]
        self.db = pymongo.MongoClient().clients
        self.allRecords = self.db.records
        self.accountIDlength = 7


    def get_label(self,x):
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

    def saveRecords(self):
        self.allRecords.drop()
        for record in self.DataSet:
            if self.checkonerecord(record):
                self.allRecords.insert_one(record)
            else:
                break
        
    def timestrip(self, timestamp):
        return time.mktime(datetime.datetime.strptime(timeString,'%Y-%m-%d %H:%M:%S').timetuple())
    

    def latestlog(self):
        latestTime = 0
        latestRecord = None
        for record in self.allRecords.find():
            if self.stripTime(record['Timestamp'])>latestTime:
            latestTime = self.stripTime(record['Timestamp'])
            latestRecord = record
        return latestRecord
    
    def insertlog(self, record):
        if self.checkonerecord(record):
            self.allRecords.insert_one(record)
        else:
            break
        
    def deleterecord(self,ID):
        try:
            self.allRecords.delete_one({"ID":ID})
        except:
            break
        
    def exists(self, ID):
        return self.allRecords.find_one({"ID":ID}) != None
    
    def checkonerecord(self,record):
        try:
            if len(str(record["ID"])) != self.accountIDlength:
                return False
            else:
                return True
        except:
            return False

        
        

data = { "name": "aaa", "time": "4:03AM 04/03/2018", "location": "Camera 1", "confidence": 31 }

response = requests.post('http://206.189.237.118/api/drones', data=data)



files = {'file': open('2a.pdf', 'rb')}

r = requests.post('http://mlc67-cmp-00.egr.duke.edu/api/photos/container/upload', files=files)