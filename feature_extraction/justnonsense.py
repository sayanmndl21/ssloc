#import time, datetime


class dronelog():
    
    def __init__(self):
        
#        Let's assume the dataset was given through and API, as an array of dictionaries per se.
        self.DataSet=[{"account_id" : 1121345, "event_date": "2017-08-24", "account_standing": "G", "account_information": {"first_name": "John", "last_name": "Doe", "date_of_birth": "1986-08-18", "address": {"street_number": "123", "street_name": "Main Street", "city": "Centerville", "state": "CA", "zip_code": "91111"}, "email_address": "john_doe@gmail.com"} },{"account_id" : 1454581, "event_date": "2018-01-09", "account_standing": "B", "account_information": {"first_name": "Jane", "last_name": "Smith", "date_of_birth": "1975-09-09", "address": {"street_number": "345", "street_name": "Oak Drive", "unit_number": "12A", "city": "Mount Pleasant", "state": "CA", "zip_code": "90010"}, "email_address": "jane_smith@yahoo.com"} }]
        self.db=pymongo.MongoClient().clients
        self.allRecords=self.db.records
        self.maxAge=120
        self.accountIDlength=7
        
    #Save all the records    
    def saveRecords(self):
        self.allRecords.drop()
        for record in self.DataSet:
            if self.checkOneRecord(record):
                self.allRecords.insert_one(record)
            else:
                print('Invalid Record')
     
    #Get the time as a tuple
    def stripTime(self,timeString):
        return time.mktime(datetime.datetime.strptime(timeString,"%Y-%m-%d").timetuple())
    
    # Get the latest record
    def findTheLatest(self):
        latestTime=0
        latestRecord=None
        
        for record in self.allRecords.find():
            if self.stripTime(record["event_date"])>latestTime:
                latestTime=self.stripTime(record["event_date"])
                latestRecord=record
        return  latestRecord  
    
    # Insert a  record
    def insertARecord(self,record):
        if self.checkOneRecord(record):
            self.allRecords.insert_one(record)
        else:
            print('Invalid Record')
     
    # Delete a  record    
    def deleteARecord(self,ID):
        try:
            self.allRecords.delete_one({"account_id" :ID})
        except:
            print('Record not deleted')
    
    
    # Check if a record exists
    def exists(self,ID):
        return self.allRecords.find_one({"account_id" :ID}) !=None
    
    # Check the validity of a record
    def checkOneRecord(self,record):
        
        try:
            if len(str(record["account_id"])) !=self.accountIDlength:
                return False
            elif len((record["account_information"]["email_address"]).split('@')[1].split('.')) !=2:
                return False
            elif int(str(datetime.datetime.now()).split('-')[0]) - int(record["account_information"]["date_of_birth"].split('-')[0]) >self.maxAge:
                return False
            else:
                return True
        except:
            print('Invalid record!')
            return False
        
    
if __name__ == "__main__":
    myCustomers=Customers()
    
    myCustomers.saveRecords()
    
    print(myCustomers.findTheLatest())