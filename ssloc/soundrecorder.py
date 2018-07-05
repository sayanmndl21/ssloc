import numpy
import pyaudio
import math
import threading

class recorder():

    def __init__(self):
        self.RATE = 44100
        self.BUFFERSIZE = 3072 #ideal buffer for pi
        self.sectorec = .1
        self.INITIAL_TAP_THRESHOLD = 200
        self.INITIAL_TAP_THRESHOLD1 = 200
        self.FORMAT = pyaudio.paInt16
        self.threadsdie = False
        self.newAudio = False
        self.detectedmic1 = False
        self.detectedmic2 = False
    
    def setup(self):
        """initiate sound card"""
        self.buffersToRecord=int(self.RATE*self.sectorec/self.BUFFERSIZE)
        if self.buffersToRecord==0: self.buffersToRecord=1
        self.samplesToRecord=int(self.BUFFERSIZE*self.buffersToRecord)
        self.chunksToRecord=int(self.samplesToRecord/self.BUFFERSIZE)
        self.secPerPoint=1.0/self.RATE
        self.p = pyaudio.PyAudio()
        self.inStream = self.open_mic_stream()
        self.inStream1 = self.open_mic_stream1()
        self.xsBuffer=numpy.arange(self.BUFFERSIZE)*self.secPerPoint
        self.xs=numpy.arange(self.chunksToRecord*self.BUFFERSIZE)*self.secPerPoint
        self.audio=numpy.empty((self.chunksToRecord*self.BUFFERSIZE),dtype=numpy.int16)
        
    def open_mic_stream(self):
        stream = self.p.open(   format = self.FORMAT,
                                 channels = 1,
                                 rate = self.RATE,
                                 input = True,
                                 input_device_index = 4,
                                 frames_per_buffer = self.BUFFERSIZE)        

        return stream

    def open_mic_stream1(self):
        stream1 = self.p.open(  format = self.FORMAT,
                                 channels = 1,
                                 rate = self.RATE,
                                 input = True,
                                 input_device_index = 5,
                                 frames_per_buffer = self.BUFFERSIZE)

        return stream1


    def close(self):
        self.p.close(self.inStream)
        self.p.close(self.inStream1)

    def getAudio(self):
        audiostring=self.inStream.read(self.BUFFERSIZE)
        audiostring1=self.inStream1.read(self.BUFFERSIZE)
        self.newAudio=True
        data = numpy.fromstring(audiostring,dtype=numpy.int16)
        data1 = numpy.fromstring(audiostring1,dtype=numpy.int16)
        return numpy.concatenate([[data], [data1]])

    def recordsec(self, forever = True):
        """record in seconds"""
        while True:
            if self.threadsdie: break
            for i in range(self.chunksToRecord):
                self.audio[i*self.BUFFERSIZE:(i+1)*self.BUFFERSIZE] = self.getAudio()
                self.newAudio = True
                if forever == False: break


    """use this to call recorder continuously"""
    def continuerecord(self):
        self.t =threading.Thread(target = self.recordsec)
        self.t.start()

    def stopcontinue(self):
        self.threadsdie = True


    ### MATH ###

    def downsample(self,data,mult):
        """Given 1D data, return the binned average."""
        overhang=len(data)%mult
        if overhang: data=data[:-overhang]
        data=numpy.reshape(data,(len(data)/mult,mult))
        data=numpy.average(data,1)
        return data

    def fft(self,data=None,trimBy=10,logScale=False,divBy=100):
        if data==None:
            data=self.audio.flatten()
        left,right=numpy.split(numpy.abs(numpy.fft.fft(data)),2)
        ys=numpy.add(left,right[::-1])
        if logScale:
            ys=numpy.multiply(20,numpy.log10(ys))
        xs=numpy.arange(self.BUFFERSIZE/2,dtype=float)
        if trimBy:
            i=int((self.BUFFERSIZE/2)/trimBy)
            ys=ys[:i]
            xs=xs[:i]*self.RATE/self.BUFFERSIZE
        if divBy:
            ys=ys/float(divBy)
        return xs,ys

def loudness(chunk):
    data = numpy.array(chunk, dtype=float) / 32768.0
    ms = math.sqrt(numpy.sum(data ** 2.0) / len(data))
    if ms < 10e-8: ms = 10e-8
    return 10.0 * math.log(ms, 10.0)

def closest_value_index(array, guessValue):
    # Find closest element in the array, value wise
    closestValue = find_nearest(array, guessValue)
    # Find indices of closestValue
    indexArray = numpy.where(array==closestValue)
    # Numpys 'where' returns a 2D array with the element index as the value
    return indexArray[0][0]
