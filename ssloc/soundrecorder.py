import numpy
import pyaudio
import math

class record():

    def __init__(self):
        self.RATE = 44100
        self.BUFFERSIZE = 3072 #ideal buffer for pi
        self.sectorec = .1
        self.threadsdie = False
        self.newAudio = False

    def setup(self):
        """initiate sound card"""
        self.buffersToRecord=int(self.RATE*self.sectorec/self.BUFFERSIZE)
        if self.buffersToRecord==0: self.buffersToRecord=1
        self.samplesToRecord=int(self.BUFFERSIZE*self.buffersToRecord)
        self.chunksToRecord=int(self.samplesToRecord/self.BUFFERSIZE)
        self.secPerPoint=1.0/self.RATE
        self.p = pyaudio.PyAudio()
        self.inStream = self.p.open(format=pyaudio.paInt16,channels=1,rate=self.RATE,input=True,frames_per_buffer=self.BUFFERSIZE)
        self.xsBuffer=numpy.arange(self.BUFFERSIZE)*self.secPerPoint
        self.xs=numpy.arange(self.chunksToRecord*self.BUFFERSIZE)*self.secPerPoint
        self.audio=numpy.empty((self.chunksToRecord*self.BUFFERSIZE),dtype=numpy.int16)

    def close(self):
        self.p.close(self.inStream)

    def getAudio(self):
        audiostring=self.inStream.read(self.BUFFERSIZE)
        self.newAudio=True
        return numpy.fromstring(audiostring,dtype=numpy.int16)

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
        self.t =threading.Thread(target = self.record)
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
