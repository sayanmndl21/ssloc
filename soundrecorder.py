

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
