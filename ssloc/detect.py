

class Detect(obj):

    def __init__(self, x, y, confidence, spl, timestamp):
        self.x = x
        self.y = y
        self.confidence = confidence
        self.spl = spl
        self.timestamp = timestamp
        self.std = None


    def get_x(self):
        return self.x


    def get_y(self):
        return self.y


    def get_confidence(self):
        return self.confidence


    def get_spl(self):
        return self.spl


    def get_timestamp(self):
        return self.timestamp


    def get_pos(self):
        return [self.x, self.y]


    def set_std(self, std):
        self.std = std


    def get_std(self):
        if self.std == None:
            raise AttributeError("Standard deviation not set")
        else:
            return self.std


    def __repr__(self):
        return ("DetectionEvent(x=" + str(self.x) +
            ", y=" + str(self.y) +
            ", confidence=" + str(self.confidence) +
            ", spl=" + str(self.spl) + ")"
        )


    def __str__(self):
        return "X: {0}, Y: {1}".format(self.x, self.y)
