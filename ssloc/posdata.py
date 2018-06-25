import json

class position(obj):

    def __init__(self, coord, confidence):
        self.coord = coord
        self.x = coord.x
        self.y = coord.y
        self.confidence = confidence
        return self

    def get_coord(self):
        return self.coord

    def get_confidence(self):
        return self.confidence
##
    def set_coord(self, coord):
        self.coord = coord
        return self


    def set_confidence(self, confidence):
        self.confidence = confidence
        return self


    def to_dict(self):
        return {
            "position": {
                "x": self.coord.x,
                "y": self.coord.y,
            },
            "confidence": self.confidence
        }


    def to_json(self):
        return json.dumps(self.to_dict())


    def __hash__(self):
        return hash(repr(self))


    def __eq__(self, other_location):
        attrs = ["x", "y", "confidence"]

        for attr in attrs:
            if not hasattr(other_location, attr):
                return False

            if not getattr(self, attr) == getattr(other_location, attr):
                return False

        return True


    def __repr__(self):
        return "Position(coord={0}, confidence={1})".format(
            repr(self.coord), self.confidence
        )
