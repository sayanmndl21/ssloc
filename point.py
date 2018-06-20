import math

Re = 1000 *6371

class Point(obj):

    def __init__(self, x, y, h):
        self.x = x
        self.y = y
        self.h = h


    def get_x(self):
        return self.x


    def get_y(self):
        return self.y

    def get_h(self):
        return self.h


    def set_x(self, x):
        self.x = x
        return self


    def set_y(self, y):
        self.y = y
        return self

    def set_h(self, h):
        self.h = h
        return self

    def dist_to(self, other):
        return math.sqrt(
            pow(self.x - other.x, 2) +
            pow(self.y - other.y, 2)
        )


    def dist_to_lat_long(self, other):
        lat1 = math.radians(self.x)
        lon1 = math.radians(self.y)
        lat2 = math.radians(other.x)
        lon2 = math.radians(other.y)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = ((math.sin(dlat / 2)) ** 2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon / 2)) ** 2)

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = Re * c

        return distance


    def to_list(self):
        return [self.x, self.y, self.h]


    def __str__(self):
        return "X: {0}, Y: {1}, H: {2}".format(self.x, self.y, self.h)


    def __repr__(self):
        return "Point({0}, {1}, {2})".format(self.x, self.y, self.h)
