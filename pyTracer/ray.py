class ray(object):
    def __init__(self, o, d):
        self.O = o
        self.D = d

    def origin(self):
        return self.O

    def direction(self):
        return self.D

    def point_at_parameter(self, t):
        return (self.O + self.D*t)
