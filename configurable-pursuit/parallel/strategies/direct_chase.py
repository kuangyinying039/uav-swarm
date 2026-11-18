from .base import Strategy

class DirectChase(Strategy):
    def target(self,c,live=True):
        return c.e[:,None].expand_as(c.p)
