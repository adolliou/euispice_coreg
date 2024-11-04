

class AlignmentResults:

    def __init__(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def savefits(self, filename: str):
        # save  co-aligned fits
        raise NotImplementedError

    def savefig(self, filename: str):
        raise NotImplementedError

    def saveyaml(self, filename: str):
        # yaml with the correlation values
        raise NotImplementedError

    def return_shift(self):
        # return the shift values after 3d polynomial computation.
        raise NotImplementedError
