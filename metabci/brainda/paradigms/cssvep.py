from .base import BaseTimeEncodingParadigm


class cSSVEP(BaseTimeEncodingParadigm):

    def is_valid(self, dataset):
        ret = True
        if dataset.paradigm != 'cSSVEP':
            ret = False
        return ret