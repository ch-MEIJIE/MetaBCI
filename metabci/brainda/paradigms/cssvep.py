from .base import BaseParadigm, BaseTimeEncodingParadigm


class cSSVEP(BaseTimeEncodingParadigm):

    def is_valid(self, dataset):
        ret = True
        if dataset.paradigm != 'cSSVEP':
            ret = False
        return ret
    

class cSSVEP_simu(BaseParadigm):
   
    def is_valid(self, dataset):
        ret = True
        if dataset.paradigm != 'cSSVEP_simu':
            ret = False
        return ret