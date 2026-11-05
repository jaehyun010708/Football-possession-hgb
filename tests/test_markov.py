import unittest
import numpy as np
from src.markov import segment_ids, viterbi_decode
from src.metrics import sequence_starts

class MarkovTests(unittest.TestCase):
    def setUp(self):
        self.probs=np.array([
            [.90,.05,.05], [.85,.10,.05], [.10,.85,.05], [.87,.08,.05], [.88,.07,.05],
        ])
        self.seg=np.array([1,1,1,1,1])
        self.trans=np.array([[.97,.02,.01],[.02,.97,.01],[.05,.05,.90]])
    def test_zero_strength_raw_argmax(self):
        np.testing.assert_array_equal(
            viterbi_decode(self.probs,self.seg,self.trans,0), self.probs.argmax(axis=1))
    def test_short_spike_removed(self):
        states=viterbi_decode(self.probs,self.seg,self.trans,.5)
        np.testing.assert_array_equal(states, np.zeros(5,dtype=np.int8))
    def test_segment_reset(self):
        ids=np.array([5,10,15,1,6,11])
        np.testing.assert_array_equal(segment_ids(ids),[1,1,1,2,2,2])
        np.testing.assert_array_equal(sequence_starts(ids),[3])
    def test_invalid_weight(self):
        with self.assertRaises(ValueError):
            viterbi_decode(self.probs,self.seg,self.trans,-.1)

if __name__=='__main__':unittest.main()
