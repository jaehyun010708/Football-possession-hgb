"""Regression tests for the fixed ball feature schema (does not require IDSSE XML)."""
import unittest

from src.ball_features import BALL_FEATURES, nearest_team
import numpy as np

class FeatureTests(unittest.TestCase):
    def test_11_ball_features(self):
        self.assertEqual(len(BALL_FEATURES),11)
        self.assertEqual(len(set(BALL_FEATURES)),11)
    def test_nearest_player(self):
        px=np.array([[.50,.60],[.40,.70]],dtype=np.float32)
        py=np.array([[.50,.50],[.20,.80]],dtype=np.float32)
        bx=np.array([.50,np.nan],dtype=np.float32)
        by=np.array([.50,np.nan],dtype=np.float32)
        dist,count=nearest_team(px,py,bx,by,np.array([True,False]))
        self.assertAlmostEqual(float(dist[0]),0.0,places=5)
        self.assertEqual(int(count[0]),1)
        self.assertTrue(np.isnan(dist[1]))

if __name__=='__main__':unittest.main()
