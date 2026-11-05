"""Numerical shape check for original 191-player-feature specification."""
import unittest
import numpy as np
import pandas as pd
from src.player_features import add_team_features, add_opposition_features, add_temporal_features
from src.config import CANONICAL_LENGTH_M, CANONICAL_WIDTH_M
from src.ball_features import BALL_FEATURES

class FeatureCountTests(unittest.TestCase):
    def test_player_191_plus_ball_11(self):
        t, n = 35, 4
        rng = np.random.default_rng(72)
        hx, hy, ax, ay = (rng.uniform(.1, .9, size=(t, n)).astype(np.float32) for _ in range(4))
        hs, ass = (rng.uniform(0, 9, size=(t, n)).astype(np.float32) for _ in range(2))
        hv, av = np.ones((t,n), dtype=bool), np.ones((t,n), dtype=bool)
        result = pd.DataFrame(index=range(t))
        add_team_features(result, hx, hy, hs, hv, 'home')
        add_team_features(result, ax, ay, ass, av, 'away')
        result['centroid_x_gap'] = result['home_x_mean']-result['away_x_mean']
        result['centroid_y_gap'] = result['home_y_mean']-result['away_y_mean']
        result['centroid_distance'] = np.sqrt(
            np.square(CANONICAL_LENGTH_M*result['centroid_x_gap'])
            +np.square(CANONICAL_WIDTH_M*result['centroid_y_gap']))
        result['speed_mean_gap'] = result['home_speed_mean']-result['away_speed_mean']
        result['n_gap'] = result['home_n']-result['away_n']
        result['x_spread_gap'] = result['home_x_width']-result['away_x_width']
        home_xy = np.stack([hx*CANONICAL_LENGTH_M,hy*CANONICAL_WIDTH_M], axis=-1)
        away_xy = np.stack([ax*CANONICAL_LENGTH_M,ay*CANONICAL_WIDTH_M], axis=-1)
        add_opposition_features(result, home_xy, away_xy, hv, av)
        self.assertEqual(result.shape[1],68)
        got=add_temporal_features(result, np.ones(t,dtype=np.int32), fps=5)
        self.assertEqual(got.shape[1],191)
        self.assertEqual(len(BALL_FEATURES),11)
        self.assertEqual(got.shape[1]+len(BALL_FEATURES),202)

if __name__=='__main__':unittest.main()
