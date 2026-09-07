import unittest
import numpy as np
from quadrotor_pursuit_env import QuadrotorPursuitConfig
from pursuit_rewards import geometry_features, shaped_rewards, obstacle_cost


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.c = QuadrotorPursuitConfig()
        self.active = np.ones(3, dtype=bool)

    def features(self, points):
        return geometry_features(np.array(points, dtype=float), np.zeros(3), self.active, self.c)

    def test_non_nearest_agent_gets_progress_and_reverse_is_negative(self):
        a = self.features([[2,0,0],[8,0,0],[10,0,0]])
        b = self.features([[2,0,0],[7,0,0],[10,0,0]])
        r, progress = shaped_rewards(a,b,self.active,self.active,self.c)
        self.assertGreater(r['individual_approach'],0)
        self.assertGreater(progress[1],0)
        reverse,_ = shaped_rewards(b,a,self.active,self.active,self.c)
        self.assertAlmostEqual(r['individual_approach'],-reverse['individual_approach'])

    def test_close_progress_stronger_and_no_singularity(self):
        close = np.log(2/1.5)
        far = np.log(8/7.5)
        self.assertGreater(close,far)
        a = self.features([[0,0,0]]*3)
        self.assertTrue(np.all(np.isfinite(a[0])))
        self.assertLess(self.c.target_proximity_weight,self.c.time_penalty)

    def test_encirclement_needs_proximity_and_height(self):
        angles=np.arange(3)*2*np.pi/3
        ring=np.column_stack([3*np.cos(angles),3*np.sin(angles),np.zeros(3)])
        score=self.features(ring)[2]
        self.assertGreater(score,self.features([[3,0,0],[4,0,0],[5,0,0]])[2])
        self.assertGreater(score,self.features(ring*5)[2])
        high=ring.copy(); high[:,2]=6
        self.assertGreater(score,self.features(high)[2])
        same,_=shaped_rewards(self.features(ring),self.features(ring),self.active,self.active,self.c)
        self.assertEqual(same['encirclement_progress'],0)

    def test_obstacle_distance_uses_height_and_is_bounded(self):
        def cost(p):
            return obstacle_cost(np.array([p]*3),self.active,[[0,0,2,2]],[3],self.c)
        self.assertLess(cost([2.5,1,1]),cost([3.5,1,1]))
        self.assertEqual(cost([1,1,8]),0)
        self.assertAlmostEqual(cost([1,1,1]),-self.c.obstacle_proximity_weight)

    def test_disabled_agent_cannot_create_positive_approach(self):
        a=self.features([[2,0,0],[8,0,0],[10,0,0]])
        b=self.features([[1,0,0],[8,0,0],[10,0,0]])
        _,progress=shaped_rewards(a,b,self.active,np.array([False,True,True]),self.c)
        self.assertEqual(progress[0],0)
