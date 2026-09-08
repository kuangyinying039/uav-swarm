import unittest
import numpy as np

from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv
from pursuit_safety import safe_velocity_step, clear_segment, paths_conflict


class SafetyV3Tests(unittest.TestCase):
    def env(self):
        env = QuadrotorPursuitEnv(QuadrotorPursuitConfig(seed=17,building_count=0,n_obstacles=0))
        env.quadrotor_states[:,:3] = [[24.79,10,5],[4,4,5],[4,20,5]]
        env.quadrotor_states[:,3:6] = 0.
        env.positions = env.quadrotor_states[:,:2].copy()
        env.altitudes = env.quadrotor_states[:,2].copy()
        return env

    def test_boundary_preserves_tangential_motion(self):
        env = self.env()
        state, ref, reasons, emergency, path = safe_velocity_step(env,0,np.array([1.,.7,.2]),0.,.2)
        self.assertFalse(emergency)
        self.assertIn('boundary',reasons)
        self.assertLess(ref[0],0.)
        self.assertAlmostEqual(ref[1],.7)
        self.assertGreater(state[1],10.)
        self.assertLessEqual(state[0],24.8)
        self.assertTrue(all(clear_segment(env,a,b) for a,b in zip(path,path[1:])))

    def test_building_preserves_tangential_motion(self):
        env = self.env()
        env.buildings = np.array([[10.,8.,12.,12.]])
        env.building_heights = np.array([8.])
        env.quadrotor_states[0,:3] = [9.5,10.,5.]
        state,ref,reasons,emergency,_ = safe_velocity_step(env,0,np.array([1.,.7,0.]),0.,.2)
        self.assertFalse(emergency)
        self.assertIn('building',reasons)
        self.assertGreater(state[1],10.)
        self.assertLess(state[0],9.55)

    def test_unrecoverable_momentum_is_counted_and_can_recover_next_step(self):
        env = self.env()
        env.quadrotor_states[0,3] = 1.4
        state,_,_,emergency,_ = safe_velocity_step(env,0,np.array([1.,.7,0.]),0.,.2)
        self.assertTrue(emergency)
        self.assertLessEqual(state[0],24.8)
        env.quadrotor_states[0] = state
        recovered,_,_,emergency,_ = safe_velocity_step(env,0,np.array([1.,.7,0.]),0.,.2)
        self.assertFalse(emergency)
        self.assertGreater(recovered[1],state[1])

    def test_swept_building_and_peer_checks_prevent_tunneling(self):
        env = self.env()
        env.buildings = np.array([[10.,8.,12.,12.]])
        env.building_heights = np.array([8.])
        self.assertFalse(clear_segment(env,np.array([8.,10.,5.]),np.array([14.,10.,5.])))
        a = np.array([[0.,0.,0.],[2.,0.,0.]])
        b = np.array([[2.,0.,0.],[0.,0.,0.]])
        self.assertTrue(paths_conflict(a,b,.5))

    def test_parent_does_not_apply_second_planar_avoidance_move(self):
        env = self.env()
        env.quadrotor_states[:,:3] = [[10.,10.,3.],[10.4,10.,10.],[18.,18.,5.]]
        env.positions = env.quadrotor_states[:,:2].copy()
        env.altitudes = env.quadrotor_states[:,2].copy()
        before = env.positions.copy()
        result = env.step_joint(np.zeros((3,4)))
        np.testing.assert_allclose(env.positions,env.quadrotor_states[:,:2],atol=1e-12)
        np.testing.assert_allclose(env.positions,before,atol=1e-12)
        self.assertEqual(result['reward_components']['safety_intervention'],0.)

    def test_disabled_agents_do_not_inflate_feasibility_and_rewards_add_up(self):
        env = self.env()
        env.disabled_uavs[1:] = True
        env.quadrotor_states[0,3] = 1.4
        result = env.step_joint(np.tile([1.,.7,0.,0.],(3,1)))
        self.assertEqual(result['controller_feasible_rate'],0.)
        self.assertEqual(result['emergency_stop_rate'],1.)
        components = {k:v for k,v in result['reward_components'].items() if k not in ('pursuit','estimation')}
        self.assertAlmostEqual(result['reward'],sum(components.values()))
        self.assertEqual(result['continuous_actions'][0],[1.,.7,0.,0.])
        self.assertNotEqual(result['desired_velocities'][0],result['executed_velocity_references'][0])


if __name__ == '__main__':
    unittest.main()
