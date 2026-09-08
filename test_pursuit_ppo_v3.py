from copy import deepcopy
import unittest
import torch
from torch import nn

from pursuit_ppo import guarded_actor_step


class PPOGuardTests(unittest.TestCase):
    def model(self):
        actor = nn.Linear(1,1,bias=False)
        nn.init.zeros_(actor.weight)
        optimizer = torch.optim.Adam(actor.parameters(),lr=.5)
        actor.weight.grad = torch.ones_like(actor.weight)
        return actor,optimizer

    def test_large_first_step_backtracks_inside_kl_limit(self):
        actor,optimizer = self.model()
        assess = lambda: actor.weight.square().sum()
        success,kl,retries = guarded_actor_step(actor,optimizer,assess,.001,retries=8)
        self.assertTrue(success)
        self.assertGreater(retries,0)
        self.assertLessEqual(kl,.0015)
        self.assertEqual(optimizer.state[actor.weight]['step'].item(),1)

    def test_rejection_restores_weights_adam_moments_and_learning_rate(self):
        actor,optimizer = self.model()
        optimizer.step()
        before = actor.weight.detach().clone()
        saved = deepcopy(optimizer.state_dict())
        success,_,_ = guarded_actor_step(actor,optimizer,lambda: ((actor.weight-before)**2).sum(),1e-20,retries=0)
        self.assertFalse(success)
        torch.testing.assert_close(actor.weight,before,rtol=0,atol=0)
        current = optimizer.state_dict()
        self.assertEqual(current['param_groups'],saved['param_groups'])
        for key,value in saved['state'][0].items():
            torch.testing.assert_close(current['state'][0][key],value,rtol=0,atol=0)

    def test_nonfinite_candidate_is_rolled_back(self):
        actor,optimizer = self.model()
        before = actor.weight.detach().clone()
        success,_,_ = guarded_actor_step(actor,optimizer,lambda: float('nan'),.01,retries=0)
        self.assertFalse(success)
        torch.testing.assert_close(actor.weight,before,rtol=0,atol=0)

    def test_failed_kl_evaluation_does_not_leave_applied_update(self):
        actor,optimizer = self.model()
        before = actor.weight.detach().clone()
        def failed():
            raise RuntimeError('evaluation failed')
        with self.assertRaisesRegex(RuntimeError,'evaluation failed'):
            guarded_actor_step(actor,optimizer,failed,.01)
        torch.testing.assert_close(actor.weight,before,rtol=0,atol=0)
        self.assertEqual(len(optimizer.state),0)


if __name__ == '__main__':
    unittest.main()
