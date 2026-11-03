import unittest
import numpy as np
from entropy_core import aligned_slice,signals_numpy


class Tests(unittest.TestCase):
    def test_uniform_entropy(self):
        r=signals_numpy(np.zeros((3,4)),np.zeros((3,4)),[0,1,2])
        np.testing.assert_allclose(r['H_pressure'],np.log(4))
        np.testing.assert_array_equal(r['delta_H'],0)
        np.testing.assert_array_equal(r['JSD'],0)

    def test_confident_opposite_answers_have_equal_entropy(self):
        r=signals_numpy(np.log([[.9,.1]]),np.log([[.1,.9]]),[0])
        self.assertAlmostEqual(r['delta_H'][0],0)
        self.assertGreater(r['JSD'][0],0.3)
        self.assertNotEqual(r['top1_pressure'][0],r['top1_neutral'][0])

    def test_entropy_difference_sign(self):
        r=signals_numpy(np.log([[.5,.5],[.99,.01]]),np.log([[.99,.01],[.5,.5]]),[0,1])
        self.assertGreater(r['delta_H'][0],0)
        self.assertLess(r['delta_H'][1],0)

    def test_masked_tokens(self):
        r=signals_numpy([[0,-np.inf]],[[0,-np.inf]],[0])
        self.assertEqual(r['H_pressure'][0],0)
        self.assertEqual(r['JSD'][0],0)

    def test_bad_distribution_rejected(self):
        for x in [[[-np.inf,-np.inf]],[[0,np.nan]],[[0,np.inf]]]:
            with self.assertRaises(ValueError):signals_numpy(x,[[0,0]],[0])

    def test_predictor_alignment_and_shared_actual_history(self):
        # Exact causal toy model: each row uses the whole prefix ending at its own position.
        # Neutral's top1 deliberately differs from the actual next token, so feeding t' breaks this test.
        vocab=7;u=[6,2,1];neutral=[6,1];actual=[4,3,0,5]
        def forward(ids):
            rows=[]
            for j in range(len(ids)):
                history=np.asarray(ids[:j+1]);weights=np.arange(1,j+2)
                center=int((history*weights).sum())%vocab
                rows.append(-np.square(np.arange(vocab)-center)/3.)
            return np.asarray(rows)
        paired=[]
        for prompt in [u,neutral]:
            seq=prompt+actual[:-1]
            full=forward(seq)[aligned_slice(len(prompt),len(actual))]
            sequential=np.stack([forward(prompt+actual[:k])[-1] for k in range(len(actual))])
            np.testing.assert_allclose(full,sequential,atol=0,rtol=0)
            paired.append(full)
        self.assertNotEqual(paired[1][0].argmax(),actual[0])
        r=signals_numpy(*paired,actual)
        self.assertEqual(len(r['delta_H']),len(actual))
        # Future output changes cannot affect the already-predicted rows.
        changed=actual.copy();changed[2]=6
        base=forward(u+actual[:-1])[aligned_slice(len(u),len(actual))]
        future=forward(u+changed[:-1])[aligned_slice(len(u),len(actual))]
        np.testing.assert_allclose(base[:3],future[:3],atol=0,rtol=0)


if __name__=='__main__':unittest.main(verbosity=2)
