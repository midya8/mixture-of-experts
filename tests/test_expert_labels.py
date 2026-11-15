"""Expert labels are arbitrary only away from index-resolved routing ties."""
import copy
import itertools
import json
import math
from pathlib import Path
import random
import unittest

from moelab.moe import MoE

FROZEN = json.loads((Path(__file__).resolve().parents[1] / 'moelab/route_data/frozen.json').read_text())


def restore(values):
    model = MoE(len(values['W'][0]), len(values['W']), random.Random(0))
    for key, value in values.items():
        setattr(model, key, copy.deepcopy(value))
    return model


def permute(model, order):
    other = copy.deepcopy(model)
    for key in ('W', 'b', 'R', 'c'):
        setattr(other, key, [copy.deepcopy(getattr(model, key)[e]) for e in order])
    return other


class ExpertLabelTests(unittest.TestCase):
    def close(self, a, b, tolerance=2e-10):
        if isinstance(a, list):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.close(x, y, tolerance)
        else:
            self.assertTrue(math.isfinite(a) and math.isfinite(b))
            self.assertLessEqual(abs(a-b), tolerance*max(1, abs(a), abs(b)), (a, b))

    def test_every_saved_row_and_all_expert_permutations(self):
        count = 0
        for values in FROZEN['models'].values():
            model = restore(values)
            rows = FROZEN['splits']['original_train'] + FROZEN['splits']['original_test']
            for order in itertools.permutations(range(model.n_experts)):
                other = permute(model, order)
                for row in rows:
                    x = row['x']
                    gates = model.gates(x)
                    # All archived points have a separated top probability.
                    self.assertGreater(sorted(gates)[-1]-sorted(gates)[-2], 1e-8)
                    self.close(other.gates(x), [gates[e] for e in order])
                    self.assertEqual(order[other.top_expert(x)], model.top_expert(x))
                    self.close(other.predict_top1(x), model.predict_top1(x))
                    self.close(other.predict_soft(x), model.predict_soft(x))
                    count += 1
        self.assertEqual(count, 24480)

    def test_permuted_online_updates_with_and_without_auxiliary_loss(self):
        rng = random.Random(614)
        updates = 0
        for case in range(40):
            d, experts = 1 + case % 7, 2 + case % 4
            for aux in (0., .05):
                model = MoE(d, experts, rng)
                order = list(range(experts))
                rng.shuffle(order)
                other = permute(model, order)
                for step in range(6):
                    x = [rng.uniform(-1, 1) for _ in range(d)]
                    target = rng.uniform(-2, 2)
                    g = model.gates(x)
                    self.assertGreater(sorted(g)[-1]-sorted(g)[-2], 1e-8)
                    self.assertEqual(order[other.top_expert(x)], model.top_expert(x))
                    self.close(model.train_step(x, target, .02, aux),
                               other.train_step(x, target, .02, aux))
                    expected = permute(model, order)
                    for key in ('W', 'b', 'R', 'c'):
                        self.close(getattr(expected, key), getattr(other, key))
                    updates += 1
        self.assertEqual(updates, 480)

    def test_common_router_shift_and_zero_sum_router_updates(self):
        rng = random.Random(190)
        for case in range(24):
            d, experts = 1 + case % 6, 2 + case % 4
            for aux in (0., .05):
                model = MoE(d, experts, rng)
                other = copy.deepcopy(model)
                shift = [rng.uniform(-.5, .5) for _ in range(d)]
                bias = rng.uniform(-.5, .5)
                for e in range(experts):
                    other.R[e] = [a+b for a, b in zip(other.R[e], shift)]
                    other.c[e] += bias
                for step in range(6):
                    x = [rng.uniform(-1, 1) for _ in range(d)]
                    target = rng.uniform(-2, 2)
                    g = model.gates(x)
                    self.assertGreater(sorted(g)[-1]-sorted(g)[-2], 1e-8)
                    self.close(g, other.gates(x))
                    before = [sum(row[j] for row in model.R) for j in range(d)]
                    before_bias = sum(model.c)
                    self.close(model.train_step(x, target, .03, aux),
                               other.train_step(x, target, .03, aux))
                    self.close(before, [sum(row[j] for row in model.R) for j in range(d)])
                    self.close(before_bias, sum(model.c))
                    self.close(model.W, other.W)
                    self.close(model.b, other.b)
                    for e in range(experts):
                        self.close([a+b for a, b in zip(model.R[e], shift)], other.R[e])
                        self.close(model.c[e]+bias, other.c[e])

    def test_exact_and_rounded_ties_do_not_preserve_sparse_relabeling(self):
        for tiny in (0., math.ulp(0.)):
            model = MoE(1, 2, random.Random(0))
            model.W = [[0.], [0.]]
            model.b = [-2., 6.]
            model.R = [[0.], [0.]]
            model.c = [0., tiny]
            self.assertEqual(model.gates([0.]), [.5, .5])
            other = permute(model, [1, 0])
            self.assertEqual(model.top_expert([0.]), 0)
            self.assertEqual(other.top_expert([0.]), 0)
            self.assertNotEqual(model.predict_top1([0.]), other.predict_top1([0.]))
            self.close(model.predict_soft([0.]), other.predict_soft([0.]))
            # Selection is the lowest index of maximal rounded gate probability,
            # not a mathematical/logit argmax when softmax erases a tiny gap.
            if tiny:
                self.assertGreater(model.c[1], model.c[0])


if __name__ == '__main__':
    unittest.main()
