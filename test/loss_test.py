import unittest

import src.network as n


class LossTest(unittest.TestCase):

    def test_loss_exist(self):
        s = n.Session()
        g = n.Graph()
        g.as_default()
        X = n.Placeholder("X")
        c = n.Placeholder("Y")

        loss = n.Negative(n.ReduceSum(
            n.ReduceSum(n.Multiply(c, n.Log(n.Add(X, n.Variable(initial_value=1e-8, name="Epsilon")))), axis=1)))

        out_loss = s.run(loss, feed_dict={X: n.np.array(
            [[0, 0, 1], [1, 0, 0]]), c: n.np.array([[1, 0, 0], [1, 0, 0]])})

        self.assertLess(out_loss, 10)
        print(out_loss)


if __name__ == '__main__':
    unittest.main(verbosity=2)
