# Owner(s): ["module: inductor"]

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import config as inductor_config, metrics
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA


@inductor_config.patch(
    {
        "benchmark_kernel": True,
        "triton.unique_kernel_names": True,
    }
)
class LoopOrderingTest(TestCase):
    def do_acc_test(self, f, *args):
        expect = f(*args)
        actual = torch.compile(f)(*args)
        self.assertTrue(same(expect, actual, tol=1e-3))

    def test_for_reordering_reindex(self):
        """
        ComputedBuffer.iter_reoredering_reindex can cause some fusion
        opportunitiies being skipped.

        In this test case, Inductor generates 2 triton kernels before.
        By removing ComputedBuffer.iter_reoredering_reindex, we can fuse those
        two kernels into a single one.
        """

        def f(x, y):
            """
            Add a matmul since inductor may force layout for output.
            """
            return (x.sum(dim=-1) + 1) @ y

        # Make the first 2 dimension not able to merge on purpose so that
        # ComputedBuffer.iter_reoredering_reindex will be updated.
        x = rand_strided([20, 20, 30], [30, 900, 1], device="cuda")
        y = torch.randn(20, 20)

        self.do_acc_test(f, x, y)
        self.assertEqual(1, metrics.generated_kernel_count)


if __name__ == "__main__":
    if HAS_CUDA:
        torch.set_default_device("cuda")
        run_tests()
