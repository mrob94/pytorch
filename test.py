import torch

class Foo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return y, x

    @staticmethod
    def backward(ctx, du, dv):
        return dv, du

class Bar(torch.nn.Module):
    def forward(self, x, y):
        x = x.relu()
        y = y.relu()
        z = Foo.apply(x, y)
        return z

x = torch.rand(3, 2, dtype=torch.double)
y = torch.rand(1, 2, dtype=torch.double)

# Generate JIT IR.
traced = torch.jit.trace(Bar(), (x, y))
print(traced.graph)
