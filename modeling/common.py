import jax
import flax.nnx as nnx
import chex


@chex.dataclass(frozen=True)
class NetworkVariables:
    graphdef: nnx.GraphDef
    params: nnx.Param
    state: nnx.BatchStat

    def merge(self):
        return nnx.merge(self.graphdef, self.params, self.state)

    def train(self):
        model = self.merge()
        model.train()
        return model.split()

    def eval(self):
        model = self.merge()
        model.eval()
        return model.split()


@chex.dataclass(frozen=True)
class NetworkOutputs:
    pi: chex.Array
    v: chex.Array


def flatten2d(x):
    return x.reshape(*x.shape[:-3], -1)


class ConvolutionalBlock(nnx.Module):
    def __init__(self, in_features, out_features, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features, out_features, (3, 3), padding='SAME', rngs=rngs)
        self.bn = nnx.BatchNorm(out_features, rngs=rngs)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = jax.nn.relu(x)
        return x


class ResidualBlock(nnx.Module):
    def __init__(self, in_features, out_features, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_features, out_features, (3, 3), padding='SAME', rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_features, rngs=rngs)
        self.conv2 = nnx.Conv(out_features, out_features, (3, 3), padding='SAME', rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_features, rngs=rngs)

    def __call__(self, x):
        skip = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + skip
        x = jax.nn.relu(x)
        return x


class PolicyHead(nnx.Module):
    def __init__(self, in_features, out_features, action_head, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features, out_features, (1, 1), rngs=rngs)
        self.bn = nnx.BatchNorm(out_features, rngs=rngs)
        self.action_head = action_head

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = jax.nn.relu(x)
        x = self.action_head(x)
        return x


class LinearActionHead(nnx.Module):
    def __init__(self, in_height, in_width, in_features, num_actions, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_height * in_width * in_features, num_actions, rngs=rngs)

    def __call__(self, x):
        x = flatten2d(x)
        x = self.linear(x)
        return x


class ConvActionHead(nnx.Module):
    def __init__(self, in_features, num_actions, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features, num_actions, (1, 1), rngs=rngs)

    def __call__(self, x):
        x = self.conv(x)
        x = flatten2d(x)
        return x


class ValueHead(nnx.Module):
    def __init__(self, in_height, in_width, in_features, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_features, 1, (1, 1), rngs=rngs)
        self.bn = nnx.BatchNorm(1, rngs=rngs)
        self.linear1 = nnx.Linear(in_width * in_height, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 1, rngs=rngs)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = jax.nn.relu(x)
        x = flatten2d(x)
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x).squeeze(-1)
        x = jax.nn.tanh(x)
        return x
