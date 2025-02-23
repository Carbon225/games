import flax.nnx as nnx

from .common import (
    ConvolutionalBlock,
    ResidualBlock,
    NetworkOutputs,
    ValueHead,
    PolicyHead,
    ConvActionHead,
    NetworkVariables,
)


class ChessNetwork(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.height = 8
        self.width = 8
        self.in_features = 119
        self.conv_features = 128
        self.policy_features = 128
        self.action_planes = 73

        self.blocks = (
            [ConvolutionalBlock(self.in_features, self.conv_features, rngs=rngs)] +
            [ResidualBlock(self.conv_features, self.conv_features, rngs=rngs) for _ in range(9)]
        )

        self.value_head = ValueHead(
            self.height,
            self.width,
            self.conv_features,
            rngs=rngs)

        self.policy_head = PolicyHead(
            self.conv_features,
            self.policy_features,
            ConvActionHead(
                self.policy_features,
                self.action_planes,
                rngs=rngs),
            rngs=rngs)

    def __call__(self, x) -> NetworkOutputs:
        for block in self.blocks:
            x = block(x)
        v = self.value_head(x)
        pi = self.policy_head(x)
        return NetworkOutputs(pi=pi, v=v)

    def split(self):
        graphdef, params, state = nnx.split(self, nnx.Param, nnx.BatchStat)
        return NetworkVariables(graphdef=graphdef, params=params, state=state)
