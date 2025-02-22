import flax.nnx as nnx

from .common import (
    ConvolutionalBlock,
    ResidualBlock,
    NetworkOutputs,
    ValueHead,
    PolicyHead,
    LinearActionHead,
    NetworkVariables,
)


class ConnectFourNetwork(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.height = 6
        self.width = 7
        self.in_features = 2
        self.conv_features = 32
        self.policy_features = 4

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
            LinearActionHead(
                self.height,
                self.width,
                self.policy_features,
                self.width,
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
