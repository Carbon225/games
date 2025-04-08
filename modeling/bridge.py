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


class BridgeNetwork(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.multihot_features = 480
        self.height = 4
        self.width = 4
        self.in_features = 32
        self.conv_features = 64
        self.policy_features = 4
        self.num_actions = 38

        self.projection = nnx.Linear(
            self.multihot_features,
            self.height * self.width * self.in_features,
            rngs=rngs
        )

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
                self.num_actions,
                rngs=rngs),
            rngs=rngs)
        
    def __call__(self, x) -> NetworkOutputs:
        x = self.projection(x)
        x = x.reshape(x.shape[0], self.height, self.width, self.in_features)
        for block in self.blocks:
            x = block(x)
        v = self.value_head(x)
        pi = self.policy_head(x)
        return NetworkOutputs(pi=pi, v=v)

    def split(self):
        graphdef, params, state = nnx.split(self, nnx.Param, nnx.BatchStat)
        return NetworkVariables(graphdef=graphdef, params=params, state=state)
