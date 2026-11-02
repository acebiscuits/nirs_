"""Example of minimal external cipher plug-in. Demonstration ONLY, not secure encryption.

To use: --plugin crypto.example_plugin:IdentityExample
"""
import numpy as np

class IdentityExample:
    def __init__(self, key, operation='signflip', decrypt=False):
        self.key = key
        self.decrypt = decrypt

    def process(self, index, pcm320):
        return np.asarray(pcm320, dtype=np.float32).copy()
