# Custom model definitions for MNIST.
# The standard FasterKAN Sequential model used here requires no custom classes,
# but this file must exist so that get_locals(custom_model) works in train/test scripts.

from kan_utils.models import FasterKAN, FasterKANLayer
