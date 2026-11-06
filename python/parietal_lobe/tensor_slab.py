"""Tensor slab — word-layer companion to ``src/daemon/tensor_slab.rs``.

The shared field memory: a named-slab store the daemon and mind
exchange tensors through — ``motor.spatial`` (where to act),
``motor.actions`` (action dispositions), the probe's band powers —
each a typed tensor published under a name, read by whoever needs
it. The field engine's working memory: where cognition writes its
decisions as fields rather than symbols, and the body decodes them.
"""
