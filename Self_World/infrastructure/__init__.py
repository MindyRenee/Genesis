"""Infrastructure — persistence, records, and support utilities.

Modules that provide storage, configuration, reporting, and reference
services to the whole system but are not brain anatomy. These live
outside the lobe packages (frontal_lobe, parietal_lobe, temporal_lobe,
occipital_lobe) and the subcortical packages (brainstem, limbic_system,
thalamus, cerebellum, basal_ganglia).

    persistence.py           State save/restore across restarts.
    config.py                Runtime configuration for the mind.
    narrative.py             The self-story over time.
    notifications.py         Subcognitive -> cognitive notification queue.
    net.py                   Bounded HTTP plumbing (body + transports).
    vq_codebook.py           Compression of the concept embedding space.
    wordnet_dictionary.py    Lexical-semantic reference.
    _npz_io.py               Atomic, pickle-free .npz artifact writes.

The world-facing records — ``journal.py`` and ``growth_ledger.py``
(Self_World) and ``user_profile.py`` (Outer_World) — live at
their worlds' roots as top-level modules, not inside this package.
"""
