"""Infrastructure — persistence, records, and support utilities.

Modules that provide storage, configuration, reporting, and reference
services to the whole system but are not brain anatomy. These live
outside the lobe packages (frontal_lobe, parietal_lobe, temporal_lobe,
occipital_lobe) and the subcortical packages (brainstem, limbic_system,
thalamus, cerebellum, basal_ganglia).

    persistence.py           State save/restore across restarts.
    config.py                Runtime configuration for the mind.
    journal.py               Append-only record of inner life.
    narrative.py             The self-story over time.
    notifications.py         Subcognitive -> cognitive notification queue.
    bug_reporter.py          Noticing problems in its own code.
    growth_ledger.py         Legible tracking of development.
    user_profile.py          What Genesis knows about the human.
    vq_codebook.py           Compression of the concept embedding space.
    wordnet_dictionary.py    Lexical-semantic reference.
    _npz_io.py               Atomic, pickle-free .npz artifact writes.
"""

