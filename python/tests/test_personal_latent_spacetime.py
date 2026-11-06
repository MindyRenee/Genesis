import numpy as np
import pytest
from self.personal_latent_spacetime import LatentDomain, PersonalLatentSpacetime


def test_domains_keep_independent_dimensions_and_metrics() -> None:
    model = PersonalLatentSpacetime(
        dimensions={
            LatentDomain.SELF: 3,
            LatentDomain.INNER: 4,
            LatentDomain.OUTER: 5,
        }
    )
    snapshots = {item.domain: item for item in model.snapshot()}
    assert snapshots[LatentDomain.SELF].dimension == 3
    assert snapshots[LatentDomain.INNER].dimension == 4
    assert snapshots[LatentDomain.OUTER].dimension == 5


def test_self_apex_does_not_relabel_other_domain_coordinates() -> None:
    model = PersonalLatentSpacetime(
        dimensions={
            LatentDomain.SELF: 2,
            LatentDomain.INNER: 2,
            LatentDomain.OUTER: 2,
        }
    )
    apex = model.record(
        LatentDomain.SELF, (0.0, 0.0), time_s=1.0, event_id="self-1"
    )
    model.record(
        LatentDomain.INNER, (1.0, 0.0), time_s=1.0, event_id="inner-1"
    )
    assert model.self_apex() == apex
    with pytest.raises(ValueError, match="explicit latent-space mapping"):
        model.cone(apex, LatentDomain.INNER)


def test_latent_cone_uses_domain_metric_and_propagation_rate() -> None:
    model = PersonalLatentSpacetime(
        dimensions={
            LatentDomain.SELF: 2,
            LatentDomain.INNER: 2,
            LatentDomain.OUTER: 2,
        },
        propagation_rates={
            LatentDomain.SELF: 1.0,
            LatentDomain.INNER: 2.0,
            LatentDomain.OUTER: 1.0,
        },
    )
    model.record(
        LatentDomain.INNER, (0.0, 0.0), time_s=0.0, event_id="inner-past"
    )
    apex = model.record(
        LatentDomain.INNER, (1.0, 0.0), time_s=1.0, event_id="inner-apex"
    )
    relations = model.cone(apex, LatentDomain.INNER)
    by_id = {relation.event_id: relation for relation in relations}
    assert by_id["inner-past"].latent_distance == pytest.approx(1.0)
    assert by_id["inner-past"].reachable


def test_rejects_non_positive_definite_metric() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        PersonalLatentSpacetime(
            dimensions={
                LatentDomain.SELF: 2,
                LatentDomain.INNER: 2,
                LatentDomain.OUTER: 2,
            },
            metrics={
                LatentDomain.SELF: np.diag([1.0, 0.0]),
            },
        )
