from vision_service.contracts import VisionRule
from vision_service.runtime.dwell import DwellTransition
from vision_service.vision.key_entity_matcher import (
    KeyEntityIdentification,
    KeyEntityMatcher,
    identify_key_entity,
)


async def identify_transition_key_entity(
    *,
    transition: DwellTransition,
    rule: VisionRule,
    matcher: KeyEntityMatcher | None,
) -> KeyEntityIdentification | None:
    if transition.status != "threshold_met":
        return None
    if not rule.key_entities:
        return None
    if not any(sample.crop_bytes is not None for sample in transition.evidence_samples):
        return None

    return await identify_key_entity(
        evidence_samples=transition.evidence_samples,
        key_entities=rule.key_entities,
        matcher=matcher,
    )
