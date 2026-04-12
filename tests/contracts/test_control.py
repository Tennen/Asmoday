from vision_service.contracts import EntitySelector


def test_entity_selector_allows_empty_value_for_wildcard_rules() -> None:
    selector = EntitySelector(value="")

    assert selector.value == ""
