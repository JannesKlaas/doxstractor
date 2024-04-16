import doxstractor as dxc


def test_graph_creation():
    lease_doc = "lease"
    employment_doc = "employment"
    model = dxc.MockModel()
    root_extractor = dxc.CategoryExtractor(
        name="doc_type",
        query="What type of document is this?",
        categories=["lease", "employment"],
        model=model,
    )

    lease_extractor = dxc.TextExtractor(
        name="text_lease", query="What is the content?", model=model
    )
    employment_extractor = dxc.TextExtractor(
        name="text_employment", query="What is the content?", model=model
    )

    children = {
        "lease": [dxc.Node(lease_extractor)],
        "employment": [dxc.Node(employment_extractor)],
    }

    root_node = dxc.Node(root_extractor, children=children)

    expected_lease_result = {"doc_type": "lease", "text_lease": "lease"}
    expected_employment_result = {
        "doc_type": "employment",
        "text_employment": "employment",
    }

    lease_result = root_node.extract(lease_doc)
    employment_result = root_node.extract(employment_doc)

    assert lease_result == expected_lease_result
    assert employment_result == expected_employment_result
