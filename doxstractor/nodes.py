from __future__ import annotations
from .extractors import BaseExtractor, CategoryExtractor
from typing import Dict, Optional, List
import collections


class Node:
    def __init__(
        self, extractor: BaseExtractor, children: Optional[Dict[str, List[Node]]] = None
    ) -> None:
        """A node is an element of a tree which has one or multiple children. Depending on the results of the extractor,
        it recursively calls all the child nodes corresponding to the result.

        E.g. if you have a `CategoryExtractor` with the categories `rental_contract` and `employment_contract`, you likely
        want to extract different attributes depending on what you are dealing with.

        So you would provide a children dictionary as below:
        ```python
        {
            "rental_contract": [
                LeaseDatesExtractor,
                SizeExtractor,
            ],
            "employment_contract": [
                SalaryExtractor,
                StockOptionsExtractor
            ]
        }
        ```

        Args:
            extractor (BaseExtractor): The extractor at the root of the node.
            children (Optional[Dict[str, List[Node]]], optional): Dictionary where key is a category of the
                root node and value is a list of nodes. Defaults to None.
        """

        self.extractor = extractor
        self.children = children
        self.validate()

    def validate(self):
        """Ensures the node is valid."""
        if self.children:
            # Ensure only a category extractor gets children
            if not (isinstance(self.extractor, CategoryExtractor)):
                raise ValueError("Children are only supported for category extractors")

            # Ensure children are only set for categories which exist in the parent
            if not set(self.children.keys()).issubset(set(self.extractor.categories)):
                raise ValueError(
                    "Every child must be a category of the parent extractor"
                )

            # Ensure all children have unique names
            names = node_names(self)
            duplicates = [
                item for item, count in collections.Counter(names).items() if count > 1
            ]
            if len(duplicates) > 0:
                raise ValueError(
                    f"Duplicate node name. All node names need to be unique. Duplicates are {duplicates}"
                )

    def extract(self, doc_text: str) -> Dict:
        """Recursively runs all extractors

        Args:
            doc_text (str): Document text from which to extract.

        Returns:
            Dict: {node_name: node_result}
        """

        # Run Extraction on own extractor
        result = self.extractor.extract(doc_text)
        result_dict = {self.extractor.name: result}

        if self.children and (result in self.children.keys()):
            # Fetch the children relevant to the extracted category
            # We will assume there is a category since we validated that earlier
            # We also assume that categorical extractors only return valid categories.
            child_list = self.children[result]
            for child_node in child_list:
                result_dict.update(child_node.extract(doc_text))
        return result_dict


def node_names(node: Node):
    names = [node.extractor.name]

    if node.children:
        for child_list in node.children.values():
            for child_node in child_list:
                names.extend(node_names(child_node))
    return names
