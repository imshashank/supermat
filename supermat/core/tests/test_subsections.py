import pytest

from supermat.core.utils import is_subsection


@pytest.mark.parametrize(
    "section,subsection,actual",
    [
        ("1.2.0", "1.2.1", True),
        ("1.2.0", "1.2.0", True),
        ("1.2.0", "1.1.1", False),
        ("1.2.0", "1.1.0", False),
        ("1.2", "1.2.1", True),
        ("1.2", "1.2", True),
        ("1.2.0", "1.3.0", False),
        ("1.2.0", "1.3.1", False),
        ("1.0.0", "1.2.1", True),
        ("1.0.0", "1.2.0", True),
        ("1", "1.2.1", True),
    ],
)
def test_is_subsection(section: str, subsection: str, actual: bool):
    assert (
        is_subsection(subsection, section) is actual
    ), f"{subsection} is {'a' if actual else 'not a'} subsection of {section}."
