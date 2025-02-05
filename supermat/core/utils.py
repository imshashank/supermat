def get_structure(*args: int, min_length: int = 3) -> str:
    """Converts the structural elements into a string with padding."""
    if len(args) < min_length:
        args = args + (0,) * (min_length - len(args))
    return ".".join(map(lambda x: str(x), args))


def split_structure(structure: str) -> tuple[int, ...]:
    """Retrieves the structural elements from structure id."""
    return tuple(map(int, structure.split(".")))


def is_subsection(subsection: str, section: str) -> bool:
    """Determines if one hierarchical section ID is a subsection of another.

    This function compares two hierarchical structure IDs (e.g., "1.2.3" and "1.2")
    to determine if the first is a subsection of the second. A section is considered
    a subsection if it shares all non-zero parts with its parent section and can have
    additional parts after the parent's sequence.

    For example:
        - "1.2.3" is a subsection of "1.2"
        - "1.2.0" is a subsection of "1.2"
        - "1.3" is not a subsection of "1.2"
        - "1.2" is not a subsection of "1.2.3"

    NOTE: Zero parts in the section ID (e.g., "1.2.0") act as wildcards and will match any corresponding part in the
     subsection.

    Args:
        subsection (str): The structure ID to check if it's a subsection
                         (e.g., "1.2.3")
        section (str): The potential parent structure ID to compare against
                      (e.g., "1.2")

    Returns:
        bool: True if 'subsection' is a subsection of 'section', False otherwise
    """
    sub_parts = split_structure(subsection)
    sec_parts = split_structure(section)
    if len(sec_parts) > len(sub_parts):
        return False
    # pad with 0s
    sec_parts = split_structure(get_structure(*sec_parts, min_length=len(sub_parts)))
    return all(
        sub_part >= sec_part if sec_part == 0 else sub_part == sec_part
        for sec_part, sub_part in zip(sec_parts, sub_parts)
    )
