def get_structure(*args: int, min_length: int = 3) -> str:
    if len(args) < min_length:
        args = args + (0,) * (min_length - len(args))
    return ".".join(map(lambda x: str(x), args))


def split_structure(structure: str) -> tuple[int, ...]:
    return tuple(map(int, structure.split(".")))


def is_subsection(subsection: str, section: str) -> bool:
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
