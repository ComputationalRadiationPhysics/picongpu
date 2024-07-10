import typeguard
import typing


@typeguard.typechecked
def my_func(a_char: str, num: int) -> typing.List[str]:
    """
    build list with "triangle" of chars

    Accepts a single char and builds a list where the n-th element contains a
    string consisting of n times the given char.
    The list has the given total length.

    This is an example for a section of the PyPIConGPU doc,
    see: :ref:`pypicongpu-misc-apidoc`.
    to show that all Sphinx directives are valid.
    Because this is reStructuredText, (nested) lists require separate
    paragraphs:

    - an item
    - another item

        - nested 1
        - nested 2

    - continuation of top level

    :param a_char: character to use, e.g. "."
    :param num: max, e.g. "3"
    :raises AssertionError: on a_char being not a single character
    :raises AssertionError: if num < 0
    :return: [".", "..", "..."]
    """
    assert 1 == len(a_char)
    assert 0 <= num

    if 0 == num:
        return []

    return my_func(a_char, num - 1) + [num * a_char]
