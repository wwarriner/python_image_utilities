from pathlib import Path, PurePath
from typing import Any, Iterable, List, Optional, Union

PathLike = Union[Path, PurePath, str]


def get_contents(
    folder: PathLike, ext: Optional[PathLike] = None, recursive: bool = False
):
    """Returns the file contents of the supplied folder as a list of PurePath. The
    optional ext argument can be used to filter the results to a single
    extension. Can also be used recursively.
    """
    glob = _create_glob(ext)
    if recursive:
        glob = str(PurePath("**") / glob)
    contents = list(Path(folder).glob(glob))
    contents = [PurePath(c) for c in contents]
    contents.sort()
    return contents


def generate_file_names(
    base_name_parts: Iterable[Any],
    ext: str,
    indices: Optional[Iterable[Any]] = None,
    delimiter: str = "_",
    folder: Optional[PathLike] = None,
) -> List[PurePath]:
    """Generates a list of file names from name parts and indices. The
    name parts are joined by a delimiter into a base name. The indices are then
    joined to the base name by the delimiter forming a list of file names. The
    extension is appended with a dot. If a folder is supplied, it is prepended
    to the base name. The number of file names is equal to the number of
    indices. If indices is not supplied, one file name will be returned.

    NOTE: It is not recommended to use this function to join paths! Use the
    built-in pathlib module instead, and supply it as the optional folder
    instead.

    NOTE: This function does not check filenames for validity. That problem is
    really hard. See e.g. https://stackoverflow.com/q/9532499/4132985

    Arguments:

    "base_name_parts": An iterable of objects that can be converted to string
    using str(). These will be joined together using the delimiter to form a
    base name.

    "ext": A string representation of a file extension. Does not need leading
    dot (".") character.

    "indices": An iterable of objects that can be converted to string using
    str(). These will be appended to the base name to produce file names, one at
    a time. If not supplied, only one file name will be returned.

    "delimiter": A string used to join the base name parts and append the
    indices.

    "folder": A PurePath object to a folder location. This is not checked.

    Returns:

    A list of PurePath file names created by joining the folder, base name
    parts, indices (one per file name), and extension, in that order. OS-level
    validity of the resulting paths is not checked!
    """
    base_name_parts = [str(part) for part in base_name_parts]
    base_name = delimiter.join(base_name_parts)
    ext = str(_normalize_ext(ext))
    if indices is not None:
        indices = [str(i) for i in indices]
        file_names = [delimiter.join([base_name, index]) + ext for index in indices]
    else:
        file_names = [base_name + ext]
    file_names = [PurePath(f) for f in file_names]
    if folder is not None:
        file_names = [PurePath(folder) / fn for fn in file_names]
    return file_names


def _create_glob(ext: Optional[PathLike] = None):
    if ext is None:
        return "*"
    else:
        return "*{}".format(_normalize_ext(str(ext)))


def _normalize_ext(ext: str):
    assert ext is not None
    if ext == "":
        return ext
    else:
        return "." + ext.lstrip(".")
