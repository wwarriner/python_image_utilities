from itertools import groupby, takewhile
from pathlib import Path, PurePath
from typing import Any, Iterable, List, Optional, Sequence, Union

PathLike = Union[Path, PurePath, str]


# TODO write tests
def append_suffix(path: PathLike, suffix: Union[str, None], delimiter: str = "_"):
    if suffix is None:
        return path
    else:
        p = PurePath(path)
        return p.parent / (delimiter.join([p.stem, suffix]) + p.suffix)


def deduplicate(s: PathLike, delimiter: str):
    out = []
    for key, group in groupby(str(s)):
        if key == delimiter:
            out.append(key)
        else:
            out.append("".join(list(group)))
    return "".join(out)


def get_contents(
    folder: PathLike, ext: Optional[PathLike] = None, recursive: bool = False
) -> List[PurePath]:
    """Returns the file contents of the supplied folder as a list of PurePath. The
    optional ext argument can be used to filter the results to a single
    extension. Can also be used recursively.
    """
    if ext is not None:
        ext = str(ext)
    glob = _create_glob(ext)
    if recursive:
        glob = str(PurePath("**") / glob)
    contents = list(Path(folder).glob(glob))
    contents = [PurePath(c) for c in contents]
    contents.sort()
    return contents


def generate_file_names(
    name: PathLike,
    ext: Optional[str] = None,
    indices: Optional[Iterable[Any]] = None,
    delimiter: str = "_",
    folder: Optional[PathLike] = None,
) -> List[PurePath]:
    """Generates a list of file names from a supplied name and indices. The
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
    if ext is not None:
        ext = str(_normalize_ext(ext))
    else:
        ext = ""
    if indices is not None:
        indices = [str(i) for i in indices]
        names = [delimiter.join([name, index]) + ext for index in indices]
    else:
        names = [str(name) + ext]
    names = [PurePath(f) for f in names]
    if folder is not None:
        names = [PurePath(folder) / fn for fn in names]
    return names


def lcp(*s: PathLike):
    """Returns longest common prefix of input strings.

    c/o: https://rosettacode.org/wiki/Longest_common_prefix#Python:_Functional
    """
    strings = [str(x) for x in s]
    return "".join(ch[0] for ch in takewhile(lambda x: min(x) == max(x), zip(*strings)))


def _create_glob(ext: str = None) -> str:
    if ext is None:
        return "*"
    else:
        return "*{}".format(_normalize_ext(str(ext)))


def _normalize_ext(ext: str) -> str:
    assert ext is not None
    if ext == "":
        return ext
    else:
        return "." + ext.lstrip(".")


class Files:
    def __init__(
        self,
        root_folder: PathLike,
        base_name: str,
        ext: Optional[str] = None,
        delimiter: str = "_",
    ):
        self._root = PurePath(root_folder)
        self._base = base_name
        self._ext = ext
        self._delimiter = delimiter

    @property
    def ext(self):
        return self._ext

    @ext.setter
    def ext(self, value: Optional[str]):
        if value is not None:
            value = _normalize_ext(value)
        self._ext = value

    def __add__(self, suffix: str):
        f = self.copy()
        f._base = f._delimiter.join([f._base, suffix])
        return f

    def __truediv__(self, sub: PathLike):
        f = self.copy()
        f._root = f._root / sub
        return f

    def mkdir(self, *args, **kwargs):
        Path(self._root).mkdir(*args, **kwargs)

    def generate_file_names(self, ext: Optional[str] = None, *args, **kwargs):
        """Generates a list of file names from a supplied name and indices. See
        documentation of free function generate_file_names(). Accepts all inputs
        except name and folder, which are supplied by the class.
        """
        if ext is None:
            ext = self._ext
        return generate_file_names(
            name=self._base,
            folder=self._root,
            ext=ext,
            delimiter=self._delimiter,
            *args,
            **kwargs
        )

    def copy(self):
        return Files(self._root, self._base, self._ext, self._delimiter)
