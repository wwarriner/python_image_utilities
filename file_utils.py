from pathlib import Path


def get_contents(folder, ext=None):
    """Returns the contents of the supplied folder as a list of Paths. The
    optional ext argument can be used to filter the results to a single
    extension.
    """
    return list(Path(folder).glob(create_glob(ext)))


def create_glob(ext=None):
    if ext is None:
        return "*"
    else:
        return "*{}".format(normalize_ext(ext))


def normalize_ext(ext):
    return "." + str(ext).lstrip(".")
