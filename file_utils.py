from pathlib import PurePath, Path


def get_contents(folder, ext=None):
    return list(Path(folder).glob(create_glob(ext)))


def create_glob(ext=None):
    if ext is None:
        return "*"
    else:
        return "*{}".format(ext)
