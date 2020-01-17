import unittest

from file_utils import *


class Test(unittest.TestCase):
    def setUp(self):
        self.base_path = PurePath("test") / "file_utils_test"
        self.a = self.base_path / "a"
        self.dota = self.base_path / ".a"
        self.atxt = self.base_path / "a.txt"
        self.sub = self.base_path / "sub"
        self.suba = self.sub / "a"
        self.subatxt = self.sub / "a.txt"

    def test_get_contents(self):
        contents = get_contents(self.base_path)
        self.assertEqual(len(contents), 4)
        self.assertEqual(contents[0], self.dota)
        self.assertEqual(contents[1], self.a)
        self.assertEqual(contents[2], self.atxt)
        self.assertEqual(contents[3], self.sub)

        contents = get_contents(self.base_path, ext="txt")
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0], self.atxt)

        contents = get_contents(self.base_path, ext=".txt")
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0], self.atxt)

        contents = get_contents(self.base_path, recursive=True)
        self.assertEqual(len(contents), 6)
        self.assertEqual(contents[0], self.dota)
        self.assertEqual(contents[1], self.a)
        self.assertEqual(contents[2], self.atxt)
        self.assertEqual(contents[3], self.sub)
        self.assertEqual(contents[4], self.suba)
        self.assertEqual(contents[5], self.subatxt)

        contents = get_contents(self.base_path, ext="txt", recursive=True)
        self.assertEqual(len(contents), 2)
        self.assertEqual(contents[0], self.atxt)
        self.assertEqual(contents[1], self.subatxt)

    def test_generate_file_names(self):
        BASE_NAME_PARTS = ["file", 1]
        RESULT = [PurePath("file_1.txt")]
        names = generate_file_names(BASE_NAME_PARTS, "txt")
        self.assertEqual(names, RESULT)

        names = generate_file_names(BASE_NAME_PARTS, ".txt")
        self.assertEqual(names, RESULT)

        RESULT = [PurePath("file-1.txt")]
        names = generate_file_names(BASE_NAME_PARTS, ".txt", delimiter="-")
        self.assertEqual(names, RESULT)

        INDICES = list(range(1, 4))
        RESULT = ["file_1_1.txt", "file_1_2.txt", "file_1_3.txt"]
        RESULT = [PurePath(r) for r in RESULT]
        names = generate_file_names(BASE_NAME_PARTS, ".txt", indices=INDICES)
        self.assertEqual(names, RESULT)

        FOLDER = PurePath("folder")
        RESULT = [FOLDER / r for r in RESULT]
        names = generate_file_names(
            BASE_NAME_PARTS, ".txt", indices=INDICES, folder=FOLDER
        )
