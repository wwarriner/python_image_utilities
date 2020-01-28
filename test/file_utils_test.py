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

    def test_deduplicate(self):
        s = deduplicate("aa", "_")
        self.assertEqual(s, "aa")

        s = deduplicate("a_a", "_")
        self.assertEqual(s, "a_a")

        s = deduplicate("a__a", "_")
        self.assertEqual(s, "a_a")

        s = deduplicate("aa_", "_")
        self.assertEqual(s, "aa_")

        s = deduplicate("aa__", "_")
        self.assertEqual(s, "aa_")

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
        BASE_NAME = "file_1"
        RESULT = [PurePath("file_1.txt")]
        names = generate_file_names(BASE_NAME, "txt")
        self.assertEqual(names, RESULT)

        names = generate_file_names(BASE_NAME, ".txt")
        self.assertEqual(names, RESULT)

        INDICES = list(range(1, 4))
        RESULT = ["file_1_1.txt", "file_1_2.txt", "file_1_3.txt"]
        RESULT = [PurePath(r) for r in RESULT]
        names = generate_file_names(BASE_NAME, ".txt", indices=INDICES)
        self.assertEqual(names, RESULT)

        FOLDER = PurePath("folder")
        RESULT = [FOLDER / r for r in RESULT]
        names = generate_file_names(BASE_NAME, ".txt", indices=INDICES, folder=FOLDER)
        self.assertEqual(names, RESULT)

    def test_lcp(self):
        self.assertEqual(lcp("interspecies", "interstellar", "interstate"), "inters")
        self.assertEqual(lcp("throne", "throne"), "throne")
        self.assertEqual(lcp("throne", "dungeon"), "")
        self.assertEqual(lcp("cheese"), "cheese")
        self.assertEqual(lcp(""), "")
        self.assertEqual(lcp("prefix", "suffix"), "")
        self.assertEqual(lcp("foo", "foobar"), "foo")

    def test_Files(self):
        BASE_NAME = "file_1"
        FOLDER = "folder"
        INDICES = list(range(1, 4))
        RESULT = ["file_1_1.txt", "file_1_2.txt", "file_1_3.txt"]
        RESULT = [PurePath(r) for r in RESULT]
        RESULT = [FOLDER / r for r in RESULT]
        files_no_ext = Files(FOLDER, BASE_NAME)
        names = files_no_ext.generate_file_names(".txt", indices=INDICES)
        self.assertEqual(names, RESULT)

        files_ext = Files(FOLDER, BASE_NAME, "wrong")
        names = files_ext.generate_file_names(".txt", indices=INDICES)
        self.assertEqual(names, RESULT)

        files_ext = Files("", BASE_NAME, "txt") / FOLDER
        names = files_ext.generate_file_names(indices=INDICES)
        self.assertEqual(names, RESULT)

        files_ext_copy = files_ext.copy()
        names_copy = files_ext_copy.generate_file_names(indices=INDICES)
        self.assertEqual(names_copy, names)

        files_ext_copy = files_ext.copy() / FOLDER
        names_copy = files_ext_copy.generate_file_names(indices=INDICES)
        self.assertNotEqual(names_copy, names)


if __name__ == "__main__":
    unittest.main()
