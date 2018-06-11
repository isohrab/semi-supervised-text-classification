import os

class Wikiloader():
    def __init__(self, files, title):
        """

        :param files: list of wikipedia files
        """
        # load name of all txt files
        self.filenames = files
        self.length = -1
        self.title = title


    def __iter__(self):
        for name in self.filenames:
            print("Reading: ", name)
            with open(name, 'r')as f:
                for l in f:
                    if l == '':
                        break
                    else:
                        yield l

        yield l


    def __len__(self):
        if self.length == -1:
            self.length = 0
            for name in self.filenames:
                with open(name, 'r')as f:
                    for l in f:
                        self.length += 1
        print(self.length, " sentences has been found in the ", self.title)
        return self.length
