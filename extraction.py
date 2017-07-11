"""Extractor class used to get the parts of speech."""
import nltk


class Extractor(object):
    """Extractor class."""

    def __init__(self, url = None):
        """Init method for the object."""
        if not url:
            raise Exception('url is required')

        self.url = url
        self.places = []

    def find_entities(self):
        """Method used to extract the parts of speech that might be places."""
        text = nltk.word_tokenize(self.url)
        text_tags = nltk.pos_tag(text)
        # might make sense to move this inside the Place Context object
        # to allow for a fuzzier search;
        # ie: what if the city name is lowercased?
        nes = nltk.ne_chunk(text_tags)

        for ne in nes:
            if type(ne) is nltk.tree.Tree:
                if ne.label() == 'GPE' or \
                   ne.label() == 'PERSON' or \
                   ne.label() == 'ORGANIZATION':
                    self.places.append(' '.join([i[0] for i in ne.leaves()]))
