"""Main entry point for the library."""
import nltk

from .extraction import Extractor
from .places import PlaceContext

# download all the basic nltk toolkit
nltk.downloader.download('maxent_ne_chunker')
nltk.downloader.download('words')
nltk.downloader.download('treebank')
nltk.downloader.download('maxent_treebank_pos_tagger')
nltk.downloader.download('punkt')
nltk.downloader.download('averaged_perceptron_tagger')


def get_place_context(text=None):
    """Wrapper function that delivers the locations in a text."""
    e = Extractor(text=text)
    e.find_entities()

    pc = PlaceContext(e.places)
    pc.set_countries()
    pc.set_regions()
    pc.set_cities()
    pc.set_other()

    return pc
