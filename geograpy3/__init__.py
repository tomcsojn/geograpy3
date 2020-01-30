import nltk

from geograpy3.extraction import Extractor
from geograpy3.places import PlaceContext

# download all the basic nltk toolkit


def get_place_context(url = None, text = None):
    e = Extractor(url = url, text = text)
    e.find_entities()

    pc = PlaceContext(e.places)
    pc.set_countries()
    pc.set_regions()
    pc.set_cities()
    pc.set_other()

    return pc
