import os.path
from .selector import Selector
from urllib.parse import urljoin


class SelectorEui(Selector):
    default_base_url = "https://www.sidc.be/EUI/data/releases"
    release_dict = {
        "1.0": "202012_release_1.0",
        "2.0": "202103_release_2.0",
        "3.0": "202107_release_3.0",
        "4.0": "202112_release_4.0",
        "5.0": "202204_release_5.0",
        "6.0": "202301_release_6.0",

    }

    level_dict = {
        "1": "L1",
        "2": "L2",
        "3": "L3",
    }

    def __init__(self, release=6.0, level=2, base_url=None):
        if base_url is None:
            base_url = SelectorEui.default_base_url
        url = base_url + '/' + SelectorEui.release_dict[str(release)] + '/' + SelectorEui.level_dict[str(level)]
        super().__init__(release_url_basis=url)
