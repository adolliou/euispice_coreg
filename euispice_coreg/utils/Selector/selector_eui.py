import os.path
from .selector import Selector
from urllib.parse import urljoin
import re


class SelectorEui(Selector):
    default_base_url = "https://www.sidc.be/EUI/data/releases"

    def __init__(self, release=6.0, level=2, base_url=None, release_dict=None, level_dict=None, 
                 year_suffix="", month_suffix="", day_suffix="", verbose=1):
        """

        :param release: Release number (e.g. 6.0)
        :param level: level of the data (e.g. 2)
        :param base_url: url or path where the release folders are located. If none, then set to the default_base_url
        :param release_dict: dict containing the names of each release folders. Change if needed
        :param level_dict: dict containing the names of each level folders. Change if needed.
        :param year_suffix: Suffix in the year index of the path
        :param month_suffix: Suffix in the month index of the path
        :param day_suffix: Suffix in the day index of the path
        :param verbose: Level of printing on the terminal. 

        """
        if release_dict is None:
            self.release_dict = {
                "1.0": "202012_release_1.0",
                "2.0": "202103_release_2.0",
                "3.0": "202107_release_3.0",
                "4.0": "202112_release_4.0",
                "5.0": "202204_release_5.0",
                "6.0": "202301_release_6.0",

            }
        else:
            self.release_dict = release_dict
        if level_dict is None:
            self.level_dict = {
                "1": "L1",
                "2": "L2",
                "3": "L3",
            }
        else:
            self.level_dict = level_dict

        if base_url is None:
            base_url = SelectorEui.default_base_url
        url = base_url + '/' + self.release_dict[str(release)] + '/' + self.level_dict[str(level)]
        super().__init__(release_url_basis=url, year_suffix=year_suffix, month_suffix=month_suffix, day_suffix=day_suffix, verbose=verbose)

    def get_regex(self):
        self.re_filename = re.compile(
            r'''solo_(?P<level>L[123])
        _eui
        -(?P<instrument>(fsi|hrieuv|hrilya))
        (?P<filter>(174|304|1216|opn))?
        -image
        (?P<exposure>-short)?
        _(?P<time>\d{8}T\d{6})
        (?P<miliseconds>\d+)?
       _(?P<version>V\d{2}).fits
        ''',
        re.VERBOSE
        )
