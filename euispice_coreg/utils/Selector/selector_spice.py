from .selector import Selector
import re


class SelectorSpice(Selector):

    default_base_url = "https://spice.osups.universite-paris-saclay.fr/spice-data"

    def __init__(self, release=5.0, level=2, base_url=None, release_dict=None, level_dict=None, 
                 year_suffix="", month_suffix="", day_suffix="", verbose=1):
        """Class to select SPICE fits files using regex

        Args:
            release (float, optional): release number. Defaults to 5.0.
            level (int, optional): level of the file. Defaults to 2.
            base_url (_type_, optional): base path or url where the files are. Defaults to None.
            release_dict (_type_, optional): dict to match a release number to a folder name in the path. Defaults to None.
            level_dict (_type_, optional): dict to match a level number to a folder name in the path. Defaults to None.
            year_suffix (str, optional): add a suffix to the year if necessary. Defaults to "".
            month_suffix (str, optional): add a suffix to the month if necessary. Defaults to "".
            day_suffix (str, optional): add a suffix to the day if necessary. Defaults to "".
            verbose (int, optional): Defaults to 1.  Level of printing on the terminal. 

        """
        if release_dict is None:
            self.release_dict = {
                "1.0": "release-1.0",
                "2.0": "release-2.0",
                "3.0": "release-3.0",
                "4.0": "release-4.0",
                "5.0": "release-5.0",
                "6.0": "release-6.0",
            }
        else:
            self.release_dict = release_dict

        if level_dict is None:
            self.level_dict = {
                "1": "level1",
                "2": "level2",
            }
        else:
            self.level_dict = level_dict

        if base_url is None:
            base_url = SelectorSpice.default_base_url
        url = base_url + '/' + self.release_dict[str(release)] + '/' + self.level_dict[str(level)]

        super().__init__(release_url_basis=url, 
                         year_suffix=year_suffix, month_suffix=month_suffix, day_suffix=day_suffix, 
                         verbose=verbose)

    def get_regex(self):
        self.re_filename = re.compile(
            r'''
            solo
            _(?P<level>L[123])
            _spice
                (?P<concat>-concat)?
                -(?P<slit>[wn])
                -(?P<type>(ras|sit|exp))
                (?P<db>-db)?
                (?P<int>-int)?
            _(?P<time>\d{8}T\d{6})
            _(?P<version>V\d{2})
            _(?P<SPIOBSID>\d+)
            -(?P<RASTERNO>\d+)
            (?P<DR>-DR\d{1})?
            (?P<ext>\..*)
            ''',
            re.VERBOSE
            )