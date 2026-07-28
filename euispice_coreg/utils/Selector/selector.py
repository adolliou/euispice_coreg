import copy
import os
from astropy.time import Time
import requests
from bs4 import BeautifulSoup
import astropy.units as u
import numpy as np
from urllib.parse import urljoin
import yaml
from pathlib import Path
from glob import glob
import warnings


class Selector:
    def __init__(self, release_url_basis, year_suffix = "", month_suffix="", day_suffix="", verbose=1):
        self.release_url_basis = release_url_basis
        self.url_list_all = None
        self.time_list_all = None
        self.re_filename = None
        self.get_regex()
        self._get_list_from_time = None
        self.year_suffix = year_suffix
        self.month_suffix = month_suffix
        self.day_suffix = day_suffix
        if "https" not in release_url_basis:
             # We are dealing with paths instead of URL. Use a different function to find files
            self._get_list_from_time = self._get_paths_list_from_time
        else: 
            self._get_list_from_time = self._get_url_list_from_time
        self.verbose = verbose

    def get_regex(self):
        pass



    def get_url_from_time_interval(self, time1: Time, time2: Time, file_name_str=None):
        if file_name_str is None:
            file_name_str = ""
        if time1 > time2:
            raise ValueError(f"{time2=} must be greater than {time1=}")

        tref = Time(time1.fits[:10] + 'T00:00:00.000')
        url_list_all, time_list_all = self._get_list_from_time(time1, return_time_list=True,
                                                                   file_name_str=file_name_str)

        while tref < time2:
            tref += 1 * u.day
            url_list_, time_list_ = self._get_list_from_time(tref, return_time_list=True,
                                                                    file_name_str=file_name_str)
            url_list_all += url_list_
            time_list_all += time_list_
            if tref >  time2:
                break

        if len(time_list_all) == 0:
            raise ValueError("could not find any FITS file")
        time_list_all = np.array(time_list_all, dtype="object")
        url_list_all = np.array(url_list_all)
        select = np.logical_and(time_list_all >= time1, time_list_all <= time2)
        self.url_list_all = url_list_all[select]
        self.time_list_all = time_list_all[select]

        self._order_time()
        # self.url_list_all = np.array([n[0] for n in url_list_all_], dtype="object")
        return self.url_list_all, self.time_list_all

    def write_txt(self, path_save_txt: str):
        if (self.url_list_all is None) or (self.time_list_all is None):
            raise ValueError("No url_list and time_list available")
        with open(path_save_txt, 'w') as f:
            for ii, file in enumerate(self.url_list_all):
                if ii != (len(self.url_list_all) - 1):
                    f.write(file + "\n")
                else:
                    f.write(file)

    def _order_time(self):

        dt_array    = np.array([(n - self.time_list_all[0]).to("s").value for n in self.time_list_all])
        sort        = np.argsort(dt_array)

        self.url_list_all   = self.url_list_all[sort]
        self.time_list_all  = self.time_list_all[sort]



    def _find_time_from_file(self, fits_file_name):
        m = self.re_filename.match(fits_file_name)
        if m is None:
            raise ValueError(f"could not parse {fits_file_name=}")
        ddd = m.groupdict()
        if 'time' not in ddd.keys():
            raise ValueError(f"could not parse time in {fits_file_name=}")
        return Time(ddd["time"][0:4] + '-' + ddd["time"][4:6] + '-' + ddd["time"][6:8] + 'T'+ ddd["time"][9:11] + ':' + ddd["time"][11:13] + ':' + ddd["time"][13:15])

    def _find_url_from_file(self, fits_file_name):
        time = self._find_time_from_file(fits_file_name)
        return (self.release_url_basis + '\\' + f'{time.ymdhms[0]:04d}' + self.year_suffix + '\\' + f"{time.ymdhms[1]:02d}" + self.month_suffix + '\\' +
                f"{time.ymdhms[2]:02d}") + self.day_suffix

    def _find_url_from_time(self, time: Time):
        url = self.release_url_basis + '\\' + f"{time.ymdhms[0]:04d}" + self.year_suffix + '\\' + f"{time.ymdhms[1]:02d}" + self.month_suffix \
            + '\\' + f"{time.ymdhms[2]:02d}" + self.day_suffix
        return url

    @property
    def release_url_basis(self):
        return self._release_url_basis

    @release_url_basis.setter
    def release_url_basis(self, value):
        self._release_url_basis = value

    def _get_paths_list_from_time(self, time: Time, return_time_list=False, file_name_str=None):

        if file_name_str is None:
            file_name_str = ""
        path_basis      = self._find_url_from_time(time)
        paths_list      = None
        paths_list     =  glob(os.path.join(path_basis, "*"))
        if file_name_str is not None:
            paths_list_     = copy.deepcopy(paths_list)
            paths_list      =  [n for n in paths_list_ if file_name_str in n]
        if (len(paths_list) == 0) and self.verbose > 0:
            warnings.warn("paths_list is empty") 
        if return_time_list:
            time_list = [self._find_time_from_file(os.path.basename(l)) for l in paths_list if (".fits" in os.path.basename(l))]
            return paths_list, time_list
        else:
            return paths_list


    def _get_url_list_from_time(self, time: Time, return_time_list=False, file_name_str=None):

        if file_name_str is None:
            file_name_str = ""
        url = self._find_url_from_time(time)
        req = requests.get(url=url)
        soup = BeautifulSoup(req.text, 'html.parser')
        url_list = [url + '/' + l.get("href") for l in soup.find_all('a')
                    if ((".fits" in l.get("href")) and (file_name_str in l.get("href")))]
        req.close()
        if return_time_list:
            time_list = [self._find_time_from_file(l.get("href")) for l in soup.find_all('a')
                         if ((".fits" in l.get("href")) and (file_name_str in l.get("href")))]
            return url_list, time_list
        else:
            return url_list
