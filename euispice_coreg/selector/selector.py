import copy
import os
from astropy.time import Time
import requests
from bs4 import BeautifulSoup
import astropy.units as u
import numpy as np
from urllib.parse import urljoin



class Selector:
    def __init__(self, release_url_basis):
        self.release_url_basis = release_url_basis

    @staticmethod
    def _find_time_from_file(fits_file_name):
        a = fits_file_name[fits_file_name.find('image') + 6:21 + fits_file_name.find('image')]
        return Time(a[:4] + "-" + a[4:6] + "-" + a[6:8] + "T" + a[9:11] + ":" + a[11:13] + ":" + a[13:15])

    def _find_url_from_file(self, fits_file_name):
        a = fits_file_name[fits_file_name.find('image') + 6:21 + fits_file_name.find('image')]
        return self.release_url_basis + '/' + a[:4] + '/' + a[4:6] + '/' + a[6:8]

    def _find_url_from_time(self, time: Time):
        url = self.release_url_basis + '/' + f"{time.ymdhms[0]:04d}" + '/' + f"{time.ymdhms[1]:02d}" \
            + '/' + f"{time.ymdhms[2]:02d}"
        return url

    @property
    def release_url_basis(self):
        return self._release_url_basis

    @release_url_basis.setter
    def release_url_basis(self, value):
        self._release_url_basis = value

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

    def get_url_from_time_interval(self, time1: Time, time2: Time, file_name_str=None):
        if file_name_str is None:
            file_name_str = ""
        if time1 > time2:
            raise ValueError(f"{time2=} must be greater than {time1=}")

        tref = Time(time1.fits[:10] + 'T00:00:00.000')
        url_list_all, time_list_all = self._get_url_list_from_time(time1, return_time_list=True,
                                                                   file_name_str=file_name_str)

        while tref < time2:
            tref += 1 * u.day
            if tref < time2:
                url_list_, time_list_ = self._get_url_list_from_time(tref, return_time_list=True,
                                                                     file_name_str=file_name_str)
                url_list_all += url_list_
                time_list_all += time_list_

        time_list_all = np.array(time_list_all, dtype="object")
        url_list_all = np.array(url_list_all, dtype="str")

        select = np.logical_and(time_list_all >= time1, time_list_all <= time2)

        return url_list_all[select], time_list_all[select]

        # time_list.append(copy.deepcopy(tref))
