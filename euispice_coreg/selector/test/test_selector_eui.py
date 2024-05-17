from astropy.time import Time
from ..selector_eui import SelectorEui


def test_selector_eui():
    t1 = Time("2022-01-18T15:00:00")
    t2 = Time("2022-01-21T00:00:00")

    s = SelectorEui()
    l_url, l_time = s.get_url_from_time_interval(time1=t1, time2=t2, file_name_str="eui-fsi304-image")
    print(l_url)
    print(l_time)

