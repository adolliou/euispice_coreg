from astropy.time import Time
from ..selector_eui import SelectorEui


def test_selector_eui_time_interval():
    t1 = Time("2022-01-18T15:00:00")
    t2 = Time("2022-01-21T00:00:00")

    s = SelectorEui(release=6.0, level=2)
    l_url, l_time = s.get_url_from_time_interval(time1=t1, time2=t2, file_name_str="eui-fsi304-image")

    assert "solo_L2_eui-fsi304-image_20220118T150015296_V02.fits" in l_url[0]
    assert "L2/2022/01/20/solo_L2_eui-fsi304-image_20220120T234515252_V02.fits" in l_url[-1]
    assert (len(l_url) == 207)


