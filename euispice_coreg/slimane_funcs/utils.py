from typing import Union, Literal, Tuple
from astropy.io import fits
from astropy.io.fits import HDUList, ImageHDU, PrimaryHDU
import astropy.units as u
from sunpy.map import Map
from sunpy.map.mapbase import GenericMap
from pathlib import PosixPath, WindowsPath, Path
import numpy as np
from datetime import datetime
from collections.abc import Iterable
from Analyses_methods import to_submap
from itertools import product
import numpy as np

def find_file(
    name: Union[str, Path],
    base_path: Union[str, Path] = Path("/archive/SOLAR-ORBITER/SPICE/fits/")
) -> Path:
    """
    Constructs the full file path to a SPICE FITS file from its name
    using the expected archive directory structure.

    Parameters
    ----------
    name : str or Path
        The filename of the SPICE FITS file. Must follow the convention that includes
        the date at index 3 when split by '_', and level in index 1.

    base_path : str or Path, optional
        The root directory of the archive. Defaults to the standard SPICE FITS archive path.

    Returns
    -------
    Path
        The constructed full path to the file.
    """

    # Extract date from filename
    date = name.split('_')[3]  # format: YYYYMMDD...
    year = date[:4]
    month = date[4:6]
    day = date[6:8]

    # Determine data level (e.g., level2, from "L2")
    level = "level" + name.split('_')[1][1]

    # Construct and return full path
    path = Path(base_path) / level / year / month / day / name
    return path

def get_data_object(
    any: Union[str, Path, HDUList, GenericMap, ImageHDU, PrimaryHDU],
    as_object: Union[
        HDUList,
        GenericMap,
        ImageHDU,
        PrimaryHDU,
        Literal['hdul', 'map', 'hdu']
    ] = 'map'
) -> Union[HDUList, GenericMap, ImageHDU, PrimaryHDU, list[GenericMap]]:
    """
    Convert various representations of solar FITS data (path, HDUList, GenericMap, HDUs)
    into one of: HDUList, GenericMap, or HDU (ImageHDU or PrimaryHDU).

    Parameters
    ----------
    any : str, Path, HDUList, GenericMap, ImageHDU, or PrimaryHDU
        The input data source. Can be a path to a FITS file, an opened HDUList, a Map object,
        or a single HDU (ImageHDU/PrimaryHDU).

    as_object : Union[HDUList, GenericMap, ImageHDU, PrimaryHDU, Literal['hdul', 'map', 'hdu']]
        The target output format:
        - 'hdul' or HDUList: return a full HDUList (all extensions)
        - 'map' or GenericMap: return a SunPy Map or list of Maps
        - 'hdu' or ImageHDU/PrimaryHDU: return the first HDU

    Returns
    -------
    Depends on `as_object`: an HDUList, a list of GenericMaps, a single GenericMap, or a single HDU.

    Raises
    ------
    TypeError if conversion is invalid or unsupported.
    ValueError if the input FITS file has no HDUs.
    """

    # --- Case 1: Input is a file path ---
    if isinstance(any, (str, PosixPath, WindowsPath)):
        if (isinstance(as_object, str) and as_object.lower() == 'hdul') or as_object is HDUList:
            return fits.open(any)

        elif (isinstance(as_object, str) and as_object.lower() == 'map') or as_object is GenericMap:
            return Map(any)

        elif (isinstance(as_object, str) and as_object.lower() == 'hdu') or as_object in (ImageHDU, PrimaryHDU):
            with fits.open(any) as hdul:
                if len(hdul) == 0:
                    raise ValueError("The FITS file does not contain any HDUs.")
                if len(hdul) > 1:
                    print("Warning: The FITS file contains multiple HDUs. Returning the first one.")
                return hdul[0]

    # --- Case 2: Input is an HDUList ---
    elif isinstance(any, HDUList):
        if (isinstance(as_object, str) and as_object.lower() == 'hdul') or as_object is HDUList:
            return any

        elif (isinstance(as_object, str) and as_object.lower() == 'map') or as_object is GenericMap:
            res = []
            for ind, hdu in enumerate(any):
                if isinstance(hdu, (ImageHDU, PrimaryHDU)):
                    try:
                        res.append(get_data_object(hdu, as_object='map'))
                    except Exception as e:
                        print(f"Warning: Failed to convert HDU[{ind}] to Map. Reason: {e}")
            return res  # list of Maps

        else:
            raise TypeError("Invalid target type: expected 'hdul', 'map', or 'hdu'.")

    # --- Case 3: Input is a SunPy GenericMap ---
    elif isinstance(any, GenericMap):
        if (isinstance(as_object, str) and as_object.lower() == 'map') or as_object is GenericMap:
            return any

        elif (isinstance(as_object, str) and as_object.lower() == 'hdul') or as_object is HDUList:
            raise TypeError("GenericMap object cannot be converted to HDUList.")

        elif (isinstance(as_object, str) and as_object.lower() == 'hdu') or as_object in (ImageHDU, PrimaryHDU):
            raise TypeError("GenericMap object cannot be converted to a single HDU.")

        else:
            raise TypeError("Invalid target type: expected 'hdul', 'map', or 'hdu'.")

    # --- Case 4: Input is a single HDU (ImageHDU or PrimaryHDU) ---
    elif isinstance(any, (ImageHDU, PrimaryHDU)):
        if (isinstance(as_object, str) and as_object.lower() == 'hdu') or as_object in (ImageHDU, PrimaryHDU):
            return any

        elif (isinstance(as_object, str) and as_object.lower() == 'map') or as_object is GenericMap:
            return Map(any.data, any.header)

        elif (isinstance(as_object, str) and as_object.lower() == 'hdul') or as_object is HDUList:
            raise TypeError("ImageHDU or PrimaryHDU cannot be converted to HDUList directly.")

        else:
            raise TypeError("Invalid target type: expected 'hdul', 'map', or 'hdu'.")

    # --- Catch-all for invalid input types ---
    else:
        raise TypeError("Unsupported input type. Expected path, HDUList, GenericMap, ImageHDU, or PrimaryHDU.")


def crop_FSI_to_SPICE_L2(
    SPICE_hdu: Union[ImageHDU, PrimaryHDU],
    FSI_174_path: Union[str, Path, GenericMap],
    FSI_304_path: Union[str, Path, GenericMap],
    expend: u.Quantity = 100 * u.arcsec,
    saving_dir: Path = Path("./tmp/"),
    verbose: int = 0
) -> Tuple[Path, Path]:
    """
    Crop FSI 174 Å and 304 Å images to match the field-of-view (FOV) of a SPICE L2 observation.

    Parameters
    ----------
    SPICE_hdu : astropy.io.fits.HDUList or str or Path
        SPICE level 2 data, can be a path to a FITS file, or an already loaded HDUList.

    FSI_174_path : str or Path
        Path to the FSI 174 Å full-frame image.

    FSI_304_path : str or Path
        Path to the FSI 304 Å full-frame image.

    expend : `~astropy.units.Quantity`, optional
        Amount by which to expand the crop region (in arcseconds).
        This will be scaled according to SPICE pixel scale (CDELTs).

    saving_dir : Path, optional
        Directory in which to save the cropped maps. Default is `./tmp/`.

    verbose : int, optional
        Verbosity level. If >= 1, print where the maps are saved.

    Returns
    -------
    (Path, Path)
        Paths to the saved cropped FSI 174 and 304 maps.
    """

    # Load FSI maps from FITS files
    FSI_174_map = Map(FSI_174_path)
    FSI_304_map = Map(FSI_304_path)

    # Convert SPICE HDU into a SunPy Map
    SPICE_map = get_data_object(SPICE_hdu, as_object='map')

    # Convert expansion from arcsec to data units using SPICE pixel scale
    SPC_CDELTs = np.array([
        SPICE_map.meta['CDELT1'],
        SPICE_map.meta['CDELT2']
    ])
    expend = SPC_CDELTs * expend  # result in arcsec * arcsec/pixel = pixels

    # Crop both FSI images to the SPICE FOV with optional margin
    cropped_174 = to_submap(
        target_map=FSI_174_map,
        source_map=SPICE_map,
        expand=expend
    )
    cropped_304 = to_submap(
        target_map=FSI_304_map,
        source_map=SPICE_map,
        expand=expend
    )

    # Create output directory if it doesn't exist
    saving_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamped filenames to avoid overwrite
    now_time = str(datetime.now()).replace(":", "-").replace(" ", "_")
    cropped_174_path = saving_dir / f"cropped_{FSI_174_path.stem}_{now_time}.fits"
    cropped_304_path = saving_dir / f"cropped_{FSI_304_path.stem}_{now_time}.fits"

    # If a file with the same name exists (e.g., in parallel runs), regenerate timestamp
    while cropped_174_path.exists() or cropped_304_path.exists():
        now_time = str(datetime.now()).replace(":", "-").replace(" ", "_")
        cropped_174_path = saving_dir / f"cropped_{FSI_174_path.stem}_{now_time}.fits"
        cropped_304_path = saving_dir / f"cropped_{FSI_304_path.stem}_{now_time}.fits"

    # Save cropped maps
    cropped_174.save(cropped_174_path)
    cropped_304.save(cropped_304_path)

    # Inform the user if verbose
    if verbose >= 1:
        print(f"✅ Saved cropped FSI 304 map to: {cropped_304_path}")
        print(f"✅ Saved cropped FSI 174 map to: {cropped_174_path}")

    return cropped_174_path, cropped_304_path


def closest_FSI_to_SPICEL2_archive(
    SPICE_L2: Union[str, Path, HDUList, GenericMap],
    archive_location: Path = Path("/archive/SOLAR-ORBITER/EUI/data_internal/L2/"),
    verbose: int = 0
) -> Tuple[Path, Path]:
    """
    Finds the closest-in-time EUI FSI 174 Å and 304 Å full-frame images to a given SPICE L2 observation.

    Parameters
    ----------
    SPICE_L2 : str, Path, HDUList, or GenericMap
        The SPICE L2 data or its path, from which the observation time will be extracted.

    archive_location : Path, optional
        Root directory of the local EUI FSI L2 archive.
        Expected folder structure: YYYY/MM/DD/*.fits

    verbose : int, optional
        Verbosity level. 0 = silent, 1 = key messages, 2 = debug-level output.

    Returns
    -------
    (Path, Path)
        The file paths of the closest FSI 174 Å and 304 Å full-frame images.
    """

    # Load SPICE data and convert it to a SunPy Map (or list of maps)
    SPICE_L2_maps = get_data_object(SPICE_L2, as_object='map')
    
    # If the result is iterable (e.g. list of maps), take the first one
    if isinstance(SPICE_L2_maps, Iterable):
        SPICE_L2_map = SPICE_L2_maps[0]
    else:
        SPICE_L2_map = SPICE_L2_maps

    archive_location = Path(archive_location)
    if verbose >= 2:
        print(f"Archive location: {archive_location}")

    # Extract average observation time from SPICE metadata
    date = np.datetime64(SPICE_L2_map.meta['date-avg'])
    if verbose >= 1:
        print(f"Searching for FSI files closest to {date}")

    # Extract year/month/day to construct folder path
    day   = date.astype(object).day
    month = date.astype(object).month
    year  = date.astype(object).year
    folder = archive_location / f"{year:04d}/{month:02d}/{day:02d}/"

    if verbose >= 2:
        print(f"Searching for FSI files in {folder}")

    # Get all matching FSI files in that folder (excluding "short" exposures)
    EUI_FSI_174_files = np.array(
        [i for i in folder.glob("*fsi174*.fits") if "short" not in i.name],
        dtype=object
    )
    EUI_FSI_304_files = np.array(
        [i for i in folder.glob("*fsi304*.fits") if "short" not in i.name],
        dtype=object
    )

    if verbose >= 2:
        print(f"Found {len(EUI_FSI_174_files)} FSI 174 files and {len(EUI_FSI_304_files)} FSI 304 files")

    # Extract observation datetime from the filename (format: YYYYMMDDTHHMMSSfff)
    EUI_FSI_174_dates = np.vectorize(
        lambda x: np.datetime64(datetime.strptime(x.name.split("_")[3], "%Y%m%dT%H%M%S%f"))
    )(EUI_FSI_174_files)
    
    EUI_FSI_304_dates = np.vectorize(
        lambda x: np.datetime64(datetime.strptime(x.name.split("_")[3], "%Y%m%dT%H%M%S%f"))
    )(EUI_FSI_304_files)

    # Find the file with the closest timestamp to the SPICE map
    closest_FSI_174 = EUI_FSI_174_files[np.abs(EUI_FSI_174_dates - date).argmin()]
    closest_FSI_304 = EUI_FSI_304_files[np.abs(EUI_FSI_304_dates - date).argmin()]

    if verbose >= 1:
        print(f"Closest FSI 174: {closest_FSI_174.name} at {EUI_FSI_174_dates[np.abs(EUI_FSI_174_dates - date).argmin()]}")
        print(f"Closest FSI 304: {closest_FSI_304.name} at {EUI_FSI_304_dates[np.abs(EUI_FSI_304_dates - date).argmin()]}")

    return closest_FSI_174, closest_FSI_304

def get_first_n_neighbors(point, n=10, step=1):
    """
    Return the first `n` neighbors of a point in ND space, expanding outward
    based on Euclidean distance. Diagonal neighbors included.

    Parameters:
    - point: Tuple[int or float], the center coordinate in N-D space.
    - n: int, number of neighbors to return.
    - step: scalar or list/array matching the dimension of `point`.

    Returns:
    - List[Tuple]: list of `n` neighbor coordinates (tuples).
    """
    ndim = len(point)
    point = np.array(point)

    # Step as ND-array
    if isinstance(step, (int, float)):
        step_array = np.ones(ndim) * step
    else:
        step_array = np.array(step)
        if step_array.shape != point.shape:
            raise ValueError("Step must be scalar or same shape as point.")

    neighbors = set()
    radius = 1

    while len(neighbors) < n:
        # All offset combinations within current radius cube
        for offset in product(range(-radius, radius+1), repeat=ndim):
            if all(o == 0 for o in offset):
                continue  # skip center point
            offset_arr = np.array(offset) * step_array
            neighbor = tuple(point + offset_arr)
            neighbors.add(neighbor)
        radius += 1

    # Sort by distance and return the first `n`
    sorted_neighbors = sorted(neighbors, key=lambda x: np.linalg.norm(np.array(x) - point))
    return sorted_neighbors[:n]

def count_touching_hypercubes(ndim: int) -> int:
    """
    Count the number of neighboring hypercubes (excluding the point itself)
    that touch a given point in an N-dimensional grid.

    Parameters
    ----------
    ndim : int
        Number of dimensions.

    Returns
    -------
    int
        Number of unit hypercubes that touch the point (including diagonals).
    """
    if ndim < 0:
        raise ValueError("Number of dimensions must be >= 0")
    
    return 3**ndim - 1  # 3^n - 1 excludes the point itself