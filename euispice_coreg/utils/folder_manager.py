import numpy as np
import os
from pathlib import Path
from datetime import datetime


class FolderManager(dict):
    def __init__(self, dict_input, list_needed_keys: list):
        super().__init__()
        self._assert_correct_dict(dict_input, list_needed_keys)

    @staticmethod
    def _assert_correct_dict(dict_input, list_needed_keys: list):
        assertion = np.array([(n in dict_input) for n in list_needed_keys], dtype="bool")
        if assertion.sum() != len(list_needed_keys):
            print(f"{dict_input=}")
            print(f"{list_needed_keys=}")
            raise ValueError("Not correct dict keys in the input")


class InputFolderManager(FolderManager):
    """
    Subclass of Foldermanager that manages the input data of a procedure
    """
    def __init__(self, dict_input: dict):
        """Initialize InputFolderManager by setting the non optional and optional keywords.
        optional keywords that are not defined are set to None
        mandatory keywords :
        - data folder
        - in_folder
        - in_subfolder : the files are located in data_folder/in_folder/in_subfolder
        - in_level : level of the input data (ie 2) 
        optional kwywords:
        - in_suffix : str suffix to look for the correct files. example : "*solo*.fits"
        - date_start : start time of the sequence (readable by astropy.Time)
        - date_stop : end time of the sequence
        - name_list_txt : name of a text files in data_folder/in_folder/in_subfolder containing all paths to the 
        files you want to add to the Foldermanager. The date_start, date_end time interval selection will still be applied.

        Args:
            dict_input (dict): _description_
        """
        list_needed_keys = ["data_folder", "sequence_folder_name",
                            "in_folder", "in_subfolder", "in_level"]
        super().__init__(dict_input, list_needed_keys)
        self._initialize_input_folder(dict_input)

    def _initialize_input_folder(self, dict_input):
        path_spice_old_level = os.path.join(dict_input["data_folder"], dict_input["sequence_folder_name"],
                                            dict_input["in_folder"], dict_input["in_subfolder"])
        self["in"] = {"path": path_spice_old_level,
                      "level": dict_input["in_level"]}

        keywords_optionals = [
            "in_suffix", "in_window", "date_start", "date_stop", "in_name_list_txt",    
        ]

        for kk in keywords_optionals:
            kk_without_in = kk.replace("in_", "")

            if kk in dict_input:
                self["in"][kk_without_in] = dict_input[kk]
            else:
                self["in"][kk_without_in] = None     

        if ("in_keyword_select" in dict_input) & ("in_values_select" in dict_input):
            self["in"]["keyword_select"] = dict_input["in_keyword_select"]
            self["in"]["values_select"] = dict_input["in_values_select"]
        else:
            self["in"]["keyword_select"] = None
            self["in"]["values_select"] =  None




class OutputFolderManager(FolderManager):
    def __init__(self, dict_input):
        list_needed_keys = ["data_folder", "sequence_folder_name",
                            "out_folder", "out_subfolder", "out_level"]
        super().__init__(dict_input, list_needed_keys)
        self._initialize_output_folder(dict_input)

    def _initialize_output_folder(self, dict_input):
        path_spice_new_level = os.path.join(dict_input["data_folder"], dict_input["sequence_folder_name"],
                                            dict_input["out_folder"], dict_input["out_subfolder"], )

        Path(os.path.join(dict_input["data_folder"], dict_input["sequence_folder_name"],
                          dict_input["out_folder"])).mkdir(parents=False, exist_ok=True)

        Path(path_spice_new_level).mkdir(parents=False, exist_ok=True)
        self["out"] = {"path": path_spice_new_level,
                       "level": dict_input["out_level"]}
        if "out_window" in dict_input:
            self["out"]["window"] = dict_input["out_window"]



class ResultFolderManager(FolderManager):
    def __init__(self, dict_input):
        list_needed_keys = ["results_folder", "sequence_folder_name", "name_function", ]
        super().__init__(dict_input, list_needed_keys)
        self._initialise_result_folder(dict_input)

    def _initialise_result_folder(self, dict_input):
        results_folder = os.path.join(dict_input["results_folder"])
        Path(results_folder).mkdir(parents=False, exist_ok=True)

        results_folder = os.path.join(results_folder, dict_input["sequence_folder_name"])
        Path(results_folder).mkdir(parents=False, exist_ok=True)

        results_folder = os.path.join(results_folder, dict_input['name_function'])
        Path(results_folder).mkdir(parents=False, exist_ok=True)
        if "personalize_name" in dict_input:
            path_save = os.path.join(results_folder, dict_input["personalize_name"])
            Path(path_save).mkdir(parents=False, exist_ok=True)
        else:
            now = datetime.now()
            dt_string = now.strftime("D_%d_%m_%Y_T_%H_%M_%S")
            path_save = os.path.join(results_folder, dt_string)
            Path(path_save).mkdir(parents=False, exist_ok=True)

        self["res"] = {"path": path_save}
