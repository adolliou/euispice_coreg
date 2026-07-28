from euispice_coreg.utils.create_dict_file import create_dict_file, select_time_interval, remove_paths_with_str, select_values_in_header_keywords
from euispice_coreg.utils.folder_manager import ResultFolderManager, InputFolderManager, OutputFolderManager


def prepare_data_dict(files, data_folder,sequence_folder_name, 
                      results_folder=None, 
                      results_subfolder=None,
                      personalize_name=None):

    files["data_folder"] = data_folder
    files["sequence_folder_name"] = sequence_folder_name

    folderman = InputFolderManager(files)
    if results_folder is not None:
        params_results = {
            "results_folder": results_folder,
            "sequence_folder_name": sequence_folder_name,
            "name_function": results_subfolder,
        }
        if personalize_name is not None:
            params_results["personalize_name"] = personalize_name
        folderman["res"] = ResultFolderManager(params_results)["res"]
    if "out_folder" in files:
        folderman["out"] = OutputFolderManager(files)["out"]

    if "remove_str" in files:
        dict_im = remove_paths_with_str(dict_file=dict_im, str_to_remove=dict_im["remove_str"])

    if folderman["in"]["keyword_select"] is not None:  
        dict_spice = select_values_in_header_keywords(dict_file=dict_spice, 
                                                        keyword=folderman["in"]["keyword_select"], 
                                                        values= folderman["in"]["values_select"], 
                                                        window= folderman["in"]["window"],)

    dict_files = create_dict_file(path_instrument=folderman["in"]["path"], suffix=folderman["in"]["suffix"],
                                     window=folderman["in"]["window"], name_list_txt = folderman["in"]["name_list_txt"])
    dict_files = select_time_interval(dict_file=dict_files,
                                         date_start=files["date_start"], date_stop=files["date_stop"])
    
    return dict_files, folderman