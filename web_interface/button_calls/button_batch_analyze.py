from database_handler import MongoDBHandler
from web_interface.button_calls.button_analyze_svg import click_analyze_svg


def click_batch_analyze(distance_type_dataset, distance_value_dataset, distance_calculation, sample_id, smooth_method,
                        smooth_factor, smooth_window, n_samples, duplicate_synonym_checkbox):
    """
        called by button
        calculates all data and stores it in db for all samples

        :param distance_type_dataset:
        :param distance_calculation:
        :param distance_value_dataset:
        :param sample_id:
        :param smooth_method:
        :param smooth_factor:
        :param smooth_window:
        :param n_samples:
        :return:
        """

    # get a database handler
    db_handler = MongoDBHandler("svg_data")
    db_handler.use_collection("svg_raw")

    # iterate over all samples
    id_list = db_handler.list_svg_ids()
    for current_sample_id in id_list:
        click_analyze_svg(distance_type_dataset, distance_value_dataset, distance_calculation, current_sample_id,
                          smooth_method, smooth_factor, smooth_window, n_samples, duplicate_synonym_checkbox)

    # return values of currently selected sample in the dropdown. It is displayed after the calculation is done
    return click_analyze_svg(distance_type_dataset, distance_value_dataset, distance_calculation,
                                                sample_id, smooth_method, smooth_factor, smooth_window, n_samples, duplicate_synonym_checkbox)
