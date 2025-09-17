# Import Libraries
from functionalities.h3_simplifier import request_processor, first_time_load
from functionalities.requests import fetch_vbox_data_as_dataframe

if __name__ == '__main__':

    # Run the function -> In theory you only need request_processor to simplify the data and create the critical points
    request_processor("137319VBOX")
    # fetch_vbox_data_as_dataframe("139581VBOX").to_csv("139581VBOX.csv", index=False)
    # first_time_load()


