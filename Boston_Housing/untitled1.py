for key in data_dict:
    if data_dict[key]["exercised_stock_options"] == "NaN":
        data_dict[key]["exercised_stock_options"] = 10000000000000000000
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

exercised_stock_options = [item["exercised_stock_options"] for k, item in data_dict.iteritems() if not item["exercised_stock_options"] == "NaN"]
        

        
        

exercised_stock_options = [item["salary"] for k, item in data_dict.iteritems() if not item["salary"] == "NaN"]
        
