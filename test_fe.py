import feature_engineering as fe 
import datetime as dt 
start = dt.date(2023,1,1) 
end = dt.date(2024,1,1) 
macro = fe.get_macro_and_sector_data(start, end) 
print(macro.head()) 
