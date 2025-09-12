from typing import Dict, List, Any

class DataUtil:
   
    
    
    @staticmethod
    def split_data(data:List,batch_size:int = 16) -> List[List[Any]]:
        """Split the list into List of sublist, ench sublist is of the batch size

        Args:
            data (List): candidate list
            batch_size (int, optional): target batch size. Defaults to 16.

        Returns:
            List[List]: splitted list
        """
        start_index=0
        result=[]
        while start_index<len(data):
            end_index=min(start_index+batch_size,len(data))
            data_chunk=data[start_index:end_index]
            result.append(data_chunk)
            start_index+=batch_size
            
        return result
    

    
    