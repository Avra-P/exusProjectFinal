
import pandas as pd

def load_input_files(Default_Data, ClientBureau_Info, Loan_Information, Client_Information):
    
    """
    Load input data (CSV) files and return them as pandas tables.

    Args:
        DefaultData (str): path string
        ClientBureauInfo (str): path string
        Client_Information (str): path string
        Loan_Information (str): path string

    Returns:
        tuple: A tuple containing four Pandas DataFrames.
    """
    try:
        defaultData = pd.read_csv(Default_Data)
        clientBureauInfo = pd.read_csv(ClientBureau_Info)
        clientInformation = pd.read_csv(Client_Information)
        loanInformation = pd.read_csv(Loan_Information)
        
        defaultData.to_pickle('./data/Default_Data.plk')
        clientBureauInfo.to_pickle('./data/clientBureauInfo.plk')
        clientInformation.to_pickle('./data/clientInformation.plk')
        loanInformation.to_pickle('./data/loanInformation.plk')
        

        return defaultData,clientBureauInfo,clientInformation,loanInformation
    except FileNotFoundError:
        print("One or more input files not found.")
        return None
