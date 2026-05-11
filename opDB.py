from sqlalchemy import create_engine, MetaData,text
import pandas as pd

class localsqlite():
    def __init__(self, db_path):
        self.db_path = db_path
    
    def conn(self):
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.connection = self.engine.connect()
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.connection)
        return self
    
    def exesql(self,sql):
        df = pd.read_sql_query(text(sql), self.engine).dropna()
        return df

if __name__ == "__main__":
    dbconn = localsqlite('./sqlite/cre_Drama_Workshop_Groups.sqlite').conn()
    sql = """
    SELECT *
    FROM Bookings B 
    INNER JOIN Customers C ON B.Customer_ID = C.Customer_ID 
    INNER JOIN Stores S ON B.Store_ID = S.Store_ID
    WHERE Customer_Name = 'Blake' AND Store_Name = 'Adan Dinning'
    """
    df = dbconn.exesql(sql)
    print(df)

