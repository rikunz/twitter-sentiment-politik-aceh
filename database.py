import pandas as pd

class Database:
    def __init__(self, data: str):
        self.data = pd.read_csv(data, header=0, names=["text"])

    def get_random_row(self) -> str:
        """Returns a random row from the DataFrame as a string."""
        random_row = self.data.sample(n=1)
        return random_row['text'].values[0] if 'text' in random_row else str(random_row)
    
    def __repr__(self):
        return f"Database with {len(self.data)} rows and {len(self.data.columns)} columns."
    
    def __len__(self):
        """Returns the number of rows in the DataFrame."""
        return len(self.data)
    
    def head(self, n: int = 5):
        """Returns the first n rows of the DataFrame."""
        if n == 0:
            return self.data
        return self.data.head(n)
    
    def exact_filter(self, value:str):
        """Returns rows where the 'text' column matches the exact value."""
        return self.data[self.data['text'].str.fullmatch(value, case=False, na=False)]
    
    def contains_filter(self, value: str):
        """Returns rows where the 'text' column contains the value."""
        return self.data[self.data['text'].str.contains(value, case=False, na=False)]
    
    def regex_filter(self, pattern: str):
        """Returns rows where the 'text' column matches the regex pattern."""
        return self.data[self.data['text'].str.contains(pattern, case=False, na=False, regex=True)]
        
    
    