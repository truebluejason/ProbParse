from torch.utils.data import Dataset

from const import PERMITTED_LETTERS


class NameDataset(Dataset):
    def __init__(self, df, name_column, max_name_length=40):
        """
        Args:
            csv_file (string): Path to the csv file WITHOUT labels
            col_name (string): The column name corresponding to the people names that'll be standardized
        """
        if 0 < len(df):
            df = self._clean_dataframe(df, name_column, max_name_length)
        self.df = df[name_column]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[index]

    def _clean_dataframe(self, df, column_name, max_name_length):
        """
        Convert everything to lowercase and remove names with invalid characters
        """
        if len(df) > 0:
            df = df.dropna()
            df[column_name] = df[column_name].map(lambda x: x.lower())
            criterion = df[column_name].apply(lambda x: set(x).issubset(PERMITTED_LETTERS) and len(x) < max_name_length)
            df = df[criterion]
        return df
