from typing import List, Tuple


class DataSplitter:
    @staticmethod
    def split_data(data: List, train_ratio: float, valid_ratio: float) -> Tuple[List, List, List]:
        dataset_size = len(data)
        
        train_set = data[0:int(dataset_size * train_ratio)]

        valid_set = data[int(dataset_size *
                            train_ratio):int(dataset_size * train_ratio) +
                        int(dataset_size * valid_ratio)]

        test_set = data[int(dataset_size * train_ratio) +
                        int(dataset_size * valid_ratio):dataset_size]

        return train_set, valid_set, test_set
