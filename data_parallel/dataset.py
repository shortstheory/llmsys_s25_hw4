from random import Random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


# ASSIGNMENT 4.1
class Partition():
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        '''Given index, get the data according to the partitioned index'''
        # BEGIN SOLUTION
        # raise NotImplementedError("Data Parallel Not Implemented Yet")
        return self.data[self.index[index]]
        # END SOLUTION

# ASSIGNMENT 4.1
class DataPartitioner():
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        ''' Create indices for different partitions
        1. Create indices and use `rng` to shuffle indices
        2. Create different partitions of indices according to `sizes` and store in `self.partitions`
        '''
        length = len(data)
        my_list = list(range(length))
        rng.shuffle(my_list)
        count = 0
        for i in range(len(sizes)):
            n = length * sizes[i]
            sublist = []
            for j in range(int(n)):
                if j + count < length:
                    sublist.append(my_list[int(j+count)])
            count = count + n
            self.partitions.append(sublist)

        # BEGIN SOLUTION

    def use(self, partition):
        ''' Return a simple dataset class `Partiton` by original data and partitioned indices

        Just one line of code. Think it simply.
        '''
        # BEGIN SOLUTION
        # raise NotImplementedError("Data Parallel Not Implemented Yet")
        # END SOLUTION
        return Partition(self.data, self.partitions[partition])

# ASSIGNMENT 4.1
def partition_dataset(rank, world_size, dataset, batch_size=128, collate_fn=None):
    """ Partitioning training dataset of the Machine Translation

    Returns:
        DataLoader: partitioned dataloader
    
    Hint:
    1. Calculate the partitioned batch size
    2. Create a partitioner class `DataPartitioner` with dataset and the list of partitioned sizes
    3. Get the current partition dataset given `rank`, use the `use` function in DataPartitioner
    4. Wrap the dataset with `DataLoader`, remember to customize the `collate_fn`
    """
    # BEGIN SOLUTION
    partition_size = batch_size//world_size
    sizes_list = [1.0/world_size] * world_size
    dp = DataPartitioner(dataset, sizes_list)
    p = dp.use(rank)
    return DataLoader(p, batch_size = partition_size, collate_fn = collate_fn, shuffle = True)
    # END SOLUTION
