import numpy as np

# class DataGenerator(object):
#     def __init__(self, batch_size, type, te_max_iter=None):
#         assert type in ['train', 'test']
#         self._batch_size_ = batch_size
#         self._type_ = type
#         self._te_max_iter_ = te_max_iter
#         
#     def generate(self, xs, ys):
#         x = xs[0]
#         y = ys[0]
#         batch_size = self._batch_size_
#         n_samples = len(x)
#         
#         index = np.arange(n_samples)
#         np.random.shuffle(index)
#         
#         iter = 0
#         epoch = 0
#         pointer = 0
#         while True:
#             if (self._type_ == 'test') and (self._te_max_iter_ is not None):
#                 if iter == self._te_max_iter_:
#                     break
#             iter += 1
#             if pointer >= n_samples:
#                 epoch += 1
#                 if (self._type_) == 'test' and (epoch == 1):
#                     break
#                 pointer = 0
#                 np.random.shuffle(index)                
#  
#             batch_idx = index[pointer : min(pointer + batch_size, n_samples)]
#             pointer += batch_size
#             yield x[batch_idx], y[batch_idx]


class DataGenerator(object):
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(0)
        self.validate_random_state = np.random.RandomState(1)
        
    def generate_train(self):
        
        batch_size = self.batch_size
        samples_num = len(self.x)
        indexes = np.arange(samples_num)

        self.random_state.shuffle(indexes)

        iteration = 0
        pointer = 0

        while True:

            # Reset pointer
            if pointer >= samples_num:
                pointer = 0
                self.random_state.shuffle(indexes)

            # Get batch indexes
            batch_indexes = indexes[pointer: pointer + batch_size]
            pointer += batch_size

            iteration += 1

            batch_x = self.x[batch_indexes]
            batch_y = self.y[batch_indexes]

            yield batch_x, batch_y
        
        
    def generate_validate(self, shuffle, max_iteration=None):
        
        batch_size = self.batch_size
        samples_num = len(self.x)
        indexes = np.arange(samples_num)

        if shuffle:
            self.validate_random_state.shuffle(indexes)

        iteration = 0
        pointer = 0

        while True:

            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= samples_num:
                break
                
            batch_indexes = indexes[pointer: pointer + batch_size]
            pointer += batch_size

            iteration += 1

            batch_x = self.x[batch_indexes]
            batch_y = self.y[batch_indexes]
            
            yield batch_x, batch_y