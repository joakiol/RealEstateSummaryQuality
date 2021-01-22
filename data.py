from pathlib import Path
import webdataset as wds
import time
import sys
import pickle
from torch.utils.data import DataLoader
import copy
import random
from common import ConditionReport, SummaryReport

class progress_bar:
    def __init__(self, iterator, length):
        """Iterator wrapper class for printing progress. Takes an iterator with a given
        length as input, and becomes an iterator that is the same, but also prints progress. 
        Especially convenient for training Word2vec/Doc2vec since these passes several times 
        over the data without much possibility of printing progress. 

        Args:
            iterator (iterator): Input iterator to add progress bar. 
            length (int): Length of input iterator. 
        """        
        self.iterator = iterator
        self.index = 0
        self.starttime = time.time()
        self.lasttime = self.starttime
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):  
        self.index += 1
        if time.time() > (self.lasttime + 1) or self.index == self.length:
            self._print_progress()
        return next(self.iterator)

    def _print_progress(self):   
        self.lasttime = time.time()
        minutes = (time.time()-self.starttime) // 60
        seconds = (time.time()-self.starttime) % 60
        if self.length != None:
            x = int(60*self.index/self.length)
            sys.stdout.write("[%s%s%s]  %i/%i  %02d:%02d \r" % ("="*x, '>'*int(60>x), "."*(60-x-1), self.index, self.length, minutes, seconds))  
        else:
            sys.stdout.write("%i  %02d:%02d \r" % (self.index, minutes, seconds))
        if self.index == self.length:
            sys.stdout.write('\n')
        sys.stdout.flush()
        

class buffer_shuffler:
    def __init__(self, iterator, apply, buffer_size=1):
        """Adds elements in iterator to a buffer, keep buffer size smaller than {buffer_size},
        shuffle elements and return (pop) element from buffer in each iteration. Can also apply
        a function to returned element. 

        Args:
            iterator (iterator): Iterator which is expected to have elements that are lists. 
            apply (func): Apply function to output elements in iterator. 
            buffer_size (int, optional): Buffer size. Defaults to 1.
        """        
        self.iterator = iterator
        self.buffer_size = buffer_size
        self.apply = apply
        self.complete = False
        self.index = 0
        self.buffer = []
        while len(self.buffer) < self.buffer_size:
            self._add_to_buffer()

    def __iter__(self):
        return self

    def __next__(self):  
        if (self.complete == False) and (len(self.buffer) < self.buffer_size):
            self._add_to_buffer()
        try:
            newindex = random.randint(0, min(self.buffer_size, len(self.buffer))-1)
            return self.apply(self.buffer.pop(newindex))
        except:
            raise StopIteration

    def _add_to_buffer(self):      
        try:
            new_elements = next(self.iterator)
            self.buffer.extend(new_elements)
            self.index += 1
        except StopIteration:
            self.complete = True

class BufferShuffler:
    """A rather special class, which takes an iterator as input. The input iterator is expected
    to have elements that are lists. The goal of this class is to create an iterator that 
    returns a single element from these lists at a time. This is done by putting all
    elements in the lists of the input iterator into a buffer, and then popping elements
    from this buffer, one at a time. The buffer is kept at a constant size, and shuffled. 
    Used when training LSA/Word2vec/Doc2vec on e.g. sentences, when the input data is
    stored as list of list of words (list[list[str]]). 
    """    
    def __init__(self, iterable, apply, buffer_size=1):
        self.iterable = iterable
        self.apply = apply
        self.buffer_size = buffer_size

    def __iter__(self):
        return buffer_shuffler(iter(self.iterable), self.apply, self.buffer_size)


class ReportData:
    def __init__(self, path, print_progress=True, shuffle_buffer_size=1, apply=None, 
                       batch_size=1, collate=None):
        """Iterable class object for storing and iterating over data. Used extensively in this 
        project for storing data in different formats that makes training faster and more 
        practical. Takes a path as input, and can either iterate over existing data at this path, 
        or create new data to path. Various arguments for different use cases. Stores data in 
        WebDataset format (tar archives), for memory-friendly reading. 

        Args:
            path (str): Path to store/read data to/from. 
            print_progress (bool, optional): Whether to print progress in iterations. 
                                             Defaults to True.
            shuffle_buffer_size (int, optional): WebDataset shuffles data by putting elements 
                                                 into a buffer with given size. 
                                                 Defaults to 1 (no shuffle).
            apply (func, optional): Apply function to elements in data. Defaults to None (no func).
            batch_size (int, optional): Data can be loaded in batches. Defaults to 1 (no batching).
            collate (func, optional): Apply func to batches. Used for PackedSequence stuff with LSTM. Defaults to None.
        """        
        self.path = path
        self.print_progress = print_progress
        self.shuffle = shuffle_buffer_size
        self.apply = apply
        if self.apply == None:
            self.apply = self._unpack_report
        self.batch_size = batch_size
        self.collate = collate
        try:
            with open('%s/length.pkl' % path, 'rb') as f:
                self.length = pickle.load(f)
        except:
            self.length = None


    def create(self, data, apply=None, overwrite=False):
        """Store dataset to path from input data. Can apply function to data before storing. 

        Args:
            data (iterator): Any iterator type. 
            apply (func, optional): Apply any function to data elements before storing. 
                                    Defaults to transforming ConditionReport object 
                                    to SummaryReport object. 
            overwrite (bool, optional): Will only overwrite existing data at path if 
                                        overwrite=True. Defaults to False.

        Returns:
            ReportData: self
        """
        if (Path('%s/data.tar' % self.path).is_file()) and (overwrite == False):
            return self
        if apply == None:
            apply = self._prepare_report_from_ConditionReport
        try:
            data.print_progress = False
        except AttributeError:
            pass
        try:
            length = len(data)
        except:
            length = None

        data = progress_bar(iter(data), length)
        Path(self.path).mkdir(parents=True, exist_ok=True)
        writer = wds.TarWriter('%s/data.tar' % self.path)
        print("\nPreparing new dataset...")
        new_length = 0
        for element in data:
            out = apply(element)
            if out != None:
                writer.write(out)
                new_length += 1
        writer.close()

        with open('%s/length.pkl' % self.path, 'wb') as f:
            pickle.dump(new_length, f)
        self.length = new_length
        return self

    def _unpack_report(self, element):
        return element['report.pyd']

    def __iter__(self):   
        data = wds.Dataset('%s/data.tar' % self.path).decode()
        if self.shuffle != False: 
            data = data.shuffle(self.shuffle)
        if self.apply != False:
            data = data.map(self.apply)
        length = self.length
        if self.batch_size > 1:
            data = DataLoader(data, batch_size=self.batch_size, collate_fn=self.collate)
            length = int(self.length/self.batch_size)+1
        data = iter(data)
        if self.print_progress:
            data = progress_bar(data, length)
        return data

    def __len__(self):
        return self.length

    def _prepare_report_from_ConditionReport(self, condition_report):
        report = SummaryReport(condition_report)
        if len(report.condition) > 0 and len(report.summary) > 0:
            return {'__key__': report.id, 'report.pyd': report}
        else:
            return None


class subset_iterator:
    def __init__(self, iterator, subset):
        """Creates an iterator with only a subset of the input elements in the input iterator."""    
        self.iterator = iterator
        self.subset = subset
        self.length = len(subset)
        self.index = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            report = next(self.iterator)
            if report.id in self.subset:
                self.index += 1
                return report



class SubsetReportData:
    def __init__(self, data, subset):
        """Iterable class for iterating over only a subset of the elements in input data.

        Args:
            data (iterable[SummaryReport]): Input data with all elements.
            subset (list[str]): Ids of elements to include in subset for iteration. 
        """        
        self.data = copy.copy(data)
        self.print_progress = data.print_progress
        self.data.print_progress = False
        self.subset = subset
        self.length = len(subset)

    def __len__(self):
        return self.length

    def __iter__(self):
        iterator = subset_iterator(iter(self.data), self.subset)
        if self.print_progress:
            iterator = progress_bar(iterator, len(self.subset))
        return iterator


class labelled_iterator:
    def __init__(self, iterator, labels):
        """Creates an iterator with given input labels of the input iterator."""  
        self.iterator = iterator
        self.labels = labels
        self.length = len(labels)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            report = next(self.iterator)
            if report.id in self.labels.index:
                self.index += 1
                return (report, (self.labels.loc[report.id, 'prob_bad'], 
                                 self.labels.loc[report.id, 'prob_good']))


class LabelledReportData:
    def __init__(self, data, labels):
        """Class for adding labels to a subset of elements in data. 
        Self will be an iterable of tuple(element, tuple(float, float)), where
        the first float represents the probability of the summary being bad, and the 
        second float is the probability of summary being good. 

        Args:
            data (iterable[SummaryReport]): Input data with elements. 
            labels (pd.DataFrame): Probabilistic labels. Report ids are expected to be found 
                                   in index, while the columns 'prob_bad' and 'prob_good'
                                   are expected. 
        """        
        self.data = copy.copy(data)
        self.print_progress = data.print_progress
        self.data.print_progress = False
        self.labels = labels
        self.length = len(labels)

    def __len__(self):
        return self.length

    def __iter__(self):
        iterator = labelled_iterator(iter(self.data), self.labels)
        if self.print_progress:
            iterator = progress_bar(iterator, len(self.labels))
        return iterator