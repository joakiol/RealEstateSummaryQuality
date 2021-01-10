from pathlib import Path
import webdataset as wds
import time
import sys
import pickle
import utils as ut
from torch.utils.data import DataLoader
import copy
import random

class progress_bar:
    def __init__(self, iterator, length):
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

    # def __len__(self):
    #     return self.length

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

class PrintIterator:
    def __init__(self, iterable, length):
        self.iterable = iterable
        self.length = length

    def __iter__(self):
        return progress_bar(iter(self.iterable), self.length)


# class ProgressBar:
#     def __init__(self, iterable, length=None):
#         self.iterable = iterable
#         self.length = 

#     def __iter__(self):
#         return progress_bar(self.iterable, self.length)

#     def __len__(self):
#         return self.length


class buffer_shuffler:
    def __init__(self, iterator, apply, buffer_size=1):
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
            #new_element = self.apply(next(self.iterator))
            new_elements = next(self.iterator)
            #if isinstance(new_element, list):
            self.buffer.extend(new_elements)
            #else:
            #    self.buffer.append(new_element)
            self.index += 1
        except StopIteration:
            self.complete = True

class BufferShuffler:
    def __init__(self, iterable, apply, buffer_size=1):
        self.iterable = iterable
        self.apply = apply
        self.buffer_size = buffer_size

    def __iter__(self):
        return buffer_shuffler(iter(self.iterable), self.apply, self.buffer_size)



# class BufferShuffler:
#     def __init__(self, iterable, buffer_size, func):
#         self.iterable = iterable
#         self.buffer_size = buffer_size
#         self.func = func
#         try:
#             self.length = len(iterable)
#         except:
#             self.length = None

#     def __iter__(self):
#         return buffer_randomizer(self.iterable, self.length, self.buffer_size, self.func)


class ReportData:
    def __init__(self, path, print_progress=True, shuffle_buffer_size=1, apply=None, batch_size=1, collate=None):#, list_unpack=False):

        self.path = path
        self.print_progress = print_progress
        self.shuffle = shuffle_buffer_size
        self.apply = apply
        if self.apply == None:
            self.apply = self._unpack_report
        self.batch_size = batch_size
        self.collate = collate
        #self.list_unpack = list_unpack
        try:
            with open('%s/length.pkl' % path, 'rb') as f:
                self.length = pickle.load(f)
        except:
            self.length = None


    def create(self, data, apply=None, overwrite=False):

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
        #if apply != False:
            #data = apply_to_iterator(data, apply=apply)

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
        

        #super(testData, self).__init__('%s/data.tar' % path)

        # self = self.decode()
        # def get_report(element):
        #     return element['report.pyd']
        # self = self.map(get_report)

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
        #if self.list_unpack:
        #    data = buffer_shuffler(data, self.shuffle)
            #data = PrintIterator(data, self.length)
        #if self.apply != False:
        #    data = ApplierIterator(data, self.apply, self.shuffle)   
        return data

    def __len__(self):
        return self.length

    def _prepare_report_from_ConditionReport(self, condition_report):
        """
        Make SummaryReport-object from ConditionReport-object. 
        Emtpy reports/summaries will return None. 
        """
        report = SummaryReport(condition_report)
        if len(report.condition) > 0 and len(report.summary) > 0:
            return {'__key__': report.id, 'report.pyd': report}
        else:
            return None



# class VenduData:
#     """General purpuse iterator for data in WebDataset-format. This will be used throughout this project."""
#     def __init__(self, path='data/VenduData/dataset', progress_bar=True, print_list=None, shuffle=1):
#         """
#         Initialize parameters
        
#         :param path: Path to area where data is stored. Extensions (.tar) should be omitted. 
#         :param progress_bar: Boolean indicator of whether progress of loops through data should be visualized.  
#         :param print_list: List of string. For each iteration through data, the next element will be printed. 
#         :param shuffle: Shuffle the dataset with a buffer of this size. 'shuffle=1' (default) will result in no shuffle. 
#         """
#         self.path = path
#         self.progress_bar = progress_bar
#         self.print_list = print_list
#         self.shuffle = shuffle
#         self.index = None
#         self.time = None
#         self.passno = 0
#         self.data = None
#         try:
#             with open('%s/length.pkl' % self.path, 'rb') as f:
#                 self.length = pickle.load(f)
#         except:
#             self.data = None
#             self.length = None
        
#     def __len__(self):
#         """Return length."""
#         return self.length

#     def __iter__(self):
#         """Prepare for iteration. Print appropriate string if print_list is given. """
#         try:
#             self.data = iter(wds.Dataset('%s/data.tar' % self.path).shuffle(self.shuffle).decode())
#             with open('%s/length.pkl' % self.path, 'rb') as f:
#                 self.length = pickle.load(f)
#         except:
#             raise FileNotFoundError("No dataset was found at given path. Dataset can be made from iterable by using 'make_dataset'-method.")
#         self.index = 0
#         self.time = time.time()
#         if self.print_list != None:
#             print(self.print_list[self.passno])
#         self.passno += 1
#         return self

#     def __next__(self):
#         """Return next element. """
#         try:
#             report = next(self.data)
#         except:
#             self._print_progress_bar(end=True)
#             raise StopIteration     
#         self.index += 1
#         if self.index % 100 == 0:
#             self._print_progress_bar()
#         return report

#     def _print_progress_bar(self, end=False):
#         """Print progress bar string. """
#         if self.progress_bar:
#             minutes = (time.time()-self.time) // 60
#             seconds = (time.time()-self.time) % 60
#             if self.length != None:
#                 x = int(60*self.index/self.length)
#                 sys.stdout.write("[%s%s%s]  %i/%i  %02d:%02d \r" % ("="*x, '>'*int(60>x), "."*(60-x-1), self.index, self.length, minutes, seconds))  
#             else:
#                 sys.stdout.write("%i  %02d:%02d \r" % (self.index, minutes, seconds))
#             if end:
#                 sys.stdout.write('\n')
#             sys.stdout.flush()

#     def make_dataset(self, iterable, func):
#         """
#         Make and save dataset to appropriate format from iterable. 
        
#         :param iterable: Iterable of elements to add to dataset. 
#         :param func: Function to apply on elements in iterable. Should return (id, element). If `element==None`, it will be omitted. 
#         """
#         print("Making dataset...")
#         self.index = 0
#         self.time = time.time()
#         try:
#             self.length = len(iterable)
#         except:
#             self.length = None

#         new_length = 0
#         if Path('%s/data.tar' % self.path).is_file():
#             raise FileExistsError("Dataset with given path already exists. Find a new name, or delete file from folder to create data.")
#         Path(self.path).mkdir(parents=True, exist_ok=True)
#         writer = wds.TarWriter('%s/data.tar' % self.path)
#         for element in iterable:
#             new_elements = func(element)
#             if not isinstance(new_elements, list):
#                 new_elements = [new_elements]
#             for report_id, report in new_elements:
#                 if report != None:
#                     writer.write({'__key__': report_id, 'report.pyd': report})
#                     new_length += 1

#             self.index += 1
#             if self.index % 100 == 0:
#                 self._print_progress_bar()

#         writer.close()
#         self._print_progress_bar(end=True)

#         with open('%s/length.pkl' % self.path, 'wb') as f:
#             pickle.dump(new_length, f)
#         print("Done")

class subset_iterator:
    def __init__(self, iterator, subset):
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
    """Class for iterating through only a subset of elements in data. Inherits from VenduData. """
    def __init__(self, data, subset):
        """
        Initialize parameters. 

        :param data: VenduData-object to make subset from. 
        :param subset: list(string). Ids of elements to include in iteration through data. 
        """
        #super(SubsetVenduData, self).__init__(path=data.path, progress_bar=data.progress_bar, 
        #                                 print_list=data.print_list, shuffle=data.shuffle)
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
    


    # def __next__(self):
    #     """Return next element."""
    #     try:
    #         while True:
    #             report = next(self.data)
    #             if report['__key__'] in self.subset:
    #                 break
    #     except:
    #         self._print_progress_bar(end=True)
    #         raise StopIteration     
    #     self.index += 1
    #     if self.index % 100 == 0:
    #         self._print_progress_bar()
    #     return report['report.pyd']



class labelled_iterator:
    def __init__(self, iterator, labels):
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


class LabelledReportData:#(VenduData):
    """
    Class for adding labels to a subset of elements in data. 
    Object will be an iterable of tuple(element, tuple(float, float)), where
    the first float represents the probability of the summary being bad, and the 
    second float is the probability of summary being good. 
    """
    def __init__(self, data, labels):
        """
        Initialize parameters. 

        :param data: VenduData-object to make subset from. 
        :param labels: Pandas df of probabilistic labels. Report ids are expected to be found in pandas index. 
        """
        # super(LabelledVenduData, self).__init__(path=data.path, progress_bar=data.progress_bar, 
        #                                       print_list=data.print_list, shuffle=data.shuffle)
        self.data = copy.copy(data)
        self.print_progress = data.print_progress
        self.data.print_progress = False
        self.labels = labels
        self.length = len(labels)

    def __len__(self):
        return self.length

    def __iter__(self):
        """Prepare for iteration"""
        iterator = labelled_iterator(iter(self.data), self.labels)
        if self.print_progress:
            iterator = progress_bar(iterator, len(self.labels))
        return iterator
        # self = super(LabelledVenduData, self).__iter__()
        # self.length = len(self.labels)
        # return self

    # def __next__(self):
    #     """Return next element."""
    #     try:
    #         while True:
    #             report = next(self.data)
    #             if report['__key__'] in self.labels.index:
    #                 break
    #     except:
    #         self._print_progress_bar(end=True)
    #         raise StopIteration     
    #     self.index += 1
    #     if self.index % 100 == 0:
    #         self._print_progress_bar()
    #     return (report['report.pyd'], (self.labels.loc[report['__key__'], 'prob_bad'], 
    #                                    self.labels.loc[report['__key__'], 'prob_good']))





# class VenduData(Data):
#     """Iterator of SummaryReport objects. This will be the input for all models in this project. """
#     def __init__(self, name='dataset', iterable=None, progress_bar=True):
#         """
#         Constructor. 

#         :param name: Name of dataset, to load from/save to file.  
#         :param iterable: Optional iterable of ConditionReport objects, to construct data from.
#         :param progress_bar: Boolean indicator of whether progress of loops through data should be shown.  
#         """
#         path = 'data/VenduData/%s' % name
#         if not Path('%s.tar' % path).is_file() and iterable == None:
#             raise FileNotFoundError("If argument iterable=None (default), the constructor expects a pre-made dataset called <name>. This was not found. ")
            
#         if iterable != None:
#             self.index = 0
#             self.time = time.time()
#             self.progress_bar = progress_bar
#             self.length = 1

#             print("Making dataset...")
#             writer = wds.TarWriter('%s.tar' % path)
#             for house in iterable:
#                 report = SummaryReport(house)
#                 if len(report.condition) > 0 and len(report.summary) > 0:
#                     self.index += 1
#                     self.length += 1
#                     writer.write({'__key__': report.id, 'report.pyd': report})
#                     self._print_progress_bar()

#             self.length = self.index
#             self._print_progress_bar()
#             writer.close()
#             with open('%s.pkl' % path, 'wb') as f:
#                 pickle.dump(self.index, f)
#             print("Done")
#         super(VenduData, self).__init__(path=path, progress_bar=progress_bar)


# class LabelledVenduData(Data):
#     """Iterable of 'SummaryReport' objects with corresponding probabilistic labels from weak supervision. """
#     def __init__(self, data, labels, progress_bar=True):
#         """
#         Constructor
        
#         :param data: `VenduData`-object of the data to label. 
#         :param labels: Pandas df with probabilistic labels of reports. 
#                        Should include labels for a subset of reports in `data`. 
#         :param progress_bar: Boolean indicator of whether progress of loops through data should be shown.  
#         """
#         super(LabelledVenduData, self).__init__(path=data.path, progress_bar=progress_bar)
#         self.labels = labels
#         self.length = len(labels)

#     def __next__(self):
#         while True:
#             report = next(self.iterator)['report.pyd']
#             if report.id in self.labels.index:
#                 self.index += 1
#                 self._print_progress_bar()
#                 return (report, [self.labels.loc[report.id, 'prob_bad'], self.labels.loc[report.id, 'prob_good']])

  





# class VenduData:
#     """
#     Class holder for vendu data. 
#     """
#     def __init__(self):
#         """
#         Initialize paths. 
#         """
#         self._length_path = 'data/VenduData/length.pkl'
#         self._initial_path = 'data/VenduData/adjusted_enebolig_reports.pkl'
#         self._full_path = 'data/VenduData/full.tar'
#         self._sections_path = 'data/VenduData/sections.tar'
#         self._sections_tokenized_path = 'data/VenduData/sections_tokenized.tar'
#         self._raw_path = 'data/VenduData/raw.tar'
#         self._words_path = 'data/VenduData/words.tar'
#         self._sentences_path = 'data/VenduData/sentences.tar'
#         self._sentences_tokenized_path = 'data/VenduData/sentences_tokenized.tar'

#         if not Path(self._full_path).is_file():
#             self._make_full()
    
#     def __len__(self):
#         """Get length from saved file (or calculate, and save). """
#         if Path(self._length_path).is_file():
#             with open(self._length_path, 'rb') as f:
#                 length = pickle.load(f)
#         else:
#             data = self.full()
#             length = 0
#             print("Counting length...")
#             for house in data:
#                 length += 1
#             print("Done!")
#             with open(self._length_path, 'wb') as f:
#                 pickle.dump(length, f)
        
#         return length

#     def _clean_text(self, text):
#         """
#         Cleans a string input the following way: 
#         - ensure period at end. 
#         - ensure no trailing spaces. 
#         - ensure no double space or period. 
#         """
#         space_re = re.compile(r"\s+")
#         dot_re = re.compile(r"\.\.+")
            
#         text = space_re.sub(' ', text).strip()
#         text = dot_re.sub('.', text)

#         if text == '':
#             return text
#         elif text[-1] == ',':
#             string_list = list(text)
#             string_list[-1] = '.'
#             text = ''.join(string_list)
#         elif text[-1] not in ['.', '!', '?', ':']:
#             text += '.'

#         return text

#     def progress_bar(self, data, length=None):
#         """Progress-bar function for iterating through data. """
#         if length == None:
#             length = len(self)
#         size = 60
#         counter = 0
#         starttime = time.time()
#         for house in data:
#             counter += 1
#             if counter % 100 == 0:
#                 x = int(size*counter/length)
#                 minutes = (time.time()-starttime) // 60
#                 seconds = (time.time()-starttime) % 60
#                 sys.stdout.write("[%s%s%s]  %i/%i  %02d:%02d \r" % ("="*x, '>', "."*(size-x-1), counter, length, minutes, seconds))
#                 sys.stdout.flush()        
#             yield house

#         sys.stdout.write("[%s] %i/%i  %02d:%02d \r" % ("="*size, length, length, minutes, seconds))
#         sys.stdout.write("\n")
#         sys.stdout.flush()

#     def _make_full(self):
#         """
#         Convert input data into tar format. 
#         """
#         print("Loading pickle file...")
#         data = pickle.load(bz2.BZ2File(self._initial_path, 'rb'), encoding='latin1')
#         print("Done!")

#         writer = wds.TarWriter(self._full_path)
#         length = 0

#         print("One time operation: Cleaning data...")
#         for house in self.progress_bar(data, len(data)):

#             key = house.id
#             condition = []
#             place = []

#             seen = set()
#             for element in house.condition:
#                 room = (element.type, element.room)
#                 descr = self._clean_text(element.description)
#                 assess = self._clean_text(element.assessment)
#                 if room not in seen and (descr != '' or assess != ''):
#                     seen.add(room)

#                     row = {}
#                     row['type'] = element.type
#                     row['room'] = element.room
#                     row['description'] = descr
#                     row['assessment'] = assess
#                     row['degree'] = element.degree
#                     row['adjusted'] = element.adjusted
#                     if hasattr(element, 'original_degree'):
#                         row['original_degree'] = element.original_degree
#                     else:
#                         row['original_degree'] = np.nan

#                     condition.append(row)

#             seen = set()
#             for element in house.place:
#                 room = element.type
#                 descr = self._clean_text(element.description)
#                 # Don't want empty summaries
#                 if room not in seen and len(descr) > 0:
#                     seen.add(room)
#                     place.append({'type': element.type, 'description': descr})
            
#             # Don't want empty report, and summary is type 19 and/or 20
#             if len(condition) > 0 and ((19 in seen) or (20 in seen)):
#                 length += 1
#                 out = {'__key__': key, 'condition.pyd': condition, 'place.pyd': place}
#                 writer.write(out)

#         print("Done!")

#         with open(self._length_path, 'wb') as f:
#             pickle.dump(length, f)

#         writer.close()

#     def _make_sections(self):
#         """Make as list with each section as one element"""

#         data = self.full()
#         writer = wds.TarWriter(self._sections_path)

#         print("One time operation: Making sections data...")
#         for house in self.progress_bar(data, len(self)): 
            
#             key = house['__key__']
#             report = []

#             for element in house['condition.pyd']:
#                 if element['description'] == '':
#                     text = element['assessment']
#                 elif element['assessment'] == '':
#                     text = element['description']
#                 else: 
#                     text = element['description'] + ' ' + element['assessment']
                
#                 report.append(text)

#             text = ''
#             for element in house['place.pyd']:
#                 if element['type'] == 19 or element['type'] == 20: 
#                     if text == '':
#                         text += element['description']
#                     else: 
#                         text += ' ' + element['description']

#             summary = nltk.sent_tokenize(text, language='norwegian')

#             out = {'__key__': key, 'report.pyd': report, 'summary.pyd': summary}
#             writer.write(out)

#         writer.close()
#         print("Done!")

#     def _word_tokenize(self, data, path):
#         """Word_tokenize (and save) input data. """
#         writer = wds.TarWriter(path)

#         print("One time operation: Tokenizing data...")
#         for house in self.progress_bar(data, len(self)): 

#             key = house['__key__']

#             if 'report.pyd' in house.keys():
#                 report = [[word.lower() for word in nltk.word_tokenize(element.replace('/', ' / '), language='norwegian')] for element in house['report.pyd']]
#                 summary = [[word.lower() for word in nltk.word_tokenize(element.replace('/', ' / '), language='norwegian')] for element in house['summary.pyd']]

#             elif 'report.txt' in house.keys():
#                 report = [word.lower() for word in nltk.word_tokenize(house['report.txt'].replace('/', ' / '), language='norwegian')]
#                 summary = [word.lower() for word in nltk.word_tokenize(house['summary.txt'].replace('/', ' / '), language='norwegian')]

#             out = {'__key__': key, 'report.pyd': report, 'summary.pyd': summary}
#             writer.write(out)

#         writer.close()
#         print("Done!")

#     def _make_raw(self):
#         """Make complete report and summary as strings. """

#         data = self.sections()
#         writer = wds.TarWriter(self._raw_path)

#         print("One time operation: Make raw data...")
#         for house in self.progress_bar(data, len(self)):

#             key = house['__key__']
            
#             report = ''
#             for section in house['report.pyd']:
#                 report += section + ' '
#             report.strip()

#             summary = ''
#             for sentence in house['summary.pyd']:
#                 summary += sentence + ' '
#             summary.strip()

#             out = {'__key__': key, 'report.txt': report, 'summary.txt': summary}
#             writer.write(out)

#         writer.close()
#         print("Done!")

#     def _make_sentences(self):
#         """Make reports and summaries like list of sentences """

#         data = self.raw()
#         writer = wds.TarWriter(self._sentences_path)

#         print("One time operation: Make sentence data...")
#         for house in self.progress_bar(data, len(self)):

#             key = house['__key__']
#             report = nltk.sent_tokenize(house['report.txt'], language='norwegian')
#             summary = nltk.sent_tokenize(house['summary.txt'], language='norwegian')

#             out = {'__key__': key, 'report.pyd': report, 'summary.pyd': summary}
#             writer.write(out)
        
#         writer.close()
#         print("Done!")

#     def _create_meta(self, path):
#         """Make empty dataframe with ids. """
#         data = self.raw()
#         keys = []
#         for house in data:
#             keys.append(house['__key__'])

#         meta = pd.DataFrame({'id': keys})
#         meta.to_csv(path, index=False)
#         return meta

#     def full(self):
#         """Return iterator for full data. """
#         return wds.Dataset(self._full_path).decode()

#     def sections(self):
#         """Return iterator for section data. """
#         if not Path(self._sections_path).is_file():
#             self._make_sections()
#         return wds.Dataset(self._sections_path).decode()

#     def sections_tokenized(self):
#         """Return iterator for tokenized section data. """
#         if not Path(self._sections_tokenized_path).is_file():
#             self._word_tokenize(self.sections(), self._sections_tokenized_path)
#         return wds.Dataset(self._sections_tokenized_path).decode()

#     def raw(self):
#         """Return iterator for raw text data. """
#         if not Path(self._raw_path).is_file():
#             self._make_raw()
#         return wds.Dataset(self._raw_path).decode()

#     def words(self):
#         """Return iterator for tokenized text data. """
#         if not Path(self._words_path).is_file():
#             self._word_tokenize(self.raw(), self._words_path)
#         return wds.Dataset(self._words_path).decode()

#     def sentences(self):
#         """Return iterator for sentences data. """
#         if not Path(self._sentences_path).is_file():
#             self._make_sentences()
#         return wds.Dataset(self._sentences_path).decode()

#     def sentences_tokenized(self): 
#         """Return iterator for tokenized sentences data. """
#         if not Path(self._sentences_tokenized_path).is_file():
#             self._word_tokenize(self.sentences(), self._sentences_tokenized_path)
#         return wds.Dataset(self._sentences_tokenized_path).decode()

