# SummaryQuality

This project is part of a master thesis on NTNU, where the goal is to measure the quality of summaries of real-estate condition reports. In this project, weak supervision is first used to obtain labels. Then, various models are trained on these labels, such that they learn to measure the quality of summaries for real-estate condition reports. 

It must be noted that the models are entirely based on weak supervision. This means that the models have **not** been evaluated on data where the true summary quality is known. It is therefore uncertain how accurate the models truly are. Any results should be regarded as an **indication** only, and not as a ground truth. 

The following explains the functionality of this project. A more detailed documentation is found at the end. 

# Usage

To work with this project, the necessary files should be copied into the working directory. This includes the following: 
- **common.py**: Classes for containing real-estate condition reports. 
- **data.py**: Classes for making, storing and iterating through datasets. 
- **weak_supervision.py**: Implements a weak supervision label model based on the Snorkel-framework, used for making labels. 
- **labelling_functions.py**: Contains the labelling functions (with helper functions) that the label model in weak_supervision.py is based on. 
- **models.py**: Implements the various models used in this work. 
- **networks.py**: Implements neural network architectures used by the models in `models.py`. 
- **utils.py**: Contains useful functions for analysing performance, and various other stuff. 

The necessary packages for these files are found in requirements.txt. 

### Data structure

Vendu has implemented the class `ConditionReport`, which represents a real-estate condition report. In this project, the class `SummaryReport` has also been implemented, which inherits from the `ConditionReport` class, and has implemented some additional methods that are useful for this project. 

The models of this work expect data input as an iterable of SummaryReport objects (with corresponding labels, when appropriate). In this project, the iterable class `ReportData` has been implemented for creating and representing datasets. This makes use of the WebDataset package, which stores data in .tar archives that can be streamed from disk, such that iterating becomes memory-friendly.  The `ReportData` class is used by our models for storing data at different stages (pre-processed, embedded, etc.), and is also made to be used to represent our dataset of real-estate condition reports. A dataset of SummaryReports can be made from a list/iterable of ConditionReport-objects the following way:

```python
import pickle
import bz2
from data import ReportData

print("Loading pickled file...")
list_of_ConditionReport = pickle.load(bz2.BZ2File('data/VenduData/adjusted_enebolig_reports.pkl', 'rb'), 
                                      encoding='latin1')
print("Done!")

data = ReportData(path='data/dataset')
data.create(data=list_of_ConditionReport)
```
The above code will save the dataset to file at './data/dataset', and will only have to run once. Afterwards, the dataset can be initialized by:
```python
data = ReportData(path='data/dataset')

for idx, report in enumerate(data):
    print(report.id)
    if idx > 5:
        break
```

### Making weak supervision labels

Now that a dataset with an appropriate structure has been created, we can train a weak supervision label model to make noisy labels. Note that the current implementation expects the following files to exist: './data/VenduData/matched_policies.zip' and 'data/VenduData/claims.csv'. 

```python
from weak_supervision import WSLabelModel

# Initialize and train model
label_model = WSLabelModel()
label_model.train(data=data)

# Save trained model
label_model.save(modelname='default')

# Load previously saved model
label_model = WSLabelModel().load(modelname='default')

# Predict labels on the data that the model was trained on
labels = label_model.predict_training_set()
```

We analyse our label model and labelling functions the following way: 

```python
print("\nlabelling function analysis:")
label_model.analyse_training_set()
print("\nEstimated labelling function accuracies:")
label_model.print_estimated_accuracies()
label_model.plot_training_labels()
```

### Making a training, validation and test set

The models in this work expect the input to train-methods to be `iterable[tuple(SummaryReport, tuple(float, float))]`, where the first float represents the probability of the summary being bad, while the second float represents the probability of the summary being good, according to the label model above. To avoid having to make new datasets (thus storing the data twice), the iterable classes `SubsetReportData` and `LabelledReportData` have been made, where`SubsetReportData` only iterates through a subset of the elements in the data, while `LabelledVenduData` also iterates through a subset and returns the labels together with the elements. With these classes, a train, val and test set can easily be made by the following code: 

```python
from data import SubsetReportData, LabelledReportData
import utils as ut

labels_train, labels_val, labels_test = ut.train_val_test_split(labels, ratio=[0.8, 0.1, 0.1], seed=1)
train = LabelledReportData(data=data, labels=labels_train)
val = LabelledReportData(data=data, labels=labels_val)
test = SubsetReportData(data=data, subset=labels_test.index)
```

### Defining and training models

We have now prepared our data, and we are ready for training models. The training is performed in four steps: 
1. The data is first pre-processed and stored in an appropriate way for the given model. Note that a name must be given to each dataset. The pre-processed data will be stored to a path based on the given name. This step must be done only once for each type of pre-processing. 
2. After pre-processing, the embedder can be trained. 
3. Once the embedder is trained, the textual data can be embedded, to prepare it for the neural network. Embeddings are stored to a path based on the given dataset name and the embedder used to embed the data. This step must be done once for each version of an embedder. 
4. When the embeddings are ready, the neural network model can finally be trained. This is the fourth and final step in the training process. 

Training and testing are done the following way: 

```python
from models import LSAEmbedder, Doc2vecEmbedder, VocabularyEmbedder, Word2vecEmbedder 
from models import FFNModel, LSTMModel, CNNModel, SummaryQualityModel

embedder = LSAEmbedder()
h = FFNModel()
model = SummaryQualityModel(embedder=embedder, model=h)

# train method performs steps 2-4 by default, and step 1 only if it has not been done before. 
model.train(train_name='train', train_data=train, val_name='validation', val_data=val)

# Predict qualities for validation and test set
q_val = model.predict(data_name='validation', data=val)
q_test = model.predict(data_name='test', data=test)

q_val.to_csv('predictions/LSA+LinTrans_val.csv')
q_test.to_csv('predictions/LSA+LinTrans_test.csv')

```
Note that the train-method by default always performs steps 2-4, while step 1 is only performed if it has not been performed before. The steps can be controlled in the following way: 

```python
# Only step 1: 
model.prepare_data(dataname='train', data=train, overwrite=True, train_embedder=False, overwrite_emb=False)

# Only step 2-3, assuming that step 1 has been performed before:
model.prepare_data(dataname='train', data=train, overwrite=False, train_embedder=True, overwrite_emb=True)

# Only step 4, assuming step 1-3 has been performed before:
model.train(train_name='train', train_data=train, val_name='validation', val_data=val, 
            overwrite=False, train_embedder=False, overwrite_emb=False)

# CPU part of training/testing
model.prepare_data(dataname='train', data=train, train_embedder=True, overwrite_emb=True)
model.prepare_data(dataname='validation', data=val, overwrite_emb=True)
model.prepare_data(dataname='test', data=test, overwrite_emb=True)

# GPU part of training. This way, the neural network can also be fine-tuned without performing 1-3 over again. 
model.train(train_name='train', train_data=train, val_name='validation', val_data=val, 
            train_embedder=False, overwrite_emb=False)
            
# Predict on prepared datasets
q_val = model.predict(data_name='validation', data=val, overwrite_emb=False)
q_test = model.predict(data_name='test', data=test, overwrite_emb=False)
```

We evaluate our models the following way:

```python
import pandas as pd

# Get qualities for val and test set from saved files
quality_val = pd.read_csv('predictions/LSA+LinTrans_val.csv', index_col=0, squeeze=True)
quality_test = pd.read_csv('predictions/LSA+LinTrans_test.csv', index_col=0, squeeze=True)

# Calculate training loss and accuracy scores
loss = ut.noise_aware_cosine_loss(quality_test, labels_test, 0.2, -0.2)
best_threshold = ut.get_best_threshold(quality_val, labels_val)
acc, prec, rec, f1 = ut.classification_scores(quality_test, labels_test, best_threshold)

# Print loss and classification scores and plot distribution of summary quality
print(f"Loss: {loss:.3f}")
print(f"acc: {acc:.3f}, prec: {prec:.3f}, rec: {rec:.3f}, f1: {f1:.3f}")
ut.plot_quality(quality_test, labels=labels_test, title='LSA+LinTrans', show=True)
```

In our work, we use the following models in the results. 

```python

# LSA baseline
LSA = LSAEmbedder()
LSA_baseline = SummaryQualityModel(embedder=LSA)

# Doc2vec baseline
Doc2vec = Doc2vecEmbedder()
Doc2vec_baseline = SummaryQualityModel(embedder=Doc2vec)

# LSA+LinTrans
LSA = LSAEmbedder()
LinTrans = FFNModel(layers=[100])
LSA_LinTrans = SummaryQualityModel(embedder=LSA, model=LinTrans)

# Doc2vec+LinTrans
Doc2vec = Doc2vecEmbedder(dim=500)
LinTrans = FFNModel(layers=[100])
Doc2vec_LinTrans = SummaryQualityModel(embedder=Doc2vec, model=LinTrans)

# LSA+FFN
LSA = LSAEmbedder()
FFN = FFNModel(layers=[1000, 1000, 100])
LSA_FFN = SummaryQualityModel(embedder=LSA, model=FFN)

# Doc2vec+LinTrans
Doc2vec = Doc2vecEmbedder(dim=500)
FFN = FFNModel(layers=[1000, 1000, 1000, 100])
Doc2vec_FFN = SummaryQualityModel(embedder=Doc2vec, model=FFN)

# LSA+LSTM
LSA = LSAEmbedder()
LSTM = LSTMModel()
LSA_LSTM = SummaryQualityModel(embedder=LSA, model=LSTM)

# Doc2vec+LSTM
Doc2vec = Doc2vecEmbedder(dim=500)
LSTM = LSTMModel()
Doc2vec_LSTM = SummaryQualityModel(embedder=Doc2vec, model=LSTM)

# EmbLayer+CNN
vocabulary = VocabularyEmbedder()
CNN = CNNModel(embedding_size=500, output_size=500, kernels=[5], learning_rate=1e-3)
EmbLayer_CNN = SummaryQualityModel(embedder=vocabulary, model=CNN)

# Word2vec+CNN
Word2vec = Word2vecEmbedder()
CNN = CNNModel()
Word2vec_CNN = SummaryQualityModel(embedder=Word2vec, model=CNN)
```

# Documentation

This documentation includes a short description of all classes and functions that are meant to be used by a user. There are more classes and functions that a user should not have to use herself. See code with docstrings for a complete overview. 

## common.py
  
-   #### `class Cadastre(Knr, Gnr, Bnr, Fnr, Snr, Anr)`  

    Contains metadata for a property. Implemented by Vendu. 
    
    
-   #### `class Building(build_year, build_type, areal_bra, areal_boa, areal_prom, debt, fortune, constr_cost)`    

    Contains metadata for a building. Implemented by Vendu. 
  
  
-   #### `class PlaceDescription(type, description)`   

    Contains a description concerning the placement of a real-estate. The summary is also a PlaceDescription. Implemented by Vendu. 
  
  
-   #### `class ConditionDescription(type, room, description, assessment, degree)`  

    Contains a description, an assessment and the TG degree for a part of the real-estate. Implemented by Vendu. 
  
  
-   #### `class ConditionReport(id, type, date, author, building, cadastre, place, condition)`  

    Contains a condition report. Implemented by Vendu.  
    
    **Arguments**
    - **`id`**  `(str)`: ID of real-estate condition report.  
    - **`type`**  `(int)`: Not sure.  
    - **`date`**  `(str)`: Date of real-estate assessment.  
    - **`author`**  `(int)`: Author of real-estate condition report.  
    - **`building`**  `(Building)`: Contains metadata about the building. 
    - **`cadastre`**  `(Cadastre)`: Contains metadata about the cadastre. 
    - **`place`**  `(list[PlaceDescription])`: Contains placement info of the real-estate.
    - **`condition`**  `(list[ConditionDescription])`: Contains condition info of the real-estate. 
  
  
-   #### `class SummaryReport(cr)`  

    Extends the `ConditionReport` class, with some useful methods for this project.  
    
    **Arguments**  
    - **`cr`**  `(ConditionReport)`: A real-estate condition report.
    
    **Methods**  
    -   ##### `SummaryReport.get_report_raw()`  
        **Return**  `(str)`: The complete report text.
    -   ##### `SummaryReport.get_summary_raw()`  
        **Return**  `(str)`: The complete summary text. 
    -   ##### `SummaryReport.get_sections()`  
        **Return**  `(list[str])`: The sections of the report. 
    -   ##### `SummaryReport.get_report_words()`  
        **Return**  `(list[str])`: The words of the report. 
    -   ##### `SummaryReport.get_summary_words()`  
        **Return**  `(list[str])`: The words of the summary. 
    -   ##### `SummaryReport.get_report_sentences()`  
        **Return**  `(list[str])`: The sentences of the report. 
    -   ##### `SummaryReport.get_summary_sentences()`  
        **Return**  `(list[str])`: The sentences of the summary.
    -   ##### `SummaryReport.get_tokenized_sections()`  
        **Return**  `(list[list[str]])`: The tokenized sections of the report. 
    -   ##### `SummaryReport.get_report_tokenized_sentences()`  
        **Return**  `(list[list[str]])`: The tokenized sentences of the report. 
    -   ##### `SummaryReport.get_summary_tokenized_sentences()`  
        **Return**  `(list[list[str]])`: The tokenized sentences of the summary. 
    
    
## data.py
  
-   #### `class ReportData(path, print_progress=True, shuffle_buffer_size=1, apply=None, batch_size=1, collate=None)`
  
    Iterable class for storing and iterating over data. Used extensively in this project for storing data in different formats that make training faster and more practical. Takes a path as input, and can either iterate over existing data at this path, or create new data to path. Various arguments for different use cases. Stores data in WebDataset format (tar archives), for memory-friendly reading. 
  
    **Arguments**  
    - **`path`**  `(str)`: Path to store/read data to/from. 
    - **`print_progress`**  `(bool, optional)`: Whether to print progress in iterations. Defaults to True.
    - **`shuffle_buffer_size`**  `(int, optional)`: WebDataset shuffles data by putting elements into a buffer with a given size. Defaults to 1 (no shuffle).
    - **`apply`**  `(func, optional)`: Apply function to elements in data. Defaults to None (no func).
    - **`batch_size`**  `(int, optional)`: Data can be loaded in batches. Defaults to 1 (no batching).
    - **`collate`**  `(func, optional)`: Apply func to batches. Used for PackedSequence stuff with LSTM. Defaults to None.
    
    **Methods**
    -   **`ReportData.create(self, data, apply=None, overwrite=False)`**  
    
        Store dataset to path from input data. Can apply function to data before storing. 
        
        **Arguments**  
        - **`data`**  `(iterator)`: Iterator of any type. 
        - **`apply`**  `(func, optional)`: Apply any function to data elements before storing. Defaults to transforming ConditionReport object to SummaryReport object. Given function must return a dictionary in the following format: `dict{'__key__': <id of element>, '<something>.pyd': <store object here>, ...}`.
        - **`overwrite`**  `(bool, optional)`: Will only overwrite existing data at path if overwrite=True. Defaults to False.
    

-   #### `class SubsetReportData(data, subset)`  

    Iterable class for iterating over only a subset of the elements in input data.

    **Arguments**  
    - **`data`**  `(iterable[SummaryReport])`: Input data with all elements.
    - **`subset`**  `(list[str])`: Ids of elements to include in subset for iteration. 
    

-   #### `class LabelledVenduData(data, labels)`  

    Class for adding labels to a subset of elements in data. Becomes `iterable[tuple(SummaryReport, tuple(float, float))]`, where the first float represents the probability of the summary being bad, and the second float is the probability of summary being good. 

    **Arguments** 
    - **`data`**  `(iterable[SummaryReport])`: Input data with elements. 
    - **`labels`**  `(pd.DataFrame)`: Probabilistic labels. Report ids are expected to be found in index, while the columns 'prob_bad' and 'prob_good' are expected. 


## weak_supervision.py 
  
-   #### `class WSLabelModel()`  

    Class for weak supervision label model, used for making labels.
    
    **Methods**  
  
    -   **`WSLabelModel.train(data)`**  
   
        Trains the model on the input data.  
        
        - **Argument `data`**  `(iterable[SummaryReport])`: Data for training the label model.
          
          
    -   **`WSLabelModel.predict(data)`**  
    
        Fit model on data. If data is the same as the label model was trained on, it is much faster to use `WSLabelModel.predict_training_set()`. 
         
        - **Argument `data`**  `(iterable[SummaryReport])`: Data to predict labels on.
          
        - **Return** `(pd.DataFrame)`: Probabilistic labels for data, with ids in index. 
          
          
    -   **`WSLabelModel.predict_training_set()`**  
    
        Predict labels on the set that was used for training the model. Much faster than calling `WSLabelModel(data=data)` if `data` was also used to train the label model, since pre-processing already has been performed in that case. 

      
        - **Return**  `(pd.DataFrame)`: Probabilistic labels for the data that the model was trained on, with ids in the index, and the columns `prob_good` and `prob_bad`. 
          
    -   **`WSLabelModel.analyse_training_set(latexpath=None)`**  
    
        Perform analysis of labelling functions based on the data that the label model was trained on. Prints coverage, overlap and conflict rates for the labelling functions. 
        
        - **Parameter `latexpath`**  `(str, optional)`: Save analysis to txt file, in latex table format. Defaults to None (no saving to file).
        
        
    -   **`WSLabelModel.print_estimated_accuracies()`**  
    
        Print estimated accuracies for the labelling functions, based on the data that the label model was trained on.
        
        
    -   **`WSLabelModel.plot_training_labels(name=None)`**  
    
        Plot estimated weak supervision labels for the data the label model was trained on.
        
        - **Argument `name`**  `(str, optional)`: Determines filename for saving plot. Defaults to None (no saving).
            
            
    -   **`WSLabelModel.save(modelname='default')`**  
    
        Save model to file.  
        
        - **Parameter `modelname`**  `(str, optional)`: Determines filename for saving model. Defaults to 'default'.
          
          
    -   **`WSLabelModel.load(modelname='default')`**  
    
        Load previously saved label model from file. 
      
        - **Parameter `modelname`**  `(str, optional)`: Determines filename to load from. Defaults to 'default'.
        
        - **Return**  `(WSLabelModel)`: Self, so that one can write `model = WSLabelModel().load(modelname)`. 
   

## utils.py

  
-   #### `function train_val_test_split(labels, ratio=[0.8, 0.1, 0.1], seed=1)`  
  
    Split input data into a train, val and test set. 
  
    **Arguments**  
    - **`labels`**  `(pd.DataFrame)`: Labels with report ids in index and columns `prob_good` and `prob_bad`. 
    - **`ratio`**  `(list[float], optional)`: Size of train, val and test. Defaults to [0.8, 0.1, 0.1].
    - **`seed`**  `(int, optional)`: Seed for shuffling. Defaults to 1.
    
    **Return**  
    - **`probs_train`**  `(pd.DataFrame)`: Dataframe with labels for training set. 
    - **`probs_val`**  `(pd.DataFrame)`: Dataframe with labels for validation set. 
    - **`probs_test`**  `(pd.DataFrame)`: Dataframe with labels for test set. 
    
    
-   #### `function plot_quality(quality, labels=False, name=False, title=False, lab=False, show=True, all=False, density=True, xlim=True)`  
  
    Plot two distributions, one for good summaries, and one for bad.
  
    **Arguments** 
    - **`quality`**  `(pd.Series)`: Quality measures to plot. Ids are expected in index. 
    - **`labels`**  `(pd.DataFrame, optional)`: Labels to use for determining which are good and bad summaries. Ids on index and columns `prob_good` and `prob_bad` are expected. Defaults to False, in which case only one distribution is shown.
    - **`name`**  `(str, optional)`: Save name. Defaults to False (no saving).
    - **`title`**  `(str, optional)`: Title. Defaults to False (No title).
    - **`lab`**  `(bool, optional)`: Whether legends should be used in plot. Defaults to False.
    - **`show`**  `(bool, optional)`: Whether to show plot. Defaults to True.
    - **`all`**  `(bool, optional)`: Whether distribution of all qualities should be shown in grey background. Defaults to False.
    - **`density`**  `(bool, optional)`: Whether normalization of distributions should be performed. Defaults to True.
    - **`xlim`**  `(bool, optional)`: Whether x-axis should be set to [-1, 1]. Defaults to True.
    

-   #### `function noise_aware_cosine_loss(quality, labels, tau_good, tau_bad)`  
  
    Calculates a noise aware cosine similarity loss for a set of qualities/labels.
  
    **Arguments**  
    - **`quality`**  `(pd.Series)`: Predicted qualities, with ids on index. 
    - **`labels`**  `(pd.DataFrame)`: Labels, with ids on index, and columns `prob_good` and `prob_bad`. 
    - **`tau_good`**  `(float)`: Which tau_good to use for loss function. 
    - **`tau_bad`**  `(float)`: Which tau_bad to use for loss function. 
    
    **Return**  `(float)`: Noise aware cosine embedding loss for input qualities and labels. 


-   #### `function get_best_threshold(quality, labels)`  
  
    Get maximum-accuracy threshold to use for classification.
  
    **Arguments**  
    - **`quality`**  `(pd.Series)`: Predicted qualities, with ids on index. 
    - **`labels`**  `(pd.DataFrame)`: Labels, with ids on index, and columns `prob_good` and `prob_bad`. 
    
    **Return**  `(float)`: Threshold that maximizes accuracy. 


-   #### `function classification_scores(quality, labels, best_threshold)`  
  
    Calculates accuracy, recall, precision and F1-score on the labels, by using given threshold for classifying.
  
    **Arguments**  
     - **`quality`**  `(pd.Series)`: Predicted qualities, with ids on index. 
    - **`labels`**  `(pd.DataFrame)`: Labels, with ids on index, and columns `prob_good` and `prob_bad`. 
    - **`best_threshold`** `(float)`: Threshold to use for classifying good/bad summaries. 
    
    **Return**  
    - **`accuracy`**  `(float)`: Accuracy for given qualities and threshold. 
    - **`precision`**  `(float)`: Precision for given qualities and threshold. 
    - **`recall`**  `(float)`: Recall for given qualities and threshold. 
    - **`f_one`**  `(float)`: F1_score for given qualities and threshold. 


## models.py


-   #### `class TFIDFEmbedder(no_below=15000, remove_stopwords=False)`  
  
    TF-IDF embedder, which was tested as an alternative to LSA. 
  
    **Arguments**  
    - **`no_below`**  `(int, optional)`: Number of documents a word must appear in for it to be included in the vocabulary. Defaults to 15000.
    - **`remove_stopwords`**  `(bool, optional)`: Whether stopwords should be removed. Defaults to False.

  
-   #### `class LSAEmbedder(dim=500, no_below=15000, no_above=1, filter_most_frequent=0, remove_stopwords=False)`  
  
    LSA embedder, to be used together with FFN or LSTM. 
  
    **Arguments**  
    - **`dim`**  `(int, optional)`: Dimensionality of embeddings. Defaults to 500.
    - **`no_below`**  `(int, optional)`: Number of documents a word must appear in for it to be included in the vocabulary. Defaults to 15000.
    - **`no_above`**  `(float, optional)`: Maximum fraction of documents a word can be in for it to be included in the vocabulary. Defaults to 1.
    - **`filter_most_frequent`**  `(int, optional)`: Most frequent words removed. Defaults to 0.
    - **`remove_stopwords`**  `(bool, optional)`: Whether stopwords should be removed. Defaults to False.
   
  
-   #### `class Doc2vecEmbedder(dim=100, window=6, mc=20, workers=4, dm=0, epochs=50)`  
  
    Doc2vec embedder, to be used together with FFN or LSTM. 
  
    **Arguments**  
    - **`dim`**  `(int, optional)`: Dimensionality of document embeddings. Defaults to 100.
    - **`window`**  `(int, optional)`: Window size in Doc2vec. Defaults to 6.
    - **`mc`**  `(int, optional)`: Number of times word must appear to be included in vocabulary. Defaults to 20.
    - **`workers`**  `(int, optional)`: Workers in training process. Defaults to 4.
    - **`dm`**  `(int, optional)`: Whether PV-DM version should be used. Defaults to 0 (PV-DBOW).
    - **`epochs`**  `(int, optional)`: Number of training epochs in Doc2vec. Defaults to 50.
    
    
-   #### `class VocabularyEmbedder(vocab_length=20000)`  
  
    Vocabulary embedder, for use together with CNN when making our own word embeddings. 
  
    **Argument `vocab_length`**  `(int, optional)`: Vocabulary size. Defaults to 20000.
    
    
-   #### `class Word2vecEmbedder(dim=100, window=10, min_count=20, workers=4, epochs=50)`  
  
    Word2vec embedder, for use together with CNN. 
  
    **Arguments**  
    - **`dim`**  `(int, optional)`: Dimensionality of word embeddings. Defaults to 100.
    - **`window`**  `(int, optional)`: Window size in Word2vec. Defaults to 10.
    - **`min_count`**  `(int, optional)`: Number of times word must appear to be included in vocabulary. Defaults to 20.
    - **`workers`**  `(int, optional)`: Workers in training process. Defaults to 4.
    - **`epochs`**  `(int, optional)`: Number of training epochs in Word2vec. Defaults to 50.
    
    
-   #### `class FFNModel(layers=[100], batch_size=64, dropout=0.2, epochs=30, learning_rate=1e-4, tau_good=0.2, tau_bad=-0.2)`  
  
    FFN summary quality model, for use together with TFIDF- LSA- or Doc2vecEmbedder.
  
    **Arguments**  
    - **`layers`**  `(list, optional)`: Number of nodes to use in each layer. Number of layers becomes length of layers list. Defaults to [100], which yields in a linear transformation only(LinTrans).
    - **`batch_size`**  `(int, optional)`: Batch size for training model. Defaults to 64.
    - **`dropout`**  `(float, optional)`: Dropout rate when training model. Defaults to 0.2.
    - **`epochs`**  `(int, optional)`: Number of epochs when training model. Defaults to 30.
    - **`learning_rate`**  `(float, optional)`: Initial learning rate. Defaults to 1e-4.
    - **`tau_good`**  `(float, optional)`: Tau_good to use for training model. Defaults to 0.2.
    - **`tau_bad`**  `(float, optional)`: Tau_bad to use for training model. Defaults to -0.2.


-   #### `class LSTMModel(lstm_dim=100, num_lstm=1, bi_dir=False, output_dim=100, batch_size=64, dropout=0, epochs=30, learning_rate=1e-3, tau_good=0.2, tau_bad=-0.2)`  
  
    LSTM summary quality model, for use together with LSAEmbedder or Doc2vecEmbedder.
  
    **Arguments** 
    - **`lstm_dim`**  `(int, optional)`: Dimensionality of LSTM cell. Defaults to 100.
    - **`num_lstm`**  `(int, optional)`: Number of LSTM layers. Defaults to 1.
    - **`bi_dir`**  `(boolean, optional)`: Whether bi-directional LSTM layers are employed or not. Defaults to False.
    - **`output_dim`**  `(int, optional)`: Output dimensionality of final, fully connected linear layer. Defaults to 100.
    - **`batch_size`**  `(int, optional)`: Batch size for training model. Defaults to 64.
    - **`dropout`**  `(float, optional)`: Dropout rate for training model. Defaults to 0.
    - **`epochs`**  `(int, optional)`: Number of epochs in training model. Defaults to 30.
    - **`learning_rate`**  `(float, optional)`: Initial learning rate for training model. Defaults to 1e-3.
    - **`tau_good`**  `(float, optional)`: Tau_good to use for training model. Defaults to 0.2.
    - **`tau_bad`**  `(float, optional)`: Tau_bad to use for training model. Defaults to -0.2.


-   #### `class CNNModel(embedding_size=100, output_size=200, kernels=[2,3,5,7,10], batch_size=64, dropout=0.1, epochs=30, learning_rate=1e-2, tau_good=0.2, tau_bad=-0.2)`  
  
    CNN summary quality model, for use together with VocabularyEmbedder or Word2vecEmbedder.
  
    **Arguments**  
    - **`embedding_size`**  `(int, optional)`: Dimensionality of word embeddings in EmbLayer/Word2vec. Defaults to 100.
    - **`output_size`**  `(int, optional)`: Number of filters per filter size and nodes in final linear layer. Defaults to 200.
    - **`kernels`**  `(list[int], optional)`: List of filter sizes. Total number of filters becomes len(kernels)*output_size. Defaults to [2,3,5,7,10].
    - **`batch_size`**  `(int, optional)`: Batch size for training model. Defaults to 64.
    - **`dropout`**  `(float, optional)`: Dropout rate for training model. Defaults to 0.1.
    - **`epochs`**  `(int, optional)`: Number of epochs when training model. Defaults to 30.
    - **`learning_rate`**  `(float, optional)`: Initial learning rate for training model. Defaults to 1e-2.
    - **`tau_good`**  `(float, optional)`: Tau_good to use for training model. Defaults to 0.2.
    - **`tau_bad`**  `(float, optional)`: Tau_bad to use for training model. Defaults to -0.2.
    
    
-   #### `class EmptyModel()`  
  
    Empty model, for using LSA or Doc2vec alone as baseline models. 
    

-   #### `class SummaryQualityModel(embedder, model=EmptyModel())`
  
    General class for summary quality models. Takes an embedder and an optional neural network model as input, and combines them accordingly. 
  
    **Arguments**  
    - **`embedder`**  `(<any>Embedder)`: Embedder model to use in summary quality model.
    - **`model`**  `(<any>Model, optional)`: Nerual network model to use together with embedder in summary quality model. Defaults to EmptyModel (baseline).
    
    **Methods**
    -   **`SummaryQualityModel.prepare_data(dataname, data, overwrite=False, overwrite_emb=False, train_embedder=False)`**  
    
        Prepare for training/testing neural network part of model. Will pre-process data (if overwrite=True/not already done), train embedder model (if train_embedder=True) and embed textual data (if overwrite_emb=True/not already done). These embeddings are the input to the neural network part of the summary quality model, and will be stored such that the neural network can be trained efficiently. 
        
        **Arguments**  
        - **`dataname`**  `(str)`: A name for the data to be prepared. Will determine the path for saving pre-processed data, as well as for saving embeddings by the embedder part of the summary quality model. 
        - **`data`**  `(iterable[SummaryReport]/LabelledReportData)`: Data to prepare for training/testing. 
        - **`overwrite`**  `(bool, optional)`: Boolean indicator of whether existing pre-processed data should be overwritten. If `False`, any pre-processed data found at the path based on `dataname` will be used instead of the data in `data` argument. Defaults to False.
        - **`overwrite_emb`**  `(bool, optional)`: Whether new embeddings should be made by embedder model. Whenever `overwrite_emb=True`, then `train_embedder`should also be `True`.  Defaults to False.
        - **`train_embedder`**  `(bool, optional)`: Whether embedder model should be trained. Defaults to False.
        
        **Returns**  `(str)`: Path to embedded data, ready for training neural network (returned path is used by the models, but should not be used by a user). 
        
        
    -   **`SummaryQualityModel.train(train_name, train_data, val_name=None, val_data=None, overwrite=False, overwrite_emb=True, train_embedder=True)`**  
    
        Train neural network part of data. Will first pre-process data (if overwrite=True/not already done), train embedder model (if train_embedder=True) and embed textual data (if overwrite_emb=True/not already done).
        
        **Arguments**  
        - **`train_name`**  `(str)`: A name for the training_data. Will determine the path for saving pre-processed data, as well as for saving embeddings by the embedder part of the summary quality model. 
        - **`train_data`**  `(LabelledReportData)`: Training dataset. 
        - **`val_name`**  `(str, optional)`: A name for the validation data. Will determine the path for saving pre-processed data, as well as for saving embeddings by the embedder part of the summary quality model.  Defaults to None (no val set used).
        - **`val_data`**  `(LabelledReportData, optional)`: Validation dataset. Defaults to None.
        - **`overwrite`**  `(bool, optional)`: Boolean indicator of whether existing pre-processed data should be overwritten. If `False`, any pre-processed data found at the path based on `train_name`/`val_name` will be used instead of the data in `train_data`/`val_data` argument. Defaults to False.
        - **`overwrite_emb`**  `(bool, optional)`: Whether new embeddings should be made by embedder model. Whenever `overwrite_emb=True`, then `train_embedder`should also be `True`.  Defaults to True.
        - **`train_embedder`**  `(bool, optional)`: Whether embedder model should be trained. Defaults to True.
        
        
    -   **`SummaryQualityModel.embed(data_name, data, overwrite=False, print_progress=True, overwrite_emb=False)`**  
    
        Return memory-friendly generator for embedded reports and summaries into the conceptual summary content space (see master thesis), ready for measuring quality (using cosine similarity). 
        
        **Arguments** 
        - **`data_name`**  `(str)`: A name for the data to embed. Will determine the path for saving pre-processed data, as well as for saving embeddings by the embedder part of the summary quality model. 
        - **`data`**  `(iterable[SummaryReport]/LabelledReportData)`: Data to create embeddings from. 
        - **`overwrite`**  `(bool, optional)`: Boolean indicator of whether existing pre-processed data should be overwritten. If `False`, any pre-processed data found at the path based on `data_name` will be used instead of the data in `data` argument. Defaults to False.
        - **`print_progress`**  `(bool, optional)`: Whether progress of embedding should be printed. Defaults to True.
        - **`overwrite_emb`**  `(bool, optional)`: Whether new embeddings should be made by embedder model. Whenever `overwrite_emb=True`, then the embedder part of the summary quality model must already be trained.  Defaults to False.
        
        **Returns**  `(generator[dict{'id', 'z_r', 'z_s'}])`: Return memory friendly generator with embedder reports and summaries. 
        
        
    -   **`SummaryQualityModel.predict(data_name, data, overwrite=False, overwrite_emb=True)`**  
    
        Return predicted qualities for input data. 
        
        **Arguments** 
        - **`data_name`**  `(str)`: A name for the data to predict quality of. Will determine the path for saving pre-processed data, as well as for saving embeddings by the embedder part of the summary quality model. 
        - **`data`**  `(iterable[SummaryReport]/LabelledReportData)`: Data to predict. 
        - **`overwrite`**  `(bool, optional)`: Boolean indicator of whether existing pre-processed data should be overwritten. If `False`, any pre-processed data found at the path based on `data_name` will be used instead of the data in `data` argument. Defaults to False.
        - **`overwrite_emb`**  `(bool, optional)`: Whether new embeddings should be made by embedder model. Whenever `overwrite_emb=True`, then the embedder part of the summary quality model must already be trained.  Defaults to True.
        
        **Returns**  `(pd.Series)`: Predicted qualities of report summaries, with ids in index. 
        
        
    -   **`SummaryQualityModel.save(modelname='default')`**  
    
        Save model using pickle. Not well tested, current implementation might have trouble with saving, especially when models are large. 
        
        **Arguments** 
        - **`modelname`**  `(str, optional)`: Save to path based on modelname. Defaults to 'default'.
        
        
    -   **`SummaryQualityModel.load(modelname='default')`**  
    
        Load model using pickle.
        
        **Arguments** 
        - **`modelname`**  `(str, optional)`: Load from path with based on modelname. Defaults to 'default'.
        
        **Returns**  `(SummaryQualityModel)`: self. 
      
        
    
