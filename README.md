# SummaryQuality

This project is part of a master thesis on NTNU, where the goal is to measure the quality of summaries of real-estate condition reports. In this project, weak supervision is first used to obtain noisy labels. Then, various models are trained on these noisy labels, such that they learn to measure the quality of summaries for real-estate condition reports. 

It must be noted that the models are entirely based on weak supervision. This means that the models have **not** been evaluated on data where the true summary quality is known. It is therefore uncertain how accurate the models truly are. Any results should be regarded as an **indication** only, and not as a ground truth. 

The following explains the functionality of this project. A more detailed documentation is found at the end. 

# Usage

To work with this project, the necessary files should be copied into the working directory. This includes the following: 
- **common.py**: Classes for containing real-estate condition reports. 
- **data.py**: Classes for making, storing and iterating through datasets. 
- **weak_supervision.py**: Implements a weak supervision label model based on the Snorkel-framework, used for making labels. 
- **labeling_functions.py**: Contains the labeling functions (with helper functions) that the label model in weak_supervision.py is based on. 
- **models.py**: Implements the various models used in this work. 
- **networks.py**: Implements neural network architectures used by the models in `models.py`. 
- **utils.py**: Contains useful functions for analysing performance, and various other stuff. 

The necessary packages for these files are found in requirements.txt. 

### Data structure

Vendu has implemented the class `ConditionReport`, which represents a real-estate condition report. In this project, the class `SummaryReport` has also been implemented, which inherits from the `ConditionReport` class, and has implemented some additional methods that are useful for this project. 

The models of this work expect data input as an iterable of SummaryReport objects (with corresponding labels, when appropriate). In this project, the iterable class `ReportData` has been implemented for creating and representing datasets. This makes use of the WebDataset package, which stores data in .tar archives that can be streamed from disk, such that iterating becomes memory-friendly.  This class is used by our models for storing data at different stages (pre-processed, embedded, etc.), and is also convenient to use for our dataset. A dataset of SummaryReports can be made from an iterable of ConditionReport-objects the following way:

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
The above code will save the data to file at './data/dataset', and will only have to run once. Afterwards, the dataset can be initialized by:
```python
data = ReportData(path='data/dataset')

for idx, report in enumerate(data):
    print(report.id)
    if idx > 5:
        break
```

### Making weak supervision labels

Now that a dataset of appropriate structure has been created, we can train a weak supervision label model to make weak supervision labels. Note that the current implementation expects the following files to exist: './data/VenduData/matched_policies.zip' and 'data/VenduData/claims.csv'. 

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

print("\nLabeling function analysis:")
label_model.analyse_training_set()
print("\nEstimated labeling function accuracies:")
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
Note that the train-method by default always performs steps 2-4, while step 1 is only performed if it has not been performed before. The steps can be controlled the following way: 

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
# Get qualitites for val and test set from saved files
quality_val = pd.read_csv('predictions/LSA+LinTrans_val.csv', index_col=0, squeeze=True)
quality_test = pd.read_csv('predictions/LSA+LinTrans_test.csv', index_col=0, squeeze=True)

# Calculate training loss and accuracy scores
loss = ut.noise_aware_cosine_loss(quality_test, labels_test, 0.2, -0.2)
best_threshold = ut.get_best_threshold(quality_val, labels_val)
acc, prec, rec, f1 = ut.classification_scores(quality_test, labels_test, best_threshold)

# Print loss and classification scores and plot distribution of summary qualitity
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
Word2vec+CNN = SummaryQualityModel(embedder=Word2vec, model=CNN)
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
  
    Iterable class object for storing and iterating over data. Used extensively in this project for storing data in different formats that makes training faster and more practical. Takes a path as input, and can either iterate over existing data at this path, or create new data to path. Various arguments for different use cases. Stores data in WebDataset format (tar archives), for memory-friendly reading. 
  
    **Arguments**  
    - **`path`**  `(str)`: Path to store/read data to/from. 
    - **`print_progress`**  `(bool, optional)`: Whether to print progress in iterations. Defaults to True.
    - **`shuffle_buffer_size`**  `(int, optional)`: WebDataset shuffles data by putting elements into a buffer with given size. Defaults to 1 (no shuffle).
    - **`apply`**  `(func, optional)`: Apply function to elements in data. Defaults to None (no func).
    - **`batch_size`**  `(int, optional)`: Data can be loaded in batches. Defaults to 1 (no batching).
    - **`collate`**  `([func, optional)`: Apply func to batches. Used for PackedSequence stuff with LSTM. Defaults to None.
    
    **Methods**
    -   **`ReportData.create(self, data, apply=None, overwrite=False)`**  
    
        Store dataset to path from input data. Can apply function to data before storing. 
        
        **Arguments**  
        - **`data`**  `(iterator)`: Any iterator type. 
        - **`apply`**  `(func, optional)`: Apply any function to data elements before storing. Defaults to transforming ConditionReport object to SummaryReport object. 
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
    
        Fit model on data. If data is the same as the label model was trained on, it is much faster to use self.predict_training_set. 
         
        - **Argument `data`**  `(iterable[SummaryReport])`: Data to predict labels on.
          
        - **Return** `(pd.DataFrame)`: Probabilistic labels for data, with ids in index. 
          
          
    -   **`WSLabelModel.predict_training_set()`**  
    
        Predict labels on the set that was used for training the model. Much faster than calling `predict(data=training_data)`, since it takes time to prepare dataset for labeling functions. 

      
        - **Return**  `(pd.DataFrame)`: Probabilistic labels for the data that the model was trained on, with ids in the index. 
          
    -   **`WSLabelModel.analyse_training_set(latexpath=None)`**  
    
        Perform analysis of labeling functions. Prints coverage, overlap and conflict rates for the labeling functions. 
        
        - **Parameter `latexpath`**  `(str, optional)`: Save analysis to txt file, in latex table format. Defaults to None (no saving to file).
        
        
    -   **`WSLabelModel.print_estimated_accuracies()`**  
    
        Print estimated accuracy for the labeling functions.
        
        
    -   **`WSLabelModel.plot_training_labels(name=None)`**  
    
        Plot estimated weak supervision labels for training set.
        
        - **Argument `name`**  `(str, optional)`: Filename for saving plot. Defaults to None (no saving).
            
            
    -   **`WSLabelModel.save(modelname='default')`**  
    
        Save model to file.  
        
        - **Parameter `modelname`**  `(str, optional)`: Filename for saving. Defaults to 'default'.
          
          
    -   **`WSLabelModel.load(modelname='default')`**  
    
        Load previously saved label model from file. 
      
        - **Parameter `modelname`**  `(str, optional)`: Filename to load from. Defaults to 'default'.
        
        - **Return**  `(WSLabelModel)`: Self, so that one can write `model = WSLabelModel().load(modelname)`. 
   

## utils.py

  
-   #### `function train_val_test_split(labels, ratio=[0.8, 0.1, 0.1], seed=1)`  
  
    Split input data into a train, val and test set. 
  
    **Arguments**  
    - **`labels`**  `(pd.DataFrame)`: Labels with report ids in index. 
    - **`ratio`**  `(list, optional)`: Size of train, val and test. Defaults to [0.8, 0.1, 0.1].
    - **`seed`**  `(int, optional)`: Seed for shuffling. Defaults to 1.
    
    **Return**  
    - **`probs_train`**  `(pd.DataFrame)`: Dataframe with labels for training set. 
    - **`probs_val`**  `(pd.DataFrame)`: Dataframe with labels for validation set. 
    - **`probs_test`**  `(pd.DataFrame)`: Dataframe with labels for test set. 
    
    
-   #### `function plot_quality(quality, labels=False, name=False, title=False, lab=False, show=True, all=False, density=True, xlim=True)`  
  
    Plot two distributions, one for good summaries, and one for bad.
  
    **Arguments** 
    - **`quality`**  `(pd.Series)`: Quality measures to plot. Ids are expected in index. 
    - **`labels`**  `(pd.DataFrame, optional)`: Labels to use for determining which are good and bad summaries. Defaults to False, in which case only one distribution is shown.
    - **`name`**  `(str, optional)`: Save name. Defaults to False (no saving).
    - **`title`**  `(str, optional)`: Title. Defaults to False (No title).
    - **`lab`**  `(bool, optional)`: Whether labels should be shown in plot. Defaults to False.
    - **`show`**  `(bool, optional)`: Whether to show plot. Defaults to True.
    - **`all`**  `(bool, optional)`: Whether distribution of all qualitites should be shown in grey background. Defaults to False.
    - **`density`**  `(bool, optional)`: Whether normalization of distributions should be performed. Defaults to True.
    - **`xlim`**  `(bool, optional)`: Whether x-axis should be set to [-1, 1]. Defaults to True.
    

-   #### `function noise_aware_cosine_loss(quality, labels, tau_good, tau_bad)`  
  
    Calculates a noise aware cosine similarity loss for a set of qualities/labels.
  
    **Arguments**  
    - **`quality`**  `(pd.Series)`: Predicted qualities, with ids on index. 
    - **`labels`**  `(pd.DataFrame)`: Labels, with ids on index. 
    - **`tau_good`**  `(float)`: Which tau_good to use for loss function. 
    - **`tau_bad`**  `(float)`: Which tau_bad to use for loss function. 
    
    **Return**  `(float)`: Noise aware cosine embedding loss for input qualities and labels. 


-   #### `function get_best_threshold(quality, labels)`  
  
    Get maximum-accuracy threshold to use for classification.
  
    **Arguments**  
    - **`quality`**  `(pd.Series)`: Predicted qualities, with ids on index. 
    - **`labels`**  `(pd.DataFrame)`: Labels, with ids on index. 
    
    **Return**  `(float)`: Threshold that maximizes accuracy. 


-   #### `function classification_scores(quality, labels, best_threshold)`  
  
    Calculates accuracy, recall, precision and F1-score on the labels, by using given threshold for classifying.
  
    **Arguments**  
     - **`quality`**  `(pd.Series)`: Predicted qualities, with ids on index. 
    - **`labels`**  `(pd.DataFrame)`: Labels, with ids on index. 
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
    - **`no_above`**  `(int, optional)`: Maximum fraction of documents a word can be in for it to be included in the vocabulary. Defaults to 1.
    - **`filter_most_frequent`**  `(int, optional)`: Most frequent words removed. Defaults to 0.
    - **`remove_stopwords`**  `(bool, optional)`: Whether stopwords should be removed. Defaults to False.
   
  
-   #### `class Doc2vecEmbedder(dim=100, window=6, mc=20, workers=4, dm=0, epochs=50)`  
  
    Vocabulary embedder, for use together with CNN when making our own word embeddings. 
  
    **Arguments**  
    - **`dim`**  `(int, optional)`: Dimensionality of doc embeddings. Defaults to 100.
    - **`window`**  `(int, optional)`: Window size in Word2vec. Defaults to 6.
    - **`mc`**  `(int, optional)`: Number of times word must appear to be included in vocabulary. Defaults to 20.
    - **`workers`**  `(int, optional)`: Workers in training process. Defaults to 4.
    - **`dm`**  `(int, optional)`: Whether PV-DM version should be used. Defaults to 0 (PV-DBOW).
    - **`epochs`**  `(int, optional)`: Number of training epochs in Doc2vec. Defaults to 50.
    
    
-   #### `class VocabularyEmbedder(vocab_length=20000)`  
  
    Doc2vec embedder, to be used together with FFN or LSTM. 
  
    **Argument `vocab_length`**  `(int, optional)`: Vocabulary size. Defaults to 20000.
    
    
-   #### `class Word2vecEmbedder(dim=100, window=10, min_count=20, workers=4, epochs=50)`  
  
    Vocabulary embedder, for use together with CNN when making our own word embeddings. 
  
    **Arguments**  
    - **`dim`**  `(int, optional)`: Dimensionality of doc embeddings. Defaults to 500.
    - **`window`**  `(int, optional)`: Window size in Word2vec. Defaults to 10.
    - **`mc`**  `(int, optional)`: Number of times word must appear to be included in vocabulary. Defaults to 20.
    - **`workers`**  `(int, optional)`: Workers in training process. Defaults to 4.
    - **`dm`**  `(int, optional)`: Whether PV-DM version should be used. Defaults to 0 (PV-DBOW).
    - **`epochs`**  `(int, optional)`: Number of training epochs in Doc2vec. Defaults to 50.
    
