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

In our work, we use the following models in our results. 

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
    
    **Parameters**
    - **`id`**: ID of real-estate condition report.  
    - **`type`**: Not sure.  
    - **`date`**: Date of real-estate assessment.  
    - **`author`**: Author of real-estate condition report.  
    - **`building`**: Contains metadata about the building. `type Building`.  
    - **`cadastre`**: Contains metadata about the cadastre. `type Cadastre`.  
    - **`place`**: Contains placement info of the real-estate. `type list(PlaceDescription)`.  
    - **`condition`**: Contains condition info of the real-estate. `type list(ConditionDescription)`.  
  
  
-   #### `class SummaryReport(cr)`  

    Extends the `ConditionReport` class, with some useful methods for this project.  
    
    **Parameters**  
    - **`cr`**: A real-estate condition report. `type ConditionReport`.  
    
    **Methods**  
    -   ##### `SummaryReport.get_report_raw()`  
        **Return:** The complete report text. `type string`.  
    -   ##### `SummaryReport.get_summary_raw()`  
        **Return:** The complete summary text. `type string`.  
    -   ##### `SummaryReport.get_sections()`  
        **Return:** The sections of the report. `type list(string)`.  
    -   ##### `SummaryReport.get_report_words()`  
        **Return:** The words of the report. `type list(string)`.  
    -   ##### `SummaryReport.get_summary_words()`  
        **Return:** The words of the summary. `type list(string)`.  
    -   ##### `SummaryReport.get_report_sentences()`  
        **Return:** The sentences of the report. `type list(string)`.  
    -   ##### `SummaryReport.get_summary_sentences()`  
        **Return:** The sentences of the summary. `type list(string)`.  
    -   ##### `SummaryReport.get_tokenized_sections()`  
        **Return:** The tokenized sections of the report. `type list(list(string))`.  
    -   ##### `SummaryReport.get_report_tokenized_sentences()`  
        **Return:** The tokenized sentences of the report. `type list(list(string))`.  
    -   ##### `SummaryReport.get_summary_tokenized_sentences()`  
        **Return:** The tokenized sentences of the summary. `type list(list(string))`.  
    
    
## data.py
  
-   #### `class VenduData(path='data/VenduData/dataset', progress_bar=True, print_list=None, shuffle=1)`
  
    General purpuse iterable for data in WebDataset-format. This will be used throughout this project. 
  
    **Parameters**  
    - **`path`**: Path to area where data is stored. `type string`.  
    - **`progress_bar`**: Indicator of whether progress of loops through data should be shown. `type boolean`.  
    - **`print_list`**: For each iteration through data, the next string will be printed. `type list(string) or None`.  
    - **`shuffle`**: Shuffle the dataset with a buffer of size `shuffle`. `shuffle=1` will result in no shuffle. `type int`  
    
    **Methods**
    -   **`VenduData.make_dataset(iterable, func)`**  
    
        Make and save dataset to appropriate format from iterable. 
        
        **Parameters**  
        - **`iterable`**: Iterable of elements to add to dataset. 
        - **`func`**: Function to apply on elements in iterable. Should return (id, element). If `element==None`, it will be omitted. 
    

-   #### `class SubsetVenduData(data, subset)`  

    Class for iterating through only a subset of elements in data. Inherits from VenduData. 

    **Parameters**  
    - **`data`**: VenduData-object to make subset from. `type VenduData`.  
    - **`subset`**: Ids of reports to include. Should be a subset of the ids of the reports in `data`. `type list(string)`.  
    

-   #### `class LabelledVenduData(data, labels)`  

    Class for adding labels to a subset of elements in data. 
    Object will be an iterable of tuple(element, tuple(float, float)), where
    the first float represents the probability of the summary being bad, and the 
    second float is the probability of summary being good. 

    **Parameters**  
    - **`data`**: VenduData-object to make subset from. `type VenduData`.  
    - **`labels`**: Probabilistic labels, where report ids are expected to be found in the index. `type pandas.DataFrame` 


## weak_supervision.py 
  
-   #### `class GenerativeModel()`  

    Class for weak supervision generative model, used for making labels.  
    
    **Methods**  
  
    -   **`GenerativeModel.train(data)`**  
   
        Trains the model on the input data.  
        
        - **Parameter `data`**: Data for training the generative model. `type iterable(SummaryReport)` 
          
          
    -   **`GenerativeModel.predict(data)`**  
    
        Fit model on data.  
         
        - **Parameter `data`**: Data to be labelled by the model. `type iterable(SummaryReport)`.  
          
        - **Return:** Probabilistic labels for input data, with ids on the index. `type pandas.DataFrame`.  
          
          
    -   **`GenerativeModel.predict_training_set()`**  
    
        Fit model on training data. Much faster than calling `predict(data=training_data)`.  
      
        - **Return:** Probabilistic labels for the data that the model was trained on, with ids on the index. `type pandas.DataFrame`.  
          
            
    -   **`GenerativeModel.save(modelname='default')`**  
    
        Save model to file.  
        
        - **Parameter `modelname`**: Give name to the model. Filename will be based on this name. `type string`.  
          
          
    -   **`GenerativeModel.load(modelname='default')`**  
    
        Load model from file.  
      
        - **Parameter `modelname`**: Name of model to load. Filename is based on this name. `type string`.  
        
        - **Return:** Self, so that one can write `model = GenerativeModel.load(modelname)`. `type GenerativeModel`.  
   

## utils.py

  
-   #### `function train_val_test_split(labels, ratio=[0.8, 0.1, 0.1], seed=1)`  
  
    Split input data into a train, val and test set. 
  
    **Parameters**  
    - **`labels`**: Complete set of labels to divide into train/val/test. `type pandas.DataFrame`.  
    - **`ratio`**: Relative size that train/val/test set should have. `type list(float)`.  
    - **`seed`**: Random state to use for splitting. `type int`.  
    
    **Return**  
    - Labels for training set, with ids on the index. `type pandas.DataFrame`.  
    - Labels for validation set, with ids on the index. `type pandas.DataFrame`.    
    - Labels for test set, with ids on the index. `type pandas.DataFrame`.    
