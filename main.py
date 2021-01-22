import pickle
import bz2
from data import ReportData, SubsetReportData, LabelledReportData
from weak_supervision import WSLabelModel
import utils as ut
from models import LSAEmbedder, Doc2vecEmbedder, VocabularyEmbedder, Word2vecEmbedder 
from models import FFNModel, LSTMModel, CNNModel, SummaryQualityModel

data = ReportData(path='data/dataset')

# Initialize and train model
#label_model = WSLabelModel()
#label_model.train(data=data)

# Save trained model
#label_model.save(modelname='default')

# Load previously saved model
label_model = WSLabelModel().load(modelname='default')

# Predict labels on the data that the model was trained on
labels = label_model.predict_training_set()

# print("\nLabeling function analysis:")
# label_model.analyse_training_set()
# print("\nEstimated labeling function accuracies:")
# label_model.print_estimated_accuracies()
# label_model.plot_training_labels()

labels_train, labels_val, labels_test = ut.train_val_test_split(labels, ratio=[0.8, 0.1, 0.1], seed=1)
train = LabelledReportData(data=data, labels=labels_train)
val = LabelledReportData(data=data, labels=labels_val)
test = SubsetReportData(data=data, subset=labels_test.index)

embedder = LSAEmbedder()
h = FFNModel()
model = SummaryQualityModel(embedder=embedder, model=h)

# train method performs steps 2-4 by default, and step 1 only if it has not been done before. 
model.train(train_name='train', train_data=train, val_name='validation', val_data=val, train_embedder=False, overwrite_emb=False)

# Predict qualities for validation and test set
q_val = model.predict(data_name='validation', data=val)
q_test = model.predict(data_name='test', data=test)

q_val.to_csv('predictions/LSA+LinTrans_val.csv')
q_test.to_csv('predictions/LSA+LinTrans_test.csv')