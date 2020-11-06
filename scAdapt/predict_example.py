import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import math

# seed_everything()
import torch
import numpy as np
seed=666
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import random
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

from utils import *
from networks import *
from evaluate import evaluate_multibatch

root_path = "../processed_data/"
print(root_path)
normcounts = pd.read_csv(root_path+'combine_baron.csv')
labels = pd.read_csv(root_path+'combine_labels_baron.csv')
domain_labels = pd.read_csv(root_path+'domain_labels_baron.csv')
data_set = {'features': normcounts.T.values, 'labels': labels.iloc[:, 0].values,
           'accessions': domain_labels.iloc[:, 0].values}

source_name = "TM_baron_mouse_for_baron"
target_name = "baron_human"
domain_to_indices = np.where(data_set['accessions'] == source_name)[0]
train_set = {'features': data_set['features'][domain_to_indices], 'labels': data_set['labels'][domain_to_indices],
             'accessions': data_set['accessions'][domain_to_indices]}
domain_to_indices = np.where(data_set['accessions'] == target_name)[0]
test_set = {'features': data_set['features'][domain_to_indices], 'labels': data_set['labels'][domain_to_indices],
            'accessions': data_set['accessions'][domain_to_indices]}
print('source labels:', np.unique(train_set['labels']), ' target labels:', np.unique(test_set['labels']))
test_set_eval = {'features': data_set['features'][domain_to_indices], 'labels': data_set['labels'][domain_to_indices],
                 'accessions': data_set['accessions'][domain_to_indices]}
print(train_set['features'].shape, test_set['features'].shape)
class_num = max(train_set['labels']) + 1
class_num_test = max(test_set['labels']) + 1


embedding_size = 256
base_network = FeatureExtractor(num_inputs=train_set['features'].shape[1], embed_size=embedding_size).cuda()
label_predictor = LabelPredictor(base_network.output_num(), class_num).cuda()
total_model = nn.Sequential(base_network, label_predictor)

### load trained model
model_file = './model_save/final_model_15000_TM_baron_mouse_for_baron_baron_human.ckpt'
print(model_file)
checkpoint = torch.load(model_file)
base_network.load_state_dict(checkpoint['base_network'])
label_predictor.load_state_dict(checkpoint['label_predictor'])
total_model = nn.Sequential(base_network, label_predictor)

### evaluation
from evaluate import evaluate_multibatch

feature_target = base_network(torch.FloatTensor(test_set['features']).cuda())
output_target = label_predictor.forward(feature_target)
softmax_out = nn.Softmax(dim=1)(output_target)
predict_prob_arr, predict_label_arr = torch.max(softmax_out, 1)

acc_by_label = np.zeros(class_num_test)
all_label = test_set['labels']
for i in range(class_num_test):
    acc_by_label[i] = np.sum(predict_label_arr.cpu().data.numpy()[all_label == i] == i) / np.sum(all_label == i)
np.set_printoptions(suppress=True)
print("average acc over all test cell types: ", round(np.nanmean(acc_by_label), 3),
      "acc of each test cell type: ", acc_by_label)

with torch.no_grad():
    code_arr_s = base_network(Variable(torch.FloatTensor(train_set['features']).cuda()))
    code_arr_t = base_network(Variable(torch.FloatTensor(test_set_eval['features']).cuda()))
    code_arr = np.concatenate((code_arr_s.cpu().data.numpy(), code_arr_t.cpu().data.numpy()), 0)
result_path = "../results/"
if not os.path.exists(result_path):
    os.makedirs(result_path)
div_score, div_score_all, ent_score, sil_score = evaluate_multibatch(code_arr, train_set, test_set_eval, epoch=0)

results_file = result_path + source_name + "_" + target_name + "_" + "_acc_div_sil.csv"
evel_res = [np.nanmean(acc_by_label), div_score, div_score_all, ent_score, sil_score]
pd.DataFrame(evel_res, index = ["acc","div_score","div_score_all","ent_score","sil_score"], columns=["values"]).to_csv(results_file, sep=',')

dict = {0:"alpha", 1: "beta", 2: "leukocyte", 3: "acinar", 4:"gamma", 5:"ductal", 6:"endothelial", 7:"delta", 8:"stellate", 9:"macrophage", 10:"B_cell"}
y_true_label = [dict[x] if x in dict else x for x in all_label]
y_pred_label = [dict[x] if x in dict else x for x in predict_label_arr.cpu().data.numpy()]
pred_labels_file = result_path + source_name + "_" + target_name + "_" + "_pred_labels.csv"
pd.DataFrame([y_pred_label, y_true_label],  index=["pred_label", "true_label"]).to_csv(pred_labels_file, sep=',')

embedding_file = result_path + source_name + "_" + target_name + "_" + "_embeddings.csv"
pd.DataFrame(code_arr).to_csv(embedding_file, sep=',')
