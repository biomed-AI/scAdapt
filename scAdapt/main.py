import argparse
import pandas as pd
from scAdapt import train

def preprocess():
    root_path = "../processed_data/"
    print(root_path)
    normcounts = pd.read_csv(root_path+'combine_baron.csv')
    labels = pd.read_csv(root_path+'combine_labels_baron.csv')
    domain_labels = pd.read_csv(root_path+'domain_labels_baron.csv')
    data_set = {'features': normcounts.T.values, 'labels': labels.iloc[:, 0].values,
               'accessions': domain_labels.iloc[:, 0].values}
    return data_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scAdapt: virtual adversarial domain adversarial network')
    parser.add_argument('--method', type=str, default='DANN', choices=['DANN', 'mmd'])
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--embedding_size', type=int, default=256, help='embedding_size')
    parser.add_argument('--source_name', type=str, default='TM_baron_mouse_for_baron')
    parser.add_argument('--target_name', type=str, default='baron_human')
    parser.add_argument('--num_iterations', type=int, default=50010, help="num_iterations")
    parser.add_argument('--BNM_coeff', type=float, default=0.2, help="regularization coefficient for BNM loss")
    parser.add_argument('--centerloss_coeff', type=float, default=1.0,  help='regularization coefficient for center loss')
    parser.add_argument('--semantic_coeff', type=float, default=1.0, help='regularization coefficient for semantic loss')
    parser.add_argument('--DA_coeff', type=float, default=1.0, help="regularization coefficient for domain alignment loss")
    parser.add_argument('--pseudo_th', type=float, default=0.0, help='pseudo_th')
    parser.add_argument('--cell_th', type=int, default=20, help='cell_th')
    parser.add_argument('--epoch_th', type=int, default=15000, help='epoch_th')
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient for VAT loss')
    parser.add_argument('--xi', type=float, default=10.0, metavar='XI',
                        help='hyperparameter of VAT')
    parser.add_argument('--eps', type=float, default=1.0, metavar='EPS',
                        help='hyperparameter of VAT')
    parser.add_argument('--ip', type=int, default=1, metavar='IP',
                        help='hyperparameter of VAT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  #'0,1,2,3'
    print(args)
    data_set = preprocess()
    train(args, data_set=data_set)


