import os
import sys
import numpy as np

sys.path.append(os.getcwd())

# Our infrastucture files
# from utils_data import * 
# from utils_nn import *
from learn.utils.data import *
from learn.utils.nn import *
import learn.utils.matplotlib as u_p
from learn.utils.plotly import plot_test_train, plot_dist, quick_iono
# neural nets
from learn.models.model_general_nn import GeneralNN
from learn.models.model_ensemble_nn import EnsembleNN
from learn.models.linear_model import LinearModel

# import omegaconf
# Torch Packages
import torch
import pandas as pd

# timing etc
import os
# import hydra

# Plotting
import matplotlib.pyplot as plt
import time


import logging

log = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_file(object, filename):
    path = os.path.join(os.getcwd(), filename)
    log.info(f"Saving File: {filename}")
    torch.save(object, path)

# ??????????????????????????????????????????????????????????????
def create_model_params(df, model_cfg):
    # only take targets from robot.yaml
    target_keys = []
    if not model_cfg.params.delta_state_targets == False:
        for typ in model_cfg.params.delta_state_targets:
            target_keys.append(typ + '_0dx')
    if not model_cfg.params.true_state_targets == False:
        for typ in model_cfg.params.true_state_targets:
            target_keys.append(typ + '_1fx')

    # grab variables
    history_states = df.filter(regex='tx')
    history_actions = df.filter(regex='tu')

    # print('history_states',history_states)
    # print('history_actions',history_actions)

    # trim past states to be what we want
    history = int(history_states.columns[-1][-3])
    if history > model_cfg.params.history:
        print('history > model_cfg.params.history')
        for i in range(history, model_cfg.params.history, -1):
            str_remove = str(i) + 't'
            for state in history_states.columns:
                if str_remove in state:
                    history_states.drop(columns=state, inplace=True)
            for action in history_actions.columns:
                if str_remove in action:
                    history_actions.drop(columns=action, inplace=True)

    # add extra inputs like objective function
    extra_inputs = []
    if model_cfg.params.extra_inputs:
        print('extra_inputs')
        for extra in model_cfg.params.extra_inputs:
            df_e = df.filter(regex=extra)
            extra_inputs.append(df_e)
            history_actions[extra] = df_e.values

    # ignore states not helpful to prediction
    for ignore in model_cfg.params.ignore_in:
        print('ignore',ignore)
        for state in history_states.columns:
            if ignore in state:
                history_states.drop(columns=state, inplace=True)

    params = dict()
    params['targets'] = df.loc[:, target_keys]
    params['states'] = history_states
    params['inputs'] = history_actions

    return params


def params_to_training(data):
    X = data['states'].values
    U = data['inputs'].values
    dX = data['targets'].values
    return X, U, dX


def train_model(X, U, dX, logged=False, model_use=None):
    if logged: log.info(f"Training Model on {np.shape(X)[0]} pts")
    start = time.time()
    train_log = dict()
    
    # print(model_cfg)
    # train_log['model_params'] = model_cfg.params
    # print(model_cfg.params)
    if model_use is None:
        model = GeneralNN().to(device)
    else:
        model = model_use

    X_t = X.squeeze()
    U_t = U
    dX_t = dX.squeeze()

    print('X', np.shape(X_t))
    print('U', np.shape(U))
    print('dX', np.shape(dX_t))

    # print('Neural network',model)
    acctest, acctrain = model.train_cust((X_t, U_t, dX_t))

    min_err = np.min(acctrain)
    min_err_test = np.min(acctest)

    train_log['testerror'] = acctest
    train_log['trainerror'] = acctrain
    train_log['min_trainerror'] = min_err
    train_log['min_testerror'] = min_err_test

    end = time.time()
    if logged: log.info(f"Trained Model in {end-start} s")
    return model, train_log


######################################################################
# @hydra.main(config_path='/home/PINN_UAV/JC-dev/Code/ROS_ws/src/motion_capture_fake/script/learn/conf', config_name='trainer.yaml')
# def trainer(cfg):
#     log.info("============= Configuration =============")
#     # log.info(f"Config:\n{cfg.pretty()}")
#     log.info("=========================================")
#     log.info('Training a new model')

#     #######################################
#     data_dir = cfg.robot.load.fname  # base_dir

#     # avail_data = os.path.join(os.getcwd()[:os.getcwd().rfind('outputs') - 1] + f"/ex_data/SAS/{cfg.robot}.csv")
#     #os.path.isfile(avail_data): 
#     df = pd.read_csv(data_dir)
#     log.info(f"Loaded preprocessed data from {data_dir}")
#     #######################################

#     data = create_model_params(df, cfg.robot.model)
#     # print(data)

#     X, U, dX = params_to_training(data)
#     print('X',np.shape(X))
#     print('U',np.shape(U))
#     print('dX',np.shape(dX))


#     model, train_log = train_model(X, U, dX, cfg.models.model)
#     model.store_training_lists(list(data['states'].columns),
#                                list(data['inputs'].columns),
#                                list(data['targets'].columns))

#     # mse = plot_test_train(model, (X, U, dX), variances=True)
#     # torch.save((mse, cfg.model.params.training.cluster), 'cluster.dat')

#     # log.info(f"MSE of test set predictions {mse}")
#     msg = "Trained Model..."
#     msg += "Prediction List" + str(list(data['targets'].columns)) + "\n"
#     msg += "Min test error: " + str(train_log['min_testerror']) + "\n"
#     msg += "Mean Min test error: " + str(np.mean(train_log['min_testerror'])) + "\n"
#     msg += "Min train error: " + str(train_log['min_trainerror']) + "\n"
#     log.info(msg)

#     if True:
#         plt.figure(2)
#         ax1 = plt.subplot(211)
#         ax1.plot(train_log['testerror'], label='Test Loss')
#         plt.title('Test Loss')
#         ax2 = plt.subplot(212)
#         ax2.plot(train_log['trainerror'], label='Train Loss')
#         plt.title('Training Loss')
#         ax1.legend()
#         # plt.show()
#         plt.savefig(os.path.join(os.getcwd() + '/modeltraining.png'))

#     # Saves NN params
#     if True:
#         file_order = 'test'
#         save_file(model, file_order + '.pth')

#         normX, normU, normdX = model.getNormScalers()
#         save_file((normX, normU, normdX), file_order + '_normparams.pkl')

#         # Saves data file
#         save_file(data, file_order + '_data.pkl')
#         log.info(f"Saved to directory {os.getcwd()}")

# if __name__ == '__main__':
#     sys.exit(trainer())
#     # my_train()


if __name__ == '__main__':
    
    file_order = '1000'
    X_path = 'sim_origin_data/5hz/' + file_order + '_X.txt'
    U_path = 'sim_origin_data/5hz/' + file_order + '_U.txt'
    dX_path = 'sim_origin_data/5hz/' + file_order + '_dX.txt'

    X = np.loadtxt(X_path, dtype='float')
    U = np.loadtxt(U_path, dtype='float')
    print(np.max(U))
    dX = np.loadtxt(dX_path, dtype='float')
    model, train_log = train_model(X, U, dX)
    # save model and train log
    torch.save(model.state_dict(), 'model.pth')
    np.save('train_log.npy', train_log)
    # plot loss
    train_log = np.load('train_log.npy', allow_pickle=True).item()
    train_loss = train_log['trainerror']
    test_loss = train_log['testerror']
    fig = plt.figure(1)
    loss_plot = fig.add_subplot(1,1,1)
    plt.ion()
    loss_plot.plot(np.arange(0, len(train_loss)), train_loss, label='Training Loss')
    loss_plot.plot(np.arange(0, len(test_loss)), test_loss, label='Validation Loss')
    loss_plot.set_xlabel('epoches')
    loss_plot.set_ylabel('loss')
    loss_plot.legend()

    plt.ioff()
    plt.show()

    # for i in range(1000):

    #     Xt = X[i, :]
    #     Ut = U[i, :]
        
    #     Xtensor = torch.from_numpy(Xt.reshape(1, 3)).to(device)
    #     print(Xtensor)
    #     Utensor = torch.from_numpy(Ut.reshape(1, 3)).to(device)
    #     print(Utensor)
    #     dXtensor = model.predict_cuda(Xtensor, Utensor)
    #     dx_p = dXtensor.cpu().detach().numpy()
    #     e = dX[i, :] - dXtensor
    #     print(e)