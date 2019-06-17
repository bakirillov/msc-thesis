#!/usr/bin/env python
# coding: utf-8


# ### Necessary imports

# In[1]:


import os
import umap
import torch
import itertools
import numpy as np
import pandas as pd
from torch import nn
import os.path as op
from tqdm import tqdm
from time import time
from copy import deepcopy
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
from capsules.capsules import *
from torch.autograd import Variable
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error


# ### Helpers

# In[2]:


def iterate_minibatches(X, y, batchsize, permute=False):
    indices = np.random.permutation(np.arange(len(X))) if permute else np.arange(len(X))
    for start in range(0, len(indices), batchsize):
        ix = indices[start: start + batchsize]
        yield X[ix], y[ix]


# In[3]:


def moving_average(net1, net2, alpha=1):
    """Moving average over weights as described in the SWA paper"""
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


# In[4]:


def onehot(u):
    encoding = {
        1: [1,0,0,0],
        2: [0,0,0,1],
        3: [0,1,0,0],
        4: [0,0,1,0],
        0: [0,0,0,0],
        "A": [1,0,0,0],
        "T": [0,0,0,1],
        "G": [0,1,0,0],
        "C": [0,0,1,0],
        "N": [0,0,0,0],
        "a": [1,0,0,0],
        "t": [0,0,0,1],
        "g": [0,1,0,0],
        "c": [0,0,1,0],
        "n": [0,0,0,0]
    }
    r = np.array(sum([encoding[a] for a in u], []))
    return(r)


# In[5]:


def correct_order(u):
    return(
        u.reshape((4,int(u.shape[0]/4)), order="f").reshape(u.shape[0])
    )


# ### Model class

# In[6]:


class GuideCaps(nn.Module):
    
    def __init__(self, guide_length, n_routes=120):
        super(GuideCaps, self).__init__()
        self.gl = guide_length
        self.n_routes = n_routes
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=4, 
                out_channels=80,
                kernel_size=5,
                stride=1
            ),
            nn.ReLU(inplace=True)
        )
        #self.conv = nn.Sequential(
        #    nn.Conv1d(
        #        in_channels=4, 
        #        out_channels=256,
        #        kernel_size=5,
        #        stride=1
        #    ),
        #    nn.ELU(inplace=True),
        #    nn.Conv1d(
        #        in_channels=256, 
        #        out_channels=512,
        #        kernel_size=1,
        #        stride=1
        #    ),
        #    nn.ELU(inplace=True)
        #)
      #  self.conv = nn.Sequential(
      #      nn.Conv2d(
      #          in_channels=1, 
      #          out_channels=512,
      #          kernel_size=2,
      #          stride=1
      #      ),
      #      nn.ReLU(inplace=True)
      #  )
        self.primcaps = PrimaryCapsuleLayer(
            n_capsules=128, in_ch=80, out_ch=16, kernel_size=(1,2),
            stride=1
        )
        self.classcaps = SecondaryCapsuleLayer(
            n_capsules=2, n_iter=5, in_ch=128, out_ch=32, n_routes=self.n_routes #120
        )
        self.decoder = RegularizingDecoder(
            dims=[64,128,256,4*self.gl]
        )
        
    def forward(self, x):
        co = self.conv(x)#.view(x.size(0), 1, x.size(1), x.size(2)))
        co = co.view(co.size(0), co.size(1), 1, co.size(2))
        pc = self.primcaps(co)
        internal = self.classcaps(pc)
        lengths = F.softmax((internal**2).sum(dim=-1)**0.5, dim=-1)
        _, max_caps_index = lengths.max(dim=-1)
        masked = Variable(torch.eye(2))
        masked = masked.cuda() if torch.cuda.is_available() else masked
        masked = masked.index_select(dim=0, index=max_caps_index.data)
        masked_internal = (internal*masked[:,:,None]).view(x.size(0), -1)
        reconstruction = self.decoder(masked_internal)
        return(internal, reconstruction, lengths, max_caps_index, masked_internal)
    
    def predict(self, X):
        test_X = Variable(
            torch.from_numpy(X.reshape(X.shape[0],4,int(X.shape[1]/4))).type(torch.FloatTensor)
        )
        test_X = test_X.cuda() if torch.cuda.is_available() else test_X
        test_int, test_rec, test_len, test_classes, masked_int = self.forward(test_X)
        t_i = test_int.cpu().data.numpy()
        t_l = test_len.cpu().data.numpy()
        t_r = test_rec.cpu().data.numpy()
        t_c = test_classes.cpu().data.numpy()
        m_i = masked_int.cpu().data.numpy()
        return(t_i, t_r, t_l, t_c, m_i)
    
    def batch_predict(self, X, batch_size, verbose=False):
        t_i = []
        t_r = []
        t_l = []
        t_c = []
        m_i = []
        f = tqdm if verbose else lambda x: x
        for batch_X, _ in f(iterate_minibatches(X, X, batch_size, False)):
            a, b, c, d, e = self.predict(batch_X)
            t_i.append(a)
            t_r.append(b)
            t_l.append(c)
            t_c.append(d)
            m_i.append(e)
        t_i = np.concatenate(t_i)
        t_r = np.concatenate(t_r)
        t_l = np.concatenate(t_l)
        t_c = np.concatenate(t_c)
        m_i = np.concatenate(m_i)
        return(t_i, t_r, t_l, t_c, m_i)
    
    def validate(self, X, y, batch_size_test, capsule_loss):
        correct_ones = 0
        aggregate_loss = 0
        for batch_X, batch_y in iterate_minibatches(X, y, batch_size_test):
            test_X = Variable(
                torch.from_numpy(batch_X.reshape(batch_X.shape[0],4,int(X.shape[1]/4))).type(torch.FloatTensor)
            ).cuda()
            test_Y = Variable(
                make_y(torch.from_numpy(batch_y).type(torch.LongTensor).cuda(), 2)
            )
            test_internal, test_reconstruction, test_lengths, test_classes, _ = self.forward(test_X)
            aggregate_loss = aggregate_loss + capsule_loss(
                    test_Y, test_X.reshape(test_X.size(0), X.shape[1]), 
                    test_lengths, test_reconstruction
                ).cpu().data.numpy()
            pred_y = test_classes.cpu().data.numpy()
            correct_ones += np.sum(
                [1 if a == b else 0 for a,b in zip(pred_y, batch_y)]
            )
        return(aggregate_loss, correct_ones/X.shape[0])
    
    def fit(self, X, y, test_x, test_y, optimizer, capsule_loss, n_epochs, batch_sizes):
        training_loss = []
        training_accuracy = []
        testing_loss = []
        testing_accuracy = []
        start_time = time()
        batch_size_train = batch_sizes[0]
        batch_size_test = batch_sizes[1]
        best_test_acc = 0
        best_model = deepcopy(self)
        for epoch in tqdm(range(n_epochs)):
            self.train()
            running_loss = []
            running_accuracy = []
            epoch_time = time()
            for batch_X, batch_y in iterate_minibatches(X, y, batch_size_train):
                optimizer.zero_grad()
                inp = Variable(
                  torch.from_numpy(
                      batch_X.reshape(batch_X.shape[0],4,int(X.shape[-1]/4))).type(torch.FloatTensor)
                ).cuda()
                real_class = Variable(
                    make_y(torch.from_numpy(batch_y).type(torch.LongTensor).cuda(), 2)
                )
                internal, reconstruction, lengths, classes, _ = self.forward(inp)
                loss = capsule_loss(
                    real_class, inp.view(inp.size(0), int(X.shape[-1])), lengths, reconstruction
                )
                loss.backward()
                optimizer.step()
                running_loss.append(loss.cpu().data.numpy())
                running_accuracy.append(
                    accuracy_score(classes.cpu().data.numpy(), batch_y)
                )
            training_loss.append(np.mean(running_loss))
            training_accuracy.append(np.mean(running_accuracy))
            self.eval()
            test_loss, test_acc = self.validate(test_x, test_y, batch_size_test, capsule_loss)
            testing_loss.append(test_loss)
            testing_accuracy.append(test_acc)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model = deepcopy(self)
        print("All epochs took "+str((time() - start_time)/60) + " minutes")
        return(
            best_model, training_loss, training_accuracy, testing_loss, testing_accuracy
        )
    
    @classmethod
    def learning_curve(
        cls, guide_length, n_routes, metric, percentages, X, Y, val_X, val_Y, n_epochs, batch_sizes, fname
    ):
        test_mt = []
        val_mt = []
        train_mt = []
        for a in percentages:
            print(str(int(a*100))+"% of the dataset")
            model = cls(guide_length, n_routes).cuda()
            train_X, test_X, train_Y, test_Y = train_test_split(
                X, Y, train_size=a
            )
            optimizer = Adam(model.parameters())
            capsule_loss = CapsuleLoss().cuda()
            best_model, _, _, _, _, _, _ = model.fit(
                train_X, train_Y, test_X, test_Y, val_X, val_Y, optimizer, capsule_loss, n_epochs, batch_sizes
            )
            best_model.eval()
            _, _, _, train_labels, _ = best_model.batch_predict(train_X, batch_sizes[1], verbose=False)
            _, _, _, test_labels, _ = best_model.batch_predict(test_X, batch_sizes[1], verbose=False)
            _, _, _, val_labels, _ = best_model.batch_predict(val_X, batch_sizes[1], verbose=False)
            test_mt.append(metric(test_Y, test_labels))
            val_mt.append(metric(val_Y, val_labels))
            train_mt.append(metric(train_Y, train_labels))
            torch.save(best_model, fname.replace(".ptch", ".lc.best."+str(a)+".ptch"))
            del model
            del best_model
            torch.cuda.empty_cache()
        return(test_mt, val_mt, train_mt)
        
    def form_fasta_headers(self, strs, batch_size):
        TMPLT = ">NUM;class=CL;lengths=LEN;"
        pics = np.stack([correct_order(onehot(a+"GG")) for a in strs])
        _, _, t_l, t_c, _ = self.batch_predict(pics, batch_size)
        h = []
        for i,a in enumerate(strs):
            c = str(t_c[i])
            l = np.array2string(t_l[i])
            s = TMPLT.replace("NUM", str(i)).replace("CL", c).replace("LEN", l)
            h.append(s)
        return(h)
            
    @classmethod
    def stratified_kfold(cls, gl, k, X, y, n_epochs, batch_sizes, swa, fname):
        kfold = StratifiedKFold(n_splits=k)
        accs = []
        precs = []
        recs = []
        mccs = []
        swa_accs = []
        swa_precs = []
        swa_recs = []
        swa_mccs = []
        i = 1
        for train_index, test_index in kfold.split(X, y):
            model = cls(gl).cuda()
            optimizer = Adam(model.parameters())
            swa_model = deepcopy(model) if swa else None
            capsule_loss = CapsuleLoss().cuda()
            print(str(i)+"th fold")
            i += 1
            current_x = X[train_index]
            current_y = y[train_index]
            current_xt = X[test_index]
            current_yt = y[test_index]
            best_model, swa_model, _, _, _, _, _, _ = model.fit(
                current_x, current_y, current_xt, current_yt, 
                optimizer, capsule_loss, n_epochs, batch_sizes, swa_model
            )
            best_model.eval()
            #_, acc = best_model.validate(current_xt, current_yt, batch_sizes[0], capsule_loss)
            _, _, _, bm_predictions, _ = best_model.batch_predict(current_xt, batch_sizes[0])
            acc = accuracy_score(bm_predictions, current_yt)
            mcc = matthews_corrcoef(bm_predictions, current_yt)
            prec = precision_score(bm_predictions, current_yt)
            rec = recall_score(bm_predictions, current_yt)
            accs.append(acc)
            precs.append(prec)
            mccs.append(mcc)
            recs.append(rec)
            if swa:
                swa_model.eval()
                #_, swa_acc = swa_model.validate(current_xt, current_yt, batch_sizes[0], capsule_loss)
                _, _, _, swa_predictions, _ = swa_model.batch_predict(current_xt, batch_sizes[0])
                swa_acc = accuracy_score(swa_predictions, current_yt)
                swa_mcc = matthews_corrcoef(swa_predictions, current_yt)
                swa_prec = precision_score(swa_predictions, current_yt)
                swa_rec = recall_score(swa_predictions, current_yt)
                swa_accs.append(swa_acc)
                swa_precs.append(swa_prec)
                swa_mccs.append(swa_mcc)
                swa_recs.append(swa_rec)
                torch.save(swa_model.state_dict(), fname.replace(".ptch", ".swa."+str(i)+".ptch"))
            torch.save(best_model.state_dict(), fname.replace(".ptch", ".best."+str(i)+".ptch"))
            del model
            torch.cuda.empty_cache()
        return(accs, swa_accs, precs, swa_precs, recs, swa_recs, mccs, swa_mccs)
    
    
# ### Capsule regression model

# In[7]:

class CapsuleRegressor(nn.Module):
    
    def __init__(self, batch_size):
        super(CapsuleRegressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, 1)
        )
        self.lstm = nn.GRU(16, 64, num_layers=8, bidirectional=True)#, batch_first=True)
        self.batch_size = batch_size
        
    def forward(self, x):
        #y = self.conv(x).view(x.shape[0], 16*15*1)
        y = self.lstm(x)[0].view(x.shape[0], 256)
        r = self.fc(y)
        return(r)#[0].view(x.shape[0], 1))
    
    def predict(self, X):
        out = []
        for batch_X, _ in iterate_minibatches(X, X, self.batch_size, False):
            tensor_X = torch.from_numpy(batch_X).type(torch.FloatTensor).cuda()
            u = self.forward(tensor_X).cpu().data.numpy()
            out.append(u)
        return(np.concatenate(out))
    
    def validate(self, X, Y):
        preds = self.predict(X)
        return(mean_squared_error(preds, Y), preds)
    
    def fit(self, X, Y, test_X, test_Y, val_X, val_Y, loss, optimizer, epochs):
        train_loss = []
        train_mae = []
        train_medae = []
        test_loss = []
        test_mae = []
        test_medae = []
        val_loss = []
        val_mae = []
        val_medae = []
        best_test_loss = np.inf
        best = deepcopy(self)
        for a in tqdm(range(epochs)):
            self.train()
            running_loss = []
            running_mae = []
            running_medae = []
            for batch_X, batch_Y in iterate_minibatches(X, Y, self.batch_size, True):
                optimizer.zero_grad()
                tensor_X = torch.from_numpy(batch_X).type(torch.FloatTensor).cuda()
                tensor_Y = torch.from_numpy(batch_Y).type(torch.FloatTensor).cuda()
                Y_hat = self.forward(tensor_X)
                current_loss = loss(Y_hat, tensor_Y)
                current_loss.backward()
                optimizer.step()
                Y_h = Y_hat.cpu().data.numpy()
                running_loss.append(current_loss.cpu().data.numpy())
                running_mae.append(mean_absolute_error(Y_h, batch_Y))
                running_medae.append(median_absolute_error(Y_h, batch_Y))
            train_loss.append(np.mean(running_loss))
            train_mae.append(np.mean(running_mae))
            train_medae.append(np.mean(running_medae))
            self.eval()
            r_tl, t_Y_h = self.validate(test_X, test_Y)
            test_loss.append(r_tl)
            test_mae.append(mean_absolute_error(t_Y_h, test_Y))
            test_medae.append(median_absolute_error(t_Y_h, test_Y))
            r_vl, v_Y_h = self.validate(val_X, val_Y)
            val_loss.append(r_vl)
            val_mae.append(mean_absolute_error(v_Y_h, val_Y))
            val_medae.append(median_absolute_error(v_Y_h, val_Y))
            if best_test_loss > test_loss[-1]:
                best = deepcopy(self)
                best_test_loss = test_loss[-1]
        return(
            train_loss, test_loss, val_loss, 
            train_mae, test_mae, val_mae,
            train_medae, test_medae, val_medae, 
            best
        )

    
class SiameseGuideCaps(nn.Module):
    
    def __init__(self, guide_length, n_routes=120, batch_size=64):
        super(SiameseGuideCaps, self).__init__()
        self.GC = GuideCaps(guide_length, n_routes)
        self.classifier = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(64, 2, kernel_size=4, stride=1),
            nn.ReLU()
        )
        self.last_linear = nn.Sequential(
            nn.Linear(58,2),#2*26, 2),
            nn.Softmax(dim=1)
        )
        self.batch_size=64
        
    def cudify(self, x):
        return(x if not torch.cuda.is_available() else x.cuda())
        
    def partial_forward(self, x1, r2):
        #internal1 = self.GC(x1)[0].reshape(x1.shape[0], 64)
        diff = torch.abs(internal1-r2)
        out = self.classifier(diff)
        return(out)
    
    def forward(self, x1, x2):
        internal1 = self.GC(x1)[0]#.reshape(x1.shape[0], 64)
        internal2 = self.GC(x2)[0]#.reshape(x2.shape[0], 64)
        diff = torch.abs(internal1-internal2)
        out = self.classifier(diff).reshape(x1.shape[0], 58)#2*26)
        out = self.last_linear(out)
        return(out)
    
    def partial_predict(self, x1, r2):
        rng = np.arange(x1.shape[0])
        y_hat = []
        f = tqdm if verbose else lambda x: x
        for x_ind, _ in f(iterate_minibatches(rng, rng, self.batch_size, False)):
            test_x1 = self.cudify(
                Variable(
                    torch.from_numpy(
                        x1[x_ind].reshape(x1[x_ind].shape[0],4,int(x1[x_ind].shape[1]/4))
                    ).type(torch.FloatTensor)
                ).cuda()
            )
            test_r2 = self.cudify(
                Variable(
                    torch.from_numpy(
                        r2[x_ind].reshape(r2[x_ind].shape[0],4,int(r2[x_ind].shape[1]/4))
                    ).type(torch.FloatTensor)
                ).cuda()
            )
            y2 = self.forward(test_x1, test_x2).cpu().data.numpy()
            y_hat.append(y2)
        y_hat = np.concatenate(y_hat,0)
        return(y_hat)
    
    def predict(self, x1, x2, verbose=False):
        rng = np.arange(x1.shape[0])
        y_hat = []
        f = tqdm if verbose else lambda x: x
        for x_ind, _ in f(iterate_minibatches(rng, rng, self.batch_size, False)):
            test_x1 = self.cudify(
                Variable(
                    torch.from_numpy(
                        x1[x_ind].reshape(x1[x_ind].shape[0],4,int(x1[x_ind].shape[1]/4))
                    ).type(torch.FloatTensor)
                ).cuda()
            )
            test_x2 = self.cudify(
                Variable(
                    torch.from_numpy(
                        x2[x_ind].reshape(x1[x_ind].shape[0],4,int(x1[x_ind].shape[1]/4))
                    ).type(torch.FloatTensor)
                ).cuda()
            )
            y2 = self.forward(test_x1, test_x2).cpu().data.numpy()
            y_hat.append(y2)
        y_hat = np.concatenate(y_hat,0)
        return(y_hat)
    
    def fit(self, X1, X2, Y, test_X1, test_X2, test_Y, loss, optimizer, epochs):
        training_loss = []
        training_bal_acc = []
        test_loss = []
        test_bal_acc = []
        best_model = deepcopy(self)
        best_test_acc = 0
        for epoch in tqdm(range(epochs)):
            self.train()
            running_loss = []
            running_bal_acc = []
            rng = np.arange(X1.shape[0])
            for x_ind, batch_y in iterate_minibatches(rng, Y, self.batch_size):
                optimizer.zero_grad()
                test_x1 = self.cudify(
                    Variable(
                        torch.from_numpy(
                            X1[x_ind].reshape(X1[x_ind].shape[0],4,int(X1[x_ind].shape[1]/4))
                        ).type(torch.FloatTensor)
                    ).cuda()
                )
                test_x2 = self.cudify(
                    Variable(
                        torch.from_numpy(
                            X2[x_ind].reshape(X2[x_ind].shape[0],4,int(X2[x_ind].shape[1]/4))
                        ).type(torch.FloatTensor)
                    ).cuda()
                )
                batch_yt = self.cudify(
                    Variable(
                        torch.from_numpy(
                            batch_y
                        ).type(torch.LongTensor)
                    )
                )
                batch_yhat = self.forward(test_x1, test_x2)
                batch_loss = loss(batch_yhat, batch_yt)
                batch_loss.backward()
                optimizer.step()
                running_loss.append(batch_loss.cpu().data.numpy())
                running_bal_acc.append(
                    balanced_accuracy_score(
                        batch_y, np.argmax(batch_yhat.cpu().data.numpy(), 1)
                    )
                )
            training_loss.append(np.mean(running_loss))
            training_bal_acc.append(np.mean(running_bal_acc))
            self.eval()
            test_yhat = self.cudify(
                Variable(
                    torch.from_numpy(
                        self.predict(test_X1, test_X2)
                    ).type(torch.FloatTensor)
                ).cuda()
            )
            test_yt = self.cudify(
                Variable(
                    torch.from_numpy(
                        test_Y
                    ).type(torch.LongTensor)
                ).cuda()
            )
            test_loss.append(
                loss(test_yhat, test_yt).cpu().data.numpy()
            )
            test_bal_acc.append(
                balanced_accuracy_score(test_Y, np.argmax(test_yhat.cpu().data.numpy(), 1))
            )
            if test_bal_acc[-1] > best_test_acc:
                best_model = deepcopy(self)
                best_test_acc = test_bal_acc[-1]
        return(best_model, training_loss, training_bal_acc, test_loss, test_bal_acc)