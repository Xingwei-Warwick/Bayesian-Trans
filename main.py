from dataset_util import *
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
#from baseline import LM_base
from bayesian_models import Bayesian_MuRE
import os


def load_comet(path="./comet-atomic_2020_BART"):
    encoder_model = AutoModelForSeq2SeqLM.from_pretrained(path).get_encoder()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return encoder_model, tokenizer


def confusion2prf(confusion):
    tp = 1.0 * np.sum([confusion[i][i] for i in range(3)])
    if tp == 0.:
        return 0., 0., 0.

    prec = tp / (np.sum(confusion[:4,:3]))
    rec = tp / (np.sum(confusion[:3,:4]))
    f1 = 2.0 / (1.0 / prec + 1.0 / rec)
    return prec,rec,f1


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed+1)
    torch.cuda.manual_seed_all(seed+2)
    np.random.seed(seed+3)
    os.environ['PYTHONHASHSEED'] = str(seed+4)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = ArgumentParser(description='Fine-tune COMET-bart on MATRES')
    parser.add_argument('--cuda', help='Use GPU', type=int, default=1)
    parser.add_argument('--dataset', help='The name of dataset', type=str, default='matres')
    parser.add_argument('--lm-lr', help='learning rate of the language model', type=float, default=1e-5)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', help='weight decay', type=float, default=1e-2)
    parser.add_argument('--step_size', help='step size', type=int, default=10)
    parser.add_argument('--max_epoch', help='max training epoch', type=int, default=20)
    parser.add_argument('--expname', help='save file name', type=str, default='test')
    parser.add_argument('--sd', help='random seed', type=int, default=722)
    parser.add_argument('--latent-dim', help='Latent dimension', type=int, default=600)
    parser.add_argument('--mure-dim', help='mure hidden dimension', type=int, default=300)
    parser.add_argument('--output-result', help='save test result to file', type=str, default='')
    parser.add_argument('--prior', help='file path that contains the prior embeddings', type=str, default='')
    parser.add_argument('--model', help='the name of the model', type=str, default='transe')
    parser.add_argument('--dropout', help='dropout rate', type=float, default=0.5)
    parser.add_argument('--hidden', help='hidden_dim', type=int, default=300)
    parser.add_argument('--load-path', help='load model from', type=str, default='')
    parser.add_argument('--kl-scaling', help='kl weight scaling', type=float, default=5e-2)

    parser.add_argument('--batch', help='batch size', type=int, default=16)
    parser.add_argument('--gamma', help='gamma', type=float, default=0.2)
    parser.add_argument('--beta', help='beta', type=float, default=1.)
    parser.add_argument('--regularization_type', help='regularization type', type=str, default='kl')

    # flags
    parser.add_argument('--skiptraining', help='skip training', action='store_true')
    parser.add_argument('--cleanmode', help='dont print', action='store_true')
    parser.add_argument('--save-prediction', help='save test prediction', action='store_true')
    parser.add_argument('--debug', help='debug mode', action='store_true')
    parser.add_argument('--write_report', help='write report to file', action='store_true')


    args = parser.parse_args()
    print(args)

    seed_everything(args.sd)

    kl_weight_scaling = np.arange(1e-2, 2.0, args.kl_scaling)

    if torch.cuda.is_available() and args.cuda > 0:
        device = f'cuda:%d' % (args.cuda-1)
    else:
        device = 'cpu'
    if not args.cleanmode:
        print('Using Device: ' + device)
    
    # load PLM model
    encoder_model, tokenizer = load_comet()

    # dataset loading
    if args.dataset == 'matres':
        num_rel = 4
        train_set, val_set, test_set = get_matres_datasets(train_xml_fname="data/trainset-temprel.xml", 
                                            test_xml_fname="data/testset-temprel.xml", tokenizer=tokenizer, sd=args.sd)
    elif args.dataset == 'tcr':
        num_rel = 4
        train_set, val_set, test_set = get_matres_datasets(train_xml_fname="data/trainset-temprel.xml", 
                                            test_xml_fname="data/tcr-temprel.xml", tokenizer=tokenizer, sd=args.sd)
    elif args.dataset == 'tbd':
        num_rel = 6
        train_set, val_set, test_set = get_tbd(tokenizer)
    else:
        print('Not recognised Dataset!')
        exit(0)
    
    if args.model == 'mure':
        model = Bayesian_MuRE(device=device, prior_path=args.prior, latent_dim=args.latent_dim,
                                mure_dim=args.mure_dim, num_rel=num_rel, beta=args.beta, dropout=args.dropout, reg_type=args.regularization_type)
    elif args.model == 'lm_base':
        model = LM_base(device=device, num_class=num_rel, dim_hidden=args.mure_dim, dim_out=args.latent_dim,
                         dropout=args.dropout)
    
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False)

    # training loop
    if not args.skiptraining:
        train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False)
        
        model.to(device)
        encoder_model.to(device)

        optim_lm = Adam(encoder_model.parameters(), lr=args.lm_lr)
        optim_mure = Adam(model.parameters(), lr=args.lr)
        scheduler1 = lr_scheduler.StepLR(optim_mure, step_size=args.step_size, gamma=args.gamma)

        best_f1 = -0.01
        train_loss_record = []
        kl_loss_record = []
        best_epo = 0

        for i in range(args.max_epoch):
            if not args.cleanmode:
                print('Epoch %d training begins ...' % (i))
            model.train()
            encoder_model.train()
            acc_loss = 0.
            kl_loss = 0.
            data_len = 0

            for num, batch in tqdm(enumerate(train_loader)):
                optim_lm.zero_grad()
                optim_mure.zero_grad()

                if args.model == 'mure':
                    hidden_states = encoder_model(batch['input_ids'].to(device),
                                        attention_mask=batch['attention_mask'].to(device)).last_hidden_state
                    loss_dict = model.loss_function(*model(hidden_states, epos_1=batch['event1pos'].to(device),
                                epos_2=batch['event2pos'].to(device), rel=batch['temp_rel'].to(device)), 
                                M_N=kl_weight_scaling[i]*batch['input_ids'].size(0)*1.0/len(train_loader))

                    train_loss_record.append(loss_dict['loss'].item()*batch['input_ids'].size(0))
                    kl_loss_record.append(loss_dict['KLD'].item()*batch['input_ids'].size(0))
                    data_len += batch['input_ids'].size(0)

                    loss_dict['loss'].backward()
                elif args.model == 'lm_base':
                    hidden_states = encoder_model(batch['input_ids'].to(device),
                                        attention_mask=batch['attention_mask'].to(device)).last_hidden_state
                    logits = model(hidden_states, epos_1=batch['event1pos'].to(device),
                            epos_2=batch['event2pos'].to(device))
                    loss = model.loss(logits, batch['temp_rel'].to(device))
                    train_loss_record.append(loss.data)
                    kl_loss_record.append(0)
                    
                    loss.backward()

                optim_lm.step()
                optim_mure.step()

            scheduler1.step()

            if not args.cleanmode:
                print('Epoch %d finished' % (i))
                print('Average Loss: %.2f, Average (normalized) KLD loss: %.2f' % (sum(train_loss_record)/data_len, sum(kl_loss_record)/data_len))

            al_predict = []
            al_gold = []

            model.eval()
            encoder_model.eval()
            for num, batch in tqdm(enumerate(val_loader)):
                with torch.no_grad():
                    if args.model == 'mure':
                        hidden_states = encoder_model(batch['input_ids'].to(device),
                                        attention_mask=batch['attention_mask'].to(device)).last_hidden_state
                        logits = model(hidden_states, epos_1=batch['event1pos'].to(device),
                            epos_2=batch['event2pos'].to(device), rel=batch['temp_rel'].to(device))[0]
                    elif args.model == 'lm_base':
                        hidden_states = encoder_model(batch['input_ids'].to(device),
                                        attention_mask=batch['attention_mask'].to(device)).last_hidden_state
                        logits = model(hidden_states, epos_1=batch['event1pos'].to(device),
                            epos_2=batch['event2pos'].to(device))

                    predicts = torch.argmax(logits, dim=-1)
                    al_predict.append(predicts.detach().cpu().numpy())
                    
                    for temp_rel in batch['temp_rel']:
                        al_gold.append(temp_rel)
                    
            al_predict = np.concatenate(al_predict, axis=0)
            acc = accuracy_score(al_gold, al_predict)

            if args.dataset == 'matres' or args.dataset == 'tcr':
                confu = confusion_matrix(al_gold, al_predict)
                prec, rec, f1 = confusion2prf(confu)
                if not args.cleanmode:
                    print(confu, flush=True)
                    print("Prec=%.4f, Rec=%.4f, F1=%.4f, Acc=%.4f" %(prec, rec, f1, acc))
            elif args.dataset == 'tbd':
                confu = confusion_matrix(al_gold, al_predict)
                f1 = f1_score(al_gold, al_predict, average='micro')
                if not args.cleanmode:
                    print(confu, flush=True)
                    print("F1=%.4f, Acc=%.4f" %(f1, acc))
            else:
                print('Not recognized dataset!')

            if f1 > best_f1:
                best_f1 = f1
                best_epo = i
                torch.save(model.state_dict(), 'model_cache/%s_model_state_dict.pt'%(args.expname))
                torch.save(encoder_model.state_dict(), 'model_cache/%s_encoder_model_state_dict.pt'%(args.expname))
        
        if best_epo != args.max_epoch-1:
            print('Testing using epoch %d (F1=%.4f)...' % (best_epo, best_f1))
            model.load_state_dict(torch.load('model_cache/%s_model_state_dict.pt'%(args.expname)))
            encoder_model.load_state_dict(torch.load('model_cache/%s_encoder_model_state_dict.pt'%(args.expname)))
            encoder_model.to(device)
            model.to(device)
    else:
        if len(args.load_path) > 0:
            load_path = args.load_path
        else:
            load_path = 'model_cache'
        model.load_state_dict(torch.load(load_path+'/%s_model_state_dict.pt'%(args.expname), map_location=device))
        model.to(device)
        encoder_model.load_state_dict(torch.load(load_path+'/%s_encoder_model_state_dict.pt'%(args.expname), map_location=device))
        encoder_model.to(device)

    al_predict = []
    al_gold = []

    model.eval()
    encoder_model.eval()

    # Evaluate on test set
    al_prob_list = []
    for num, batch in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            prob_list = []
            if args.model == 'mure':
                hidden_states = encoder_model(batch['input_ids'].to(device),
                                    attention_mask=batch['attention_mask'].to(device)).last_hidden_state
                logits = model(hidden_states, epos_1=batch['event1pos'].to(device),
                        epos_2=batch['event2pos'].to(device), rel=batch['temp_rel'].to(device))[0]
            elif args.model == 'lm_base':
                hidden_states = encoder_model(batch['input_ids'].to(device),
                                    attention_mask=batch['attention_mask'].to(device)).last_hidden_state
                logits = model(hidden_states, epos_1=batch['event1pos'].to(device),
                        epos_2=batch['event2pos'].to(device)) 
            prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            predicts = np.argmax(prob, axis=-1)

            al_prob_list.append(prob)
            al_predict.append(predicts)
            for temp_rel in batch['temp_rel']:
                al_gold.append(temp_rel)
    al_predict = np.concatenate(al_predict, axis=0)
    acc = accuracy_score(al_gold, al_predict)

    ## save prediction probabilities
    # al_probs = np.concatenate(al_prob_list, axis=0)
    # with open('model_pred_probs/new_prediction_probs'+str(args.sd)+'.npy', 'wb') as f:
    #     np.save(f, al_probs)
    # print('Prediction probabilities saved!')

    if args.dataset == 'matres' or args.dataset == 'tcr':
        confu = confusion_matrix(al_gold, al_predict)
        cl_report = classification_report(al_gold, al_predict, digits=4)
        print(cl_report)
        prec, rec, f1 = confusion2prf(confu)
        report = "Prec=%.4f, Rec=%.4f, F1=%.4f, Acc=%.4f" %(prec, rec, f1, acc)
        print(confu, flush=True)
        print(report)
        # per class results
    elif args.dataset == 'tbd':
        confu = confusion_matrix(al_gold, al_predict)
        f1 = f1_score(al_gold, al_predict, average='micro')
        recall = recall_score(al_gold, al_predict, average='micro')
        precision = precision_score(al_gold, al_predict, average='micro')
        cl_report = classification_report(al_gold, al_predict, digits=4)
        report = "F1=%.4f, Acc=%.4f, Prec=%.4f, Recall=%.4f" %(f1, acc, precision, recall)
        report += '\n' + cl_report
        print(confu, flush=True)
        print(report)
    else:
        print('Not recognized dataset!')
    