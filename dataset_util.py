from dataclasses import dataclass
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import spacy
from os import listdir
from os.path import isfile, join
from nltk.tokenize import sent_tokenize
import re
import numpy as np
import random
import math
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.spatial
#from sentence_transformers import SentenceTransformer


nlp = spacy.load("en_core_web_sm")


MATRES_labels = {"BEFORE":0,"AFTER":1,"EQUAL":2,"VAGUE":3}
TBD_labels = {'SIMULTANEOUS': 2, 'AFTER': 1, 'INCLUDES': 3, 'IS_INCLUDED': 4, 'BEFORE': 0, 'NONE': 5}

@dataclass(frozen=True)
class Atomic_ins:
    head: str
    rel: str      
    tail: str


################## Atomic Dataset #################################
class ATOMIC_dataset():
    def __init__(self, device='cuda:0',  path='data/atomic2020/', rel_type='temporal', load_cache=False):
        self.device = device
        self.train = []
        self.dev = []
        self.test = []

        self.event_dict = {}
        self.event_list = []

        self.rel_dict = {}
        self.rel_list = []

        if rel_type == 'temporal':
            self.needed_rel = ['isAfter', 'isBefore']
        elif rel_type == 'hieve':
            self.needed_rel = ['HasSubEvent']
        elif rel_type == 'causal':
            self.needed_rel = ['Causes', 'xReason']
        else:
            print('Specific Relation Type is not recoginizable!')
            return None

        # duplicate entries
        entry_set = set()

        with open(path+'train.tsv', 'r') as f:
            for line in f.readlines():
                this_entries = line.strip().split('\t')
                if len(this_entries) < 3:
                    continue
                elif len(this_entries) > 3:
                    tail = ' '.join(this_entries[2:])
                else:
                    tail = this_entries[2]
                
                if this_entries[1] not in self.needed_rel:
                    continue
                
                if tail != 'none':
                    #self.train.append(Atomic_ins(this_entries[0], this_entries[1], tail))
                    if self.event_dict.get(this_entries[0]) is None:
                        self.event_dict[this_entries[0]]=len(self.event_list)
                        self.event_list.append(this_entries[0])               
                    if self.event_dict.get(tail) is None:
                        self.event_dict[tail]=len(self.event_list)
                        self.event_list.append(tail)
                    if self.rel_dict.get(this_entries[1]) is None:
                        self.rel_dict[this_entries[1]] = len(self.rel_list)
                        self.rel_list.append(this_entries[1])  
                    
                    
                    event_pair_key = str(self.event_dict[this_entries[0]])+'_'+str(self.event_dict[tail])
                    
                    if event_pair_key+'_R_'+str(self.rel_dict[this_entries[1]]) in entry_set:
                        continue
                    else:
                        self.train.append([self.event_dict[this_entries[0]],
                                        self.rel_dict[this_entries[1]],
                                        self.event_dict[tail]])  
                        entry_set.add(event_pair_key+'_R_'+str(self.rel_dict[this_entries[1]]))
                    '''
                    if test_multi_label.get(event_pair_key) is not None:
                        #test_multi_label[event_pair_key] += ','+this_entries[1]
                        print(test_multi_label[event_pair_key], this_entries)
                    else:
                        test_multi_label[event_pair_key] = this_entries'''
        #self.train = np.array(self.train, dtype=np.int64)
        #self.train_nodes = len(self.event_list)
        
        with open(path+'dev.tsv', 'r') as f:
            for line in f.readlines():
                this_entries = line.strip().split('\t')

                if this_entries[1] not in self.needed_rel:
                    continue
                
                if this_entries[2] != 'none':
                    #self.dev.append(Atomic_ins(this_entries[0], this_entries[1], this_entries[2]))
                    if self.event_dict.get(this_entries[0]) is None:          
                        self.event_dict[this_entries[0]]=len(self.event_list)
                        self.event_list.append(this_entries[0])              
                    if self.event_dict.get(this_entries[2]) is None:
                        self.event_dict[this_entries[2]]=len(self.event_list)
                        self.event_list.append(this_entries[2]) 
                    if self.rel_dict.get(this_entries[1]) is None:
                        self.rel_dict[this_entries[1]] = len(self.rel_list)
                        self.rel_list.append(this_entries[1])

                    event_pair_key = str(self.event_dict[this_entries[0]])+'_'+str(self.event_dict[this_entries[2]])
                    
                    if event_pair_key+'_R_'+str(self.rel_dict[this_entries[1]]) in entry_set:
                        continue
                    else:
                        self.train.append([self.event_dict[this_entries[0]],
                                    self.rel_dict[this_entries[1]],
                                    self.event_dict[this_entries[2]]])   
                        entry_set.add(event_pair_key+'_R_'+str(self.rel_dict[this_entries[1]]))
                         
        #self.dev = np.array(self.dev, dtype=np.int64)
        with open(path+'test.tsv', 'r') as f:
            for line in f.readlines():
                this_entries = line.strip().split('\t')

                if this_entries[1] not in self.needed_rel:
                    continue
                
                if this_entries[2] != 'none':
                    #self.test.append(Atomic_ins(this_entries[0], this_entries[1], this_entries[2]))
                    if self.event_dict.get(this_entries[0]) is None:          
                        self.event_dict[this_entries[0]]=len(self.event_list)
                        self.event_list.append(this_entries[0])              
                    if self.event_dict.get(this_entries[2]) is None:
                        self.event_dict[this_entries[2]]=len(self.event_list)
                        self.event_list.append(this_entries[2]) 
                    if self.rel_dict.get(this_entries[1]) is None:
                        self.rel_dict[this_entries[1]] = len(self.rel_list)
                        self.rel_list.append(this_entries[1])

                    event_pair_key = str(self.event_dict[this_entries[0]])+'_'+str(self.event_dict[this_entries[2]])
                    
                    if event_pair_key+'_R_'+str(self.rel_dict[this_entries[1]]) in entry_set:
                        continue
                    else:
                        self.train.append([self.event_dict[this_entries[0]],
                                    self.rel_dict[this_entries[1]],
                                    self.event_dict[this_entries[2]]])   
                        entry_set.add(event_pair_key+'_R_'+str(self.rel_dict[this_entries[1]]))
     
        print('Loaded train instances: %d, dev instances: %d, test instances: %d' % (len(self.train), len(self.dev), len(self.test)))

        self.build_semantic_links(load_cache=load_cache)

        random.shuffle(self.train)
        num_tri = len(self.train)

        test_size = math.floor(num_tri * 0.1)

        self.test = np.array([k for k in self.train[num_tri-test_size:] if k[1]!=self.rel_dict['similar']], dtype=np.int64) # only temporal relation
        #self.dev = np.array(self.train[num_tri-2*test_size:num_tri-test_size], dtype=np.int64)
        self.dev = np.array([k for k in self.train[num_tri-2*test_size:num_tri-test_size] if k[1]!=self.rel_dict['similar']], dtype=np.int64)
        self.train = np.array(self.train[:num_tri-2*test_size], dtype=np.int64)
        
        print('Loaded train instances: %d, dev instances: %d, test instances: %d' % (len(self.train), len(self.dev), len(self.test)))
        print(self.rel_dict)

    def __len__(self):
        return len(self.event_list)
    
    def get_num_rels(self):
        return len(self.rel_list)
    
    # for building graph
    def semantic_encoding(self, t=0.9):
        encoder_model = SentenceTransformer('bert-base-nli-mean-tokens', device=self.device)
        self.encoded = encoder_model.encode(self.event_list)
        self.encoded = torch.tensor(np.array(self.encoded))
        encoder_model.cpu()
        del encoder_model
    
    def encode_event(self, path="./comet-atomic_2020_BART", save=True):
        encoder_model = AutoModelForSeq2SeqLM.from_pretrained(path).get_encoder()
        tokenizer = AutoTokenizer.from_pretrained(path)
        
        encoder_model.eval()
        encoder_model.to(self.device)

        #event_idx = []
        prefix_list = []
        for event in self.event_list:
            processed = nlp(event)
            prefix = []
            for tok in processed:
                if tok.dep_ == 'ROOT':
                    break
                prefix.append(tok.text)
            prefix_list.append(' '.join(prefix))
            #event_idx.append(r_id)
        encoding = tokenizer(prefix_list, padding=False)
        event_posi = []
        for ins in encoding['input_ids']:
            event_posi.append(len(ins) - 1)

        encoding = tokenizer(self.event_list, truncation=True, padding=True)
        
        data = Temp_torch_dataset(encoding)
        dataload = DataLoader(data, batch_size=256, shuffle=False)

        self.event_rep = []
        count = 0
        for num, batch in tqdm(enumerate(dataload)):
            with torch.no_grad():
                hid_states = encoder_model(batch['input_ids'].to(self.device),
                            attention_mask=batch['attention_mask'].to(self.device)).last_hidden_state

                hid_states = hid_states.detach().cpu().numpy()
            '''
            for i in range(hid_states.shape[0]):
                self.event_rep.append(hid_states[i,event_posi[count],:])
                count += 1
            '''
            self.event_rep.append(hid_states[:,0,:])
        
        self.event_rep = np.concatenate(self.event_rep, 0)
        print(self.event_rep.shape)
        with open("event_rep_cache_cls.npy", 'wb') as f:
            np.save(f, self.event_rep)
        
    def build_semantic_links(self, t=0.95, load_cache=True):
        #knng = kneighbors_graph(self.encoded)
        
        if load_cache:
            sim_matrix = np.load('sim_matrix_temporal.npy')
            print(sim_matrix.shape)
            print('loaded similarity matrix')
        else:
            # self.semantic_encoding()
            # self.encode_event()
            self.encoded = np.load("event_rep_cache_cls.npy")
            print('Computing sentence representations...')
            print('Computing the similarity matrix...')
            sim_matrix = scipy.spatial.distance.cdist(self.encoded, self.encoded, "cosine")
            with open('sim_matrix_temporal.npy', 'wb') as f:
                np.save(f, sim_matrix)
 
        #semantic_sim_links = []
        #for head, row in enumerate(sim_matrix):
        #    for tail, dist in enumerate(row):
        #        if 1-dist >= t:
        #            semantic_sim_links.append([head, len(self.rel_list), tail]) # "semantic related" relation'''
        head, tail = np.where(sim_matrix<=(1-t))
        rel = np.full_like(head.reshape(-1,1), len(self.rel_list))
        semantic_sim_links = np.concatenate([head.reshape(-1,1), rel, tail.reshape(-1,1)], 1)
        semantic_sim_links_reverse = np.concatenate([tail.reshape(-1,1), rel, head.reshape(-1,1)], 1)
        ### this relation is symmetric

        print('Build semantic similarity edges: %d' % (semantic_sim_links.shape[0]*2))
        
        self.train = np.concatenate([self.train, semantic_sim_links, semantic_sim_links_reverse])
        self.rel_dict['similar'] = len(self.rel_list)
        self.rel_list.append('similar')


@dataclass(frozen=True)
class Event_Rel:
    docid: str
    label: str      # label is a word, BEFORE, AFTER...
    source: str
    target: str
    token: list
    event_ix: list
    verbs: list
    lemma: list
    part_of_speech: list
    position: list
    length: int


def load_xml(xml_element):
    xml_element = xml_element
    label = xml_element.attrib['LABEL']
    sentdiff = int(xml_element.attrib['SENTDIFF'])
    docid = xml_element.attrib['DOCID']
    source = xml_element.attrib['SOURCE']
    target = xml_element.attrib['TARGET']
    data = xml_element.text.strip().split()
    token = []
    lemma = []
    part_of_speech = []
    position = []
    length = len(data)
    event_ix = []
    verbs = []

    for i,d in enumerate(data):
        tmp = d.split('///')
        part_of_speech.append(tmp[-2])
        position.append(tmp[-1])
        if tmp[-1] == 'E1':
            event_ix.append(i)
            verbs.append(tmp[0])
        elif tmp[-1] == 'E2':
            event_ix.append(i)
            verbs.append(tmp[0])
        token.append(tmp[0])
        lemma.append(tmp[1])

    return Event_Rel(docid=docid, label=label, source=source,
                    target=target, token=token, lemma=lemma,
                    part_of_speech=part_of_speech, position=position,
                    length=length, event_ix=event_ix, verbs=verbs)


class Temp_torch_dataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        it = {key: torch.tensor(val[idx]) for key, val in self.encodings.items() if key !='offset_mapping'}
        return it


######################### Matres Dataset ################################
def get_matres_datasets(tokenizer, train_xml_fname='', test_xml_fname='', get_test_text=False, sd=42):
    # load train file
    train_set, val_set, test_set = None, None, None

    if train_xml_fname != '':
        tree = ET.parse(train_xml_fname)
        root = tree.getroot()
        trainval_ins_list = []

        for e in root:
            ins = load_xml(e)
            trainval_ins_list.append(ins)

        # train val split
        train_ins_list, val_ins_list = train_test_split(trainval_ins_list, test_size=0.2, random_state=sd)

        train_encoding = tokenizer([' '.join(ins.token) for ins in train_ins_list],
                        truncation=True, padding=True, return_offsets_mapping=True)
        train_label = []
        train_event_pos1 = []
        train_event_pos2 = []
        train_context = []
        for idx, ins in enumerate(train_ins_list):
            train_context.append(' '.join(ins.token))
            train_label.append(ins.label)

            # trigger position
            e1_char_start = len(' '.join(ins.token[:ins.event_ix[0]]))
            if e1_char_start != 0:
                e1_char_start += 1
            e2_char_start = len(' '.join(ins.token[:ins.event_ix[1]]))
            if e2_char_start != 0:
                e2_char_start += 1

            lm_tk_e1_pos, lm_tk_e2_pos = None, None
            for tk_id, offset_pair in enumerate(train_encoding['offset_mapping'][idx]):
                if offset_pair[0] == e1_char_start and offset_pair[0] != offset_pair[1]:
                    lm_tk_e1_pos = tk_id
                
                if offset_pair[0] == e2_char_start and offset_pair[0] != offset_pair[1]:
                    lm_tk_e2_pos = tk_id
            
            if lm_tk_e1_pos is None or lm_tk_e2_pos is None:
                print('Event position not found!')

            train_event_pos1.append(lm_tk_e1_pos)          # first token of the verb
            train_event_pos2.append(lm_tk_e2_pos)

        if len(train_encoding['input_ids']) < len(train_event_pos1):
            train_encoding = tokenizer(train_context,
                        truncation=True, padding=True)
                
        train_encoding['event1pos'] = train_event_pos1
        train_encoding['event2pos'] = train_event_pos2
        train_encoding['temp_rel'] = list(map(lambda x: MATRES_labels[x], train_label))

        val_encoding = tokenizer([' '.join(ins.token) for ins in val_ins_list],
                        truncation=True, padding=True, return_offsets_mapping=True)
        val_label = []
        val_event_pos1 = []
        val_event_pos2 = []
        for idx, ins in enumerate(val_ins_list):
            val_label.append(ins.label)

            # trigger position
            e1_char_start = len(' '.join(ins.token[:ins.event_ix[0]]))
            if e1_char_start != 0:
                e1_char_start += 1
            e2_char_start = len(' '.join(ins.token[:ins.event_ix[1]]))
            if e2_char_start != 0:
                e2_char_start += 1

            lm_tk_e1_pos, lm_tk_e2_pos = None, None
            for tk_id, offset_pair in enumerate(val_encoding['offset_mapping'][idx]):
                if offset_pair[0] == e1_char_start and offset_pair[0] != offset_pair[1]:
                    lm_tk_e1_pos = tk_id
                
                if offset_pair[0] == e2_char_start and offset_pair[0] != offset_pair[1]:
                    lm_tk_e2_pos = tk_id
            
            if lm_tk_e1_pos is None or lm_tk_e2_pos is None:
                print('Event position not found!')

            val_event_pos1.append(lm_tk_e1_pos)          # first token of the verb
            val_event_pos2.append(lm_tk_e2_pos)

        val_encoding['event1pos'] = val_event_pos1
        val_encoding['event2pos'] = val_event_pos2
        val_encoding['temp_rel'] = list(map(lambda x: MATRES_labels[x], val_label))

        train_set = Temp_torch_dataset(train_encoding)
        val_set = Temp_torch_dataset(val_encoding)

    # load test file
    if test_xml_fname != '':
        tree = ET.parse(test_xml_fname)
        root = tree.getroot()

        test_context = []
        test_label = []
        test_event_pos1 = []
        test_event_pos2 = []

        for e in root:
            ins = load_xml(e)

            this_context = ' '.join(ins.token)
            test_context.append(this_context)
            tokenized_output = tokenizer(this_context,
                        truncation=False, padding=False, return_offsets_mapping=True)
            test_label.append(ins.label)

            # trigger position
            e1_char_start = len(' '.join(ins.token[:ins.event_ix[0]]))
            if e1_char_start != 0:
                e1_char_start += 1
            e2_char_start = len(' '.join(ins.token[:ins.event_ix[1]]))
            if e2_char_start != 0:
                e2_char_start += 1

            lm_tk_e1_pos, lm_tk_e2_pos = None, None
            for tk_id, offset_pair in enumerate(tokenized_output['offset_mapping']):
                if offset_pair[0] == e1_char_start and offset_pair[0] != offset_pair[1]:
                    lm_tk_e1_pos = tk_id
                
                if offset_pair[0] == e2_char_start and offset_pair[0] != offset_pair[1]:
                    lm_tk_e2_pos = tk_id
            
            if lm_tk_e1_pos is None or lm_tk_e2_pos is None:
                print('Event position not found!')

            test_event_pos1.append(lm_tk_e1_pos)          # first token of the verb
            test_event_pos2.append(lm_tk_e2_pos)

        test_encoding = tokenizer(test_context,
                        truncation=True, padding=True)
        test_encoding['event1pos'] = test_event_pos1
        test_encoding['event2pos'] = test_event_pos2
        test_encoding['temp_rel'] = list(map(lambda x: MATRES_labels[x], test_label))
        test_set = Temp_torch_dataset(test_encoding)

    if test_set is None:
        return train_set, val_set
    elif train_set is None:
        return test_set
    else:
        if get_test_text:
            return train_set, val_set, test_set, test_context
        else:
            return train_set, val_set, test_set
    

# TBD dataset
def get_tbd(tokenizer, path='data/TimeBank-dense/'):
    encoding_list = []
    test_context = None
    for set_name in ['train', 'dev', 'test']:
        label_dict = {}
        dir_name = path + set_name
        onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
        context = []
        event1pos = []
        event2pos = []
        label = []
        for f in onlyfiles:
            doc, link_list, eiid_2_position, eiid_2_sent_id = read_tml(dir_name, f)
            if doc[-1] == ' ':
                doc = doc[:-1]
            sentence_list = sent_tokenize(doc)
            for l in link_list:
                ep1 = None
                ep2 = None
                if len(tokenizer.encode(doc)) > 400:
                    sent_id1 = eiid_2_sent_id[l[0]]
                    sent_id2 = eiid_2_sent_id[l[2]]
                    if max(sent_id1, sent_id2) - min(sent_id1, sent_id2) > 12: #> 12:
                        sent_span = sentence_list[min(sent_id1, sent_id2)] + ' ' + sentence_list[max(sent_id1, sent_id2)]
                        tokenized_output = tokenizer(sent_span, padding=False, truncation=False, return_offsets_mapping=True)
                        s1len = len(sentence_list[min(sent_id1, sent_id2)])
                        if sent_id1 == min(sent_id1, sent_id2): # e1 is the first sentence
                            to_s2_start = eiid_2_position[l[2]][0] - len(' '.join(sentence_list[:sent_id2])) - 1 # the char idex from the start of the sentence 2
                            to_s2_start += 1+s1len
                            e2_char_idx = to_s2_start
                            to_s1_start = eiid_2_position[l[0]][0] - len(' '.join(sentence_list[:sent_id1]))
                            if sent_id1 != 0:
                                to_s1_start -= 1
                            e1_char_idx = to_s1_start
                        else:
                            to_s2_start = eiid_2_position[l[0]][0] - len(' '.join(sentence_list[:sent_id1])) - 1 # the char idex from the start of the sentence 2
                            to_s2_start += 1+s1len
                            e1_char_idx = to_s2_start
                            to_s1_start = eiid_2_position[l[2]][0] - len(' '.join(sentence_list[:sent_id2]))
                            if sent_id2 != 0:
                                to_s1_start -= 1
                            e2_char_idx = to_s1_start

                        for tk_id, offset_pair in enumerate(tokenized_output['offset_mapping']):
                            if offset_pair[0] == e1_char_idx and offset_pair[0] != offset_pair[1]:
                                ep1 = tk_id

                            if offset_pair[0] == e2_char_idx and offset_pair[0] != offset_pair[1]:
                                ep2 = tk_id

                    else:
                        sent_span = sentence_list[min(sent_id1, sent_id2):max(sent_id1, sent_id2)+1]
                        sent_span = ' '.join(sent_span)

                        tokenized_output = tokenizer(sent_span, padding=False, truncation=False, return_offsets_mapping=True)
                        # can tokenized together
                        ppfix = len(' '.join(sentence_list[:min(sent_id1, sent_id2)]))
                        if ppfix != 0:
                            ppfix += 1
                        
                        for tk_id, offset_pair in enumerate(tokenized_output['offset_mapping']):
                            if offset_pair[0]+ppfix == eiid_2_position[l[0]][0] and offset_pair[0] != offset_pair[1]:
                                ep1 = tk_id

                            if offset_pair[0]+ppfix == eiid_2_position[l[2]][0] and offset_pair[0] != offset_pair[1]:
                                ep2 = tk_id

                    context.append(sent_span)
                else:
                    tokenized_output = tokenizer(doc, padding=False, truncation=False, return_offsets_mapping=True)
                    
                    for tk_id, offset_pair in enumerate(tokenized_output['offset_mapping']):
                        if offset_pair[0] == eiid_2_position[l[0]][0] and offset_pair[0] != offset_pair[1]:
                            ep1 = tk_id

                        if offset_pair[0] == eiid_2_position[l[2]][0] and offset_pair[0] != offset_pair[1]:
                            ep2 = tk_id
                    context.append(doc)

                event1pos.append(ep1)
                event2pos.append(ep2)
                label.append(l[1])
                
                if label_dict.get(l[1]) is None:
                    label_dict[l[1]] = 0
                label_dict[l[1]] += 1
        #print(label_dict)
        test_context = context
        encoding = tokenizer(context, truncation=True, padding=True)
        length = len(encoding['input_ids'][0])
        encoding['event1pos'] = event1pos 
        encoding['event2pos'] = event2pos 
        encoding['temp_rel'] = list(map(lambda x: TBD_labels[x], label))
        encoding_list.append(encoding)


    return Temp_torch_dataset(encoding_list[0]), Temp_torch_dataset(encoding_list[1]), Temp_torch_dataset(encoding_list[2]), test_context
            

def read_tml(dir_name, file_name):
    #lemmatizer = WordNetLemmatizer()

    tree = ET.parse(dir_name+'/'+file_name)
    root = tree.getroot()

    eiids = root.findall('./MAKEINSTANCE')
    eid_2_eiid = {}
    for e in eiids:
        if eid_2_eiid.get(e.attrib['eventID']) is None:
            eid_2_eiid[e.attrib['eventID']] = e.attrib['eiid']

    text = root.find('./TEXT')
    doc = re.sub(r'\n', '', text.text)

    eiid_2_position = {}
    eiid_2_sent_id = {}
    #eiid_2_trigger = {}
    for child in text:
        if child.tag == 'EVENT': # or TIMEX tag
            doc = re.sub(r'\s+', ' ', doc)
            word_index_start = len(doc)
            sent_id = len(sent_tokenize(doc+'.')) - 1

        doc += child.text
        doc = re.sub(r'\s+', ' ', doc)

        if child.tag == 'EVENT':
            doc = re.sub(r'\s+', ' ', doc)
            word_index_end = len(doc)
            if eid_2_eiid.get(child.attrib['eid']) is not None:
                eiid_2_position[eid_2_eiid[child.attrib['eid']]] = (word_index_start, word_index_end)
                eiid_2_sent_id[eid_2_eiid[child.attrib['eid']]] = sent_id
                #eiid_2_trigger[eid_2_eiid[child.attrib['eid']]] = child.text.lower()
        doc += child.tail
        doc = re.sub(r'\s+', ' ', doc)
    
    #tk_list = word_tokenize(doc)

    tlinks = root.findall('./TLINK')
    link_list = []
    for tlink in tlinks:
        try:
            head_eiid = tlink.attrib['eventInstanceID']
            tail_eiid = tlink.attrib['relatedToEventInstance']
        except:
            continue
        
        rel = tlink.attrib['relType']

        link_list.append((head_eiid, rel, tail_eiid))

    return doc, link_list, eiid_2_position, eiid_2_sent_id#, eiid_2_trigger


if __name__ == '__main__':
    atomic_test = ATOMIC_dataset()

