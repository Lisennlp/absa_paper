import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model_gcn import GAT, GCN, Rel_GAT
from model_utils import LinearAttention, DotprodAttention, RelationAttention, Highway, mask_logits
from tree import *
from transformers import XLNetModel, XLNetTokenizer,XLNetConfig

import nltk
nltk.download('punkt_tab')

class Aspect_Text_GAT_ours(nn.Module):
    """
    Full model in reshaped tree
    """
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Aspect_Text_GAT_ours, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)

        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)
        gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        self.gat_dep = [RelationAttention(in_dim = args.embedding_dim).to(args.device) for i in range(args.num_heads)]
        if args.gat_attention_type == 'linear':
            self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        elif args.gat_attention_type == 'dotprod':
            self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        else:
            # reshaped gcn
            self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)

        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim)

        last_hidden_size = args.hidden_size * 4

        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps-1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        feature = self.embed(sentence)  # (N, L, D)
        aspect_feature = self.embed(aspect) # (N, L', D)
        feature = self.dropout(feature)
        aspect_feature = self.dropout(aspect_feature)

        if self.args.highway:
            feature = self.highway(feature)
            aspect_feature = self.highway(aspect_feature)

        feature, _ = self.bilstm(feature) # (N,L,D)
        aspect_feature, _ = self.bilstm(aspect_feature) #(N,L,D)

        aspect_feature = aspect_feature.mean(dim = 1) # (N, D)

        ############################################################################################
        # do gat thing, NNP, NP之类的
        dep_feature = self.dep_embed(dep_tags)
        if self.args.highway:
            dep_feature = self.highway_dep(dep_feature)

        dep_out = [g(feature, dep_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out = torch.cat(dep_out, dim = 1) # (N, H, D)
        dep_out = dep_out.mean(dim = 1) # (N, D)

        if self.args.gat_attention_type == 'gcn':
            gat_out = self.gat(feature) # (N, L, D)
            fmask = fmask.unsqueeze(2)
            gat_out = gat_out * fmask
            gat_out = F.relu(torch.sum(gat_out, dim = 1)) # (N, D)

        else:
            gat_out = [g(feature, aspect_feature, fmask).unsqueeze(1) for g in self.gat]
            gat_out = torch.cat(gat_out, dim=1)
            gat_out = gat_out.mean(dim=1)

        feature_out = torch.cat([dep_out,  gat_out], dim = 1) # (N, D')
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit

class Aspect_Text_GAT_only(nn.Module):
    """
    reshape tree in GAT only
    """
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Aspect_Text_GAT_only, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        if args.highway:
            self.highway = Highway(args.num_layers, args.embedding_dim)

        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                                  bidirectional=True, batch_first=True, num_layers=args.num_layers)
        gcn_input_dim = args.hidden_size * 2

        # if args.gat:
        if args.gat_attention_type == 'linear':
            self.gat = [LinearAttention(in_dim = gcn_input_dim, mem_dim = gcn_input_dim).to(args.device) for i in range(args.num_heads)] # we prefer to keep the dimension unchanged
        elif args.gat_attention_type == 'dotprod':
            self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        else:
            # reshaped gcn
            self.gat = nn.Linear(gcn_input_dim, gcn_input_dim)


        last_hidden_size = args.hidden_size * 2

        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps-1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_rels: (batch_size, text_length) relation
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
            dep_dirs: (batch_size, text_length) the directions each node to the aspect
        '''
        fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        dmask = (torch.zeros_like(dep_tags) != dep_tags).float()  # (N ,L)

        feature = self.embed(sentence)  # (N, L, D)
        aspect_feature = self.embed(aspect) # (N, L', D)
        feature = self.dropout(feature)
        aspect_feature = self.dropout(aspect_feature)


        if self.args.highway:
            feature = self.highway(feature)
            aspect_feature = self.highway(aspect_feature)

        feature, _ = self.bilstm(feature) # (N,L,D)
        aspect_feature, _ = self.bilstm(aspect_feature) #(N,L,D)

        aspect_feature = aspect_feature.mean(dim = 1) # (N, D)

        ############################################################################################

        if self.args.gat_attention_type == 'gcn':
            gat_out = self.gat(feature) # (N, L, D)
            fmask = fmask.unsqueeze(2)
            gat_out = gat_out * fmask
            gat_out = F.relu(torch.sum(gat_out, dim = 1)) # (N, D)

        else:
            gat_out = [g(feature, aspect_feature, fmask).unsqueeze(1) for g in self.gat]
            gat_out = torch.cat(gat_out, dim=1)
            gat_out = gat_out.mean(dim=1)

        feature_out = gat_out # (N, D')
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit


class Pure_Bert(nn.Module):
    '''
    Bert for sequence classification.
    '''

    def __init__(self, args, hidden_size=256):
        super(Pure_Bert, self).__init__()

        config = BertConfig.from_pretrained(args.bert_model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        self.bert = BertModel.from_pretrained(
            args.bert_model_dir, config=config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        layers = [nn.Linear(
            config.hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, args.num_classes)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids, token_type_ids):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids)
        # pool output is usually *not* a good summary of the semantic content of the input,
        # you're often better with averaging or poolin the sequence of hidden-states for the whole input sequence.
        pooled_output = outputs[1]
        # pooled_output = torch.mean(pooled_output, dim = 1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class Aspect_Bert_GAT(nn.Module):
    '''
    R-GAT with bert
    '''

    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Aspect_Bert_GAT, self).__init__()
        self.args = args

        # Bert
        config = BertConfig.from_pretrained(args.bert_model_dir)
        self.bert = BertModel.from_pretrained(
            args.bert_model_dir, config=config, from_tf =False)
        self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(args.dropout)
        args.embedding_dim = config.hidden_size  # 768

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)


        gcn_input_dim = args.embedding_dim

        # GAT
        self.gat_dep = [RelationAttention(in_dim=args.embedding_dim).to(args.device) for i in range(args.num_heads)]


        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim)

        last_hidden_size = args.embedding_dim * 2
        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, input_ids, input_aspect_ids, word_indexer, aspect_indexer,input_cat_ids,segment_ids, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs):
        fmask = (torch.ones_like(word_indexer) != word_indexer).float()  # (N，L)
        fmask[:,0] = 1
        outputs = self.bert(input_cat_ids, token_type_ids = segment_ids)
        feature_output = outputs[0] # (N, L, D)
        pool_out = outputs[1] #(N, D)

        # index select, back to original batched size.
        feature = torch.stack([torch.index_select(f, 0, w_i)
                               for f, w_i in zip(feature_output, word_indexer)])

        ############################################################################################
        # do gat thing
        dep_feature = self.dep_embed(dep_tags)
        if self.args.highway:
            dep_feature = self.highway_dep(dep_feature)

        dep_out = [g(feature, dep_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out = torch.cat(dep_out, dim=1)  # (N, H, D)
        dep_out = dep_out.mean(dim=1)  # (N, D) # 头平均


        feature_out = torch.cat([dep_out,  pool_out], dim=1)  # (N, D')
        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit


class Aspect_Xlnet_Cnn_Lstm(nn.Module):
    '''
    R-GAT with xlnet
    '''
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(Aspect_Xlnet_Cnn_Lstm, self).__init__()
        self.args = args

        # xlnet
        config = XLNetConfig.from_pretrained(args.bert_model_dir)

        self.xlnet = XLNetModel.from_pretrained(args.bert_model_dir, config=config, from_tf =False)
        self.dropout_xlnet = nn.Dropout(0.0)
        self.dropout = nn.Dropout(args.dropout)
        args.embedding_dim = config.hidden_size  # 1024

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)

        gcn_input_dim = args.embedding_dim

        # GAT
        self.gat_dep = [RelationAttention(in_dim=args.embedding_dim).to(args.device) for i in range(args.num_heads)]

        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim)
        self.pos_embed= nn.Embedding(pos_tag_num, args.embedding_dim)
        self.gat = [DotprodAttention().to(args.device) for i in range(args.num_heads)]
        last_hidden_size = args.embedding_dim * 4
        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.BatchNorm1d(args.final_hidden_size),nn.ReLU()]
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)
        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)
        self.cnn = nn.Conv1d(in_channels=args.embedding_dim, out_channels=args.embedding_dim, kernel_size=5, padding=2)
    def forward(self, input_ids, input_aspect_ids, word_indexer, aspect_indexer,input_cat_ids,segment_ids, pos_class, dep_tags, text_len, aspect_len, dep_rels, dep_heads, aspect_position, dep_dirs, step=0):
        fmask = (torch.ones_like(word_indexer) != word_indexer).float()  # (N，L)
        fmask[:,0] = 1
        outputs = self.xlnet(input_cat_ids)
        feature_asp_output = outputs[0]
        #Apply CNN
        feature_asp_output = feature_asp_output.permute(0, 2, 1)  # (N, D, L)
        feature_asp_output = self.cnn(feature_asp_output)
        feature_asp_output = feature_asp_output.permute(0, 2, 1)  # (N, L, D)
        feature_asp_output, _ = self.bilstm(feature_asp_output)
        pool_out2 = torch.mean(feature_asp_output, dim=1)
        # index select, back to original batched size.
        feature = torch.stack([torch.index_select(f, 0, w_i)
                               for f, w_i in zip(feature_asp_output, word_indexer)])

        ############################################################################################
        # do gat thing
        dep_feature = self.dep_embed(dep_tags)
        # print(f'step: {step}')
        # if step == 22:
        #     __import__('ipdb').set_trace()
        pos_feature = self.dep_embed(pos_class)

        if self.args.highway:
            dep_feature = self.highway_dep(dep_feature)
        if self.args.highway:
            pos_feature = self.highway_dep(pos_feature)
        # fmask是前面为1，后面的aspect为0

        dep_out = [g(feature, pos_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out = torch.cat(dep_out, dim=1)  # (N, H, D)  多个 dep_out沿着维度 1 进行拼接
        dep_out = dep_out.mean(dim=1)  # (N, D)求均值

        dep_out2 = [g(feature, dep_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        dep_out2 = torch.cat(dep_out2, dim=1)  # (N, H, D)  多个 dep_out沿着维度 1 进行拼接
        dep_out2 = dep_out2.mean(dim=1)  # (N, D)求均值

        feature_out1 = torch.cat([dep_out2,  pool_out2], dim=1)  # (N, D')拼接
        feature_out2=torch.cat([dep_out,  pool_out2], dim=1)  # (N, D')拼接

        feature_out=torch.cat([feature_out1, feature_out2], dim=1)
        # feature_out=torch.cat([dep_out2, dep_out, pool_out2], dim=1)
        # __import__('ipdb').set_trace()

        # feature_out = gat_out
        #############################################################################################
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit

        # fmask = (torch.ones_like(word_indexer) != word_indexer).float()  # (N，L)
        # fmask[:,0] = 1
        # outputs3=self.xlnet(input_cat_ids)
        # feature_asp_output=outputs3[0]
        # #Apply CNN
        # feature_asp_output, _ = self.bilstm(feature_asp_output)
        # pool_out2 = torch.mean(feature_asp_output, dim=1)
        # # index select, back to original batched size.
        # feature = torch.stack([torch.index_select(f, 0, w_i)
        #                        for f, w_i in zip(feature_asp_output, word_indexer)])

        # ############################################################################################
        # # do gat thing
        # dep_feature = self.dep_embed(dep_tags)
        # pos_feature = self.dep_embed(pos_class)

        # if self.args.highway:
        #     dep_feature = self.highway_dep(dep_feature)
        # if self.args.highway:
        #     pos_feature = self.highway_dep(pos_feature)
        # dep_out2 = [g(feature, dep_feature, fmask).unsqueeze(1) for g in self.gat_dep]  # (N, 1, D) * num_heads
        # dep_out2 = torch.cat(dep_out2, dim=1)  # (N, H, D)  多个 dep_out沿着维度 1 进行拼接
        # dep_out2 = dep_out2.mean(dim=1)  # (N, D)求均值
        # feature_out1 = torch.cat([dep_out2,  pool_out2], dim=1)  # (N, D')拼接
        # # __import__('ipdb').set_trace()
        # x = self.dropout(feature_out1)
        # x = self.fcs(x)
        # logit = self.fc_final(x)
        # return logit

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0
