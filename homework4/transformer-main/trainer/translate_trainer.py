import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from utils import metrics
from utils.util import make_src_mask, make_trg_mask
from trainer.trainer import Trainer
from nltk.translate.bleu_score import sentence_bleu
from utils.vocab import Vocab
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from config import Config, Logger
from model import Transformer
from utils.vocab import Vocab
from utils.tokenizer import Tokenizer
from utils.util import make_src_mask, make_trg_mask
from nltk.translate.bleu_score import sentence_bleu
import math
def individual_bleu(reference, candidate):
    
    bleu_1_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu_2_gram = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
    bleu_3_gram = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
    bleu_4_gram = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
    return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram
def translate_sentence(sentence, model, device, zh_vocab, en_vocab, zh_tokenizer, max_len = 100):
    model.eval()
    tokens = zh_tokenizer.tokenizer(sentence)
    tokens = ['<sos>'] + tokens + ['<eos>']
    print(tokens)
    tokens = [zh_vocab.word2id[word] for word in tokens]

    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    src_mask = make_src_mask(src_tensor, zh_vocab, device)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    trg = [en_vocab.word2id['<sos>']]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg).unsqueeze(0).to(device)
        trg_mask = make_trg_mask(trg_tensor, en_vocab, device)
        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_src, src_mask, trg_mask)
            output = model.fc(output)

        pred_token = output.argmax(2)[:,-1].item()
        trg.append(pred_token)
        if pred_token == en_vocab.word2id['<eos>']:
            break
    
    trg_tokens = [en_vocab.id2word[idx] for idx in trg]
    return trg_tokens

class TranslateTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, cfg, logger, data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model=model, optimizer=optimizer, criterion=criterion, cfg=cfg, logger=logger)
        
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.do_validation = self.valid_data_loader is not None
        self.device = 'cuda:0' if cfg['cuda'] else 'cpu'
        self.log_step = cfg['log_step']


    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for idx, (src, trg) in enumerate(self.data_loader):
            src = src.to(self.device)        # tensor type   
            trg = trg.to(self.device) 
            src_mask = make_src_mask(src, self.data_loader.src_vocab, self.device)
            trg_mask = make_trg_mask(trg[:,:-1], self.data_loader.trg_vocab, self.device) 

            self.optimizer.zero_grad()
            
            output = self.model(src, trg[:,:-1], src_mask, trg_mask) # output = [batch_size, target_len-1, target_vocab_size]
            # trg = <sos>, token1, token2, token3, ... 
            # output = token1, token2, token3, ..., <eos>

            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim) # output = [batch size * target_len - 1, target_vocab_size]
            
            trg = trg[:,1:].contiguous().view(-1) # target = [batch_size * targey_len - 1]

            loss = self.criterion(output, trg)
            
            loss.backward()
            # 可调参数 1 可以改为 其他值进行尝试
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            total_loss += loss.item()
            if idx % self.log_step == 0:
                self.logger.info('Train Epoch: {}, {}/{} ({:.0f}%), Loss: {:.6f}'.format(epoch, 
                            idx, 
                            len(self.data_loader), 
                            idx * 100 / len(self.data_loader), 
                            loss.item()
                            ))

        self.logger.info('Train Epoch: {}, total Loss: {:.6f}, mean Loss: {:.6f}'.format(
                epoch,
                total_loss, 
                total_loss / len(self.data_loader)
                ))
        
        if self.do_validation:
            self.logger.debug("start validation")
            val_loss = self._valid_epoch()
        self.logger.info('Train Epoch: {}, validation loss is : {:.3f}'.format(epoch, val_loss))
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return val_loss, None

    def _valid_epoch(self):
        self.model.eval()
        val_loss = 0
        pred = []
        labels = []
        with torch.no_grad():
            for idx, (src, trg) in enumerate(self.valid_data_loader):
                src = src.to(self.device)
                trg = trg.to(self.device) 

                src_mask = make_src_mask(src, self.valid_data_loader.src_vocab, self.device)
                trg_mask = make_trg_mask(trg[:,:-1], self.data_loader.trg_vocab, self.device)

                output = self.model(src, trg[:,:-1], src_mask, trg_mask)
                output = F.log_softmax(output, dim=-1)
                output_dim = output.shape[-1]
                # output = [batch size * target_len - 1, target_vocab_size]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)
                #print("output is:",output)
                #print("target is:",trg)
                val_loss += self.criterion(output, trg)
            src_vocab = Vocab()
            trg_vocab = Vocab()
            
            logger = Logger()
            cfg = Config(logger)
            cfg.load_config('config.json')
            src_vocab.load(cfg.config['src_vocab'])
            trg_vocab.load(cfg.config['trg_vocab'])

            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

            zh = 'zh_core_web_md'
            en = 'en_core_web_md'

            model = Transformer(src_vocab_size=src_vocab.vocab_size, target_vocab_size=trg_vocab.vocab_size, device=device, **cfg.config)
            checkpoint = torch.load(cfg.config['resume_path'])
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            

            zh_tokenizer = Tokenizer(zh) 
            all_bleu = 0   
            with open('dataset/val.zh', 'r') as file_val_zh, open('dataset/val.en', 'r') as file_val_en:
            # 确保两个文件的行是同步处理的
                for line_zh, line_en in zip(file_val_zh, file_val_en):
                    # 处理每一行的数据
                    print(type(line_zh))
                    res = translate_sentence(line_zh, model, device, src_vocab, trg_vocab, zh_tokenizer)
                    i_bleu = individual_bleu(line_en, res)
                    print(i_bleu)

                    # 计算 Bleu 指标
                    i_bleu = individual_bleu(line_en, res)  
                    all_bleu += i_bleu  
                
        return val_loss / len(self.valid_data_loader), all_bleu/len(file_val_en)
