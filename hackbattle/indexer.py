#!/usr/bin/python3



import re
import json
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import transformers
from transformers import RobertaModel, RobertaTokenizerFast, RobertaConfig
from sklearn.preprocessing import LabelEncoder
import sys
import time

import PyPDF2
from io import BytesIO


#PARAMS
MAX_SEQ_LENGTH = 512
MODEL = "roberta-base"
EMBEDDING_DIM = 1280


class DanEncoder(nn.Module):
  def __init__(self, input_dim:int, embedding_dim:int, n_hidden_layers:int, n_hidden_units:[int], dropout_prob:int):
    super(DanEncoder, self).__init__()

    assert n_hidden_layers != 0
    assert len(n_hidden_units) + 1 == n_hidden_layers

    encoder_layers = []
    for i in range(n_hidden_layers):
      if i == n_hidden_layers - 1:
        out_dim = embedding_dim
        encoder_layers.extend(
          [
            nn.Linear(input_dim, out_dim),
          ])
        continue
      else:
        out_dim = n_hidden_units[i]

      encoder_layers.extend(
        [
          nn.Linear(input_dim, out_dim),
          nn.Tanh(),
          nn.Dropout(dropout_prob, inplace=False),
        ]
      )
      input_dim = out_dim
    self.encoder = nn.Sequential(*encoder_layers)

  def forward(self, x_array):
      return self.encoder(x_array)

class Model(nn.Module):
  def __init__(self):
    super().__init__()

    cfg = RobertaConfig.from_pretrained("FacebookAI/roberta-base")
    self.roberta = RobertaModel(cfg)
    # self.roberta.resize_token_embeddings(len(TOKENIZER))
    self.hid_mix = 5
    self.feats = self.roberta.pooler.dense.out_features
    self.embedding_layers = DanEncoder(self.feats, EMBEDDING_DIM, 4, [896, 1024, 1152], 0.1)
    self.tanh = nn.Tanh()

  def forward(self, calc_triplets=False, **kwargs):
    if calc_triplets:
        x = kwargs["x"]
        return self.forward_once(x)
    x1 = kwargs["x1"]
    x2 = kwargs["x2"]
    x3 = kwargs["x3"]
    return self.forward_once(x1), self.forward_once(x2), self.forward_once(x3)

  def forward_once(self, x):
    outputs = self.roberta(x, output_hidden_states=True)
    hidden_states = outputs[2]
    hmix = []
    for i in range(1, self.hid_mix+1):
      hmix.append(hidden_states[-i][:,0].reshape((-1,1,self.feats)))

    hmix_tensor = torch.cat(hmix,1)
    pool_tensor = torch.mean(hmix_tensor,1)
    pool_tensor = self.tanh(pool_tensor)
    embeddings = self.embedding_layers(pool_tensor)
    return embeddings


def extract_text_from_pdf(pdf_bytes):
    # Create a file-like buffer object from the bytes
    pdf_buffer = BytesIO(pdf_bytes)
    
    # Create a PDF reader object
    reader = PyPDF2.PdfReader(pdf_buffer)
    
    # Extract text from each page
    extracted_text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        extracted_text += page.extract_text()

    return extracted_text


if __name__ == "__main__":

    import torch
    import pickle
    import pymongo
    from flask import Flask, render_template, request
    import io
    import re
    import os
    from tika import parser

    LAST_MODIFIED_INDEX = 0
    UPDATE_USER_FILE = "temp.txt"
    DEVICE = "cuda:0"
    MATCH_THRESHOLD = 0.7

    tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
    model = Model().to(DEVICE)


    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["hackbattle"]

    def cleanResume(resumeText):
        resumeText = re.sub(r'http\S+\s*', ' ', resumeText)  # remove URLs
        resumeText = re.sub(r'RT|cc', ' ', resumeText)  # remove RT and cc
        resumeText = re.sub(r'#\S+', '', resumeText)  # remove hashtags
        resumeText = re.sub(r'@\S+', '  ', resumeText)  # remove mentions
        resumeText = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
        resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
        resumeText = re.sub(r'\s+', ' ', resumeText)  # remove extra whitespace
        return resumeText

    while True:

        if LAST_MODIFIED_INDEX < os.path.getmtime(UPDATE_USER_FILE):

            LAST_MODIFIED_INDEX = os.path.getmtime(UPDATE_USER_FILE)
            print("Updating the matches index.")

            db.matches.drop()

            resumes = []
            for applicant in db.applicants.find().sort("_id"):
                resumes.append(
                    cleanResume(extract_text_from_pdf(applicant["resume"]))
                )

            
            resumes = tokenizer(resumes, truncation=True, padding="max_length", return_tensors="pt")["input_ids"].to(DEVICE)

            with torch.no_grad():
                resumes= model.forward_once(resumes)

            matches_found = {}
            # 
            for employer in db.employer.find().sort("_id"):

                positions = []

                for position in employer["positions"]:
                    positions.append(position["desc"])
                    # position

                positions = tokenizer(positions, truncation=True, padding="max_length", return_tensors="pt")["input_ids"].to(DEVICE)

                with torch.no_grad():
                    positions = model(positions)

        break
        time.sleep(1)
