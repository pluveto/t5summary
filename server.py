import logging
import time
from common import init_logger, load_dotenv
from data import prepare_data
from tokenizer import T5TokenizerFast
from transformers import BertTokenizer
from torch._six import container_abcs, string_classes, int_classes
import torch
from torch.utils.data import DataLoader, Dataset
import re
import os
import csv
import argparse
from tqdm.auto import tqdm
from multiprocessing import Pool, Process
import pandas as pd
import numpy as np
from flask_cors import CORS


class SummaryInferrer(object):

    def __init__(self, model, preprocess_fn):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.preprocess_fn = preprocess_fn

    def generate(self, content: str, max_len_summary: int):
        feature = self.preprocess_fn(content)
        raw_data = feature['raw_data']
        content = {k: v for k, v in feature.items() if k != 'raw_data'}
        gen = model.generate(max_length=max_len_summary,
                             eos_token_id=tokenizer.sep_token_id,
                             decoder_start_token_id=tokenizer.cls_token_id,
                             **content)
        gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
        results = [
            "{}\t{}".format(x.replace(' ', ''), y)
            for x, y in zip(gen, raw_data)
        ]
        return results


if __name__ == '__main__':

    # initializing arguments

    try:
        load_dotenv([".env", "dev.env"])
    except FileNotFoundError:
        print("not found .env file")

    PROJECT_ID = os.getenv('PROJECT_ID')
    MAX_LEN = int(os.getenv('MAX_LEN'))
    MAX_LEN_SUMMARY = int(os.getenv('MAX_LEN_SUMMARY'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
    MODEL_PATH = os.getenv('MODEL_PATH')
    TOKENIZER_PATH = os.getenv('TOKENIZER_PATH')
    DEBUG_MODE = os.getenv('DEBUG')
    HOST = os.getenv('HOST', 'localhost')
    PORT = int(os.getenv('PORT', 5000))

    assert PROJECT_ID is not None, 'PROJECT_ID is not set'
    assert MAX_LEN is not None, 'MAX_LEN is not set'
    assert MAX_LEN > 0, 'MAX_LEN should be greater than 0'
    assert BATCH_SIZE > 0, 'BATCH_SIZE should be greater than 0'
    assert MAX_LEN_SUMMARY > 0, 'MAX_LEN_SUMMARY should be greater than 0'
    assert BATCH_SIZE is not None, 'BATCH_SIZE is not set'
    assert MODEL_PATH is not None, 'MODEL_PATH is not set'
    assert TOKENIZER_PATH is not None, 'TOKENIZER_PATH is not set'

    logger = init_logger(logging.getLogger(PROJECT_ID))

    logger.info('Project %s loaded', PROJECT_ID)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Using device: {}'.format(device))

    # initializing model
    logger.info('Loading tokenzier...')
    tokenizer = T5TokenizerFast.from_pretrained(TOKENIZER_PATH)
    logger.info('Loading model...')
    model = torch.load(MODEL_PATH, map_location=device)


    def preprocess_fn(content):
        contents = []
        # if content is str
        if isinstance(content, str):
            contents.append([{
                'title': None,
                'content': content,
            }])
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    contents.append([{
                        'title': None,
                        'content': item,
                    }])
                else:
                    raise ValueError('content should be str or list of str')
        else:
            raise ValueError('content should be str or list of str')

        data = prepare_data(tokenizer=tokenizer,
                            max_len=MAX_LEN,
                            batch_size=BATCH_SIZE,
                            mode='predict',
                            data=contents)
        return data

    inferrer = SummaryInferrer(model, preprocess_fn)
    logger.info('Starting server')
    # start flask server
    from flask import Flask, request, jsonify
    from flask_restful import Resource, Api

    API_OK = 0
    API_ECODE_INVALID_INPUT = 1
    API_ECODE_SERVER_ERR = 2

    class SummaryResource(Resource):

        def post(self):
            content = request.json['content']
            if content is None:
                return jsonify({
                    'status': API_ECODE_INVALID_INPUT,
                    'message': 'content is required',
                    'result': None,
                })
            content = content.strip()

            if len(content) <= MAX_LEN_SUMMARY:
                # return jsonify({
                #     'status': API_ECODE_INVALID_INPUT,
                #     'message': 'content is too short',
                #     'result': None,
                # })
                return jsonify({
                    'status': API_OK,
                    'message': None,
                    'result': content,
                })

            # truncate content if it is too long
            if len(content) > MAX_LEN:
                content = content[:MAX_LEN]
            try:
                results = inferrer.generate(content, MAX_LEN_SUMMARY)
            except Exception as e:
                logger.error(e)
                return jsonify({
                    'status': API_ECODE_SERVER_ERR,
                    'message': str(e),
                    'result': None,
                })

            if len(results) != 1:
                return jsonify({
                    'status': API_ECODE_SERVER_ERR,
                    'message': 'inferrer returned result with wrong length',
                    'result': None,
                })

            return jsonify({
                'status': API_OK,
                'message': None,
                'result': results,
            })

    starttime = time.time()

    class IndexResource(Resource):

        def get(self):
            return jsonify({
                'status': API_OK,
                'result': {
                    'uptime':
                    int(time.time() - starttime),  # uptime in seconds
                },
                'message': None,
            })

    app = Flask(__name__)
    app = CORS(app, supports_credentials=True)
    api = Api(app)
    api.add_resource(SummaryResource, '/summary')
    api.add_resource(IndexResource, '/')
    app.run(host=HOST, port=PORT, debug=DEBUG_MODE, load_dotenv=False, use_reloader=False)