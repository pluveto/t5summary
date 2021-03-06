import logging
import time
import traceback
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
from flask_ngrok import run_with_ngrok


class SummaryInferrer(object):

    def __init__(self, model, preprocess_fn):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.preprocess_fn = preprocess_fn

    def generate(self, content: str, max_len_summary: int):
        print("content", content)
        feature = self.preprocess_fn(content)
        print("feature", feature)
        raw_data = feature['raw_data']
        # 剔除 raw_data、title，剩下 input_id 和 attention_mask
        content = {
            k: v
            for k, v in feature.items() if k not in ['raw_data', 'title']
        }
        print("content", content)
        gen = model.generate(
            max_length=max_len_summary,
            eos_token_id=self.tokenizer.sep_token_id,
            decoder_start_token_id=self.tokenizer.cls_token_id,
            **content)
        gen = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        print("gen", gen)
        gen = [item.replace(' ', '') for item in gen]
        print("gen", gen)
        return gen[0]


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
        print("preprocess_fn content", content)

        def _preprocess(s: str):
            # remove \n and space
            s = s.replace('\n', '')
            s = s.replace('　', '')
            s = s.replace(' ', '')
            return s

        contents = []
        # if content is str
        if isinstance(content, str):
            contents.append((None,  _preprocess(content)))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    contents.append((None, _preprocess(item)))
                else:
                    raise ValueError('content should be str or list of str')
        else:
            raise ValueError('content should be str or list of str')
        print("preprocess_fn content 2", contents)

        dataloader = prepare_data(device,
                                  tokenizer=tokenizer,
                                  max_len=MAX_LEN,
                                  batch_size=BATCH_SIZE,
                                  mode='predict',
                                  data=contents)
        ret = next(iter(dataloader))
        print("preprocess_fn ret", ret)
        return ret

    inferrer = SummaryInferrer(model, preprocess_fn)
    logger.info('Starting server')
    # start flask server
    from flask import Flask, request, jsonify
    from flask_restful import Resource, Api

    API_OK = 0
    API_ECODE_INVALID_INPUT = 1
    API_ECODE_SERVER_ERR = 2

    class SummaryResource(Resource):

        def options(self):
            pass

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
                result = inferrer.generate(content, MAX_LEN_SUMMARY)
            except Exception as e:
                logger.error(e)
                logger.error(traceback.format_exc())
                return jsonify({
                    'status': API_ECODE_SERVER_ERR,
                    'message': str(e),
                    'result': None,
                })

            # if len(results) == 0:
            #     return jsonify({
            #         'status': API_ECODE_SERVER_ERR,
            #         'message': 'inferrer returned result with wrong length',
            #         'result': None,
            #     })

            return jsonify({
                'status': API_OK,
                'message': None,
                'result': result,
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
    CORS(app, supports_credentials=True)
    run_with_ngrok(app)
    api = Api(app)
    api.add_resource(SummaryResource, '/summary')
    api.add_resource(IndexResource, '/')
    # app.run(host=HOST, port=PORT, debug=DEBUG_MODE, load_dotenv=False, use_reloader=False)
    logger.info("run first predict")
    test_in = "为促进高校毕业生就业，近日，重庆市工商联举行“民企高校携手促就业行动”推进会。各相关政府部门、高校、行业商协会、民营企业等携手推出多种举措，千方百计促进大学生就业。西南大学启动实施“书记校长走访拓岗促就业”专项行动和“千人千岗就业见习助力乡村振兴”专项行动，进一步深化校地、校企合作和供需对接机制，建立起经常性走访用人单位的长效机制。"
    inferrer.generate(test_in, MAX_LEN_SUMMARY)
    app.run()
