import json
import csv
import numpy as np
from torch.utils.data import DataLoader, Dataset

from torch.utils.data import DataLoader, Dataset
from torch._six import container_abcs, string_classes, int_classes
from transformers import PreTrainedTokenizer
import torch
import re


def load_data_tsv(filename: str, sep: str = '\t') -> list:
    """Load data from a tsv file.
    Args:
        filename (str): filename of the tsv file.
        sep (str, optional): Seperator for a single line. Defaults to '\t'.

    Returns:
        list: A list of (title, content) tuples.
    """
    ret = []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
            cur = l.strip().split(sep, maxsplit=2)
            assert len(cur) == 2, 'Invalid format of data.'
            title, content = cur[0], cur[1]
            ret.append((title, content))
    return ret


def load_data_csv(filename: str, sep: str = ',', skip_first=False) -> list:
    """Load data from a csv file.
    Args:
        filename (str): filename of the csv file.
        sep (str, optional): Seperator for a single line. Defaults to ','.

    Returns:
        list: A list of (title, content) tuples.
    """
    ret = []
    csv_reader = csv.reader(open(filename, encoding='utf-8'), delimiter=sep)
    if skip_first:
        next(csv_reader)
    for l in csv_reader:
        title, content = l[0], l[1]
        ret.append((title, content))
    return ret


def load_data_json(filename: str,
                   title_key: str = "title",
                   content_key: str = "content") -> list:
    """Load data from a json file.
    Args:
        filename (str): filename of the json file.
        title_key (str, optional): Key of the title. Defaults to 'title'.
        content_key (str, optional): Key of the content. Defaults to 'content'.

    Returns:
        list: A list of (title, content) tuples.
    """
    ret = []
    json_obj = json.load(open(filename, encoding='utf-8'))
    for cur in json_obj:
        title = cur[title_key]
        content = cur[content_key]
        ret.append((title, content))
    return ret


def load_data_json_lines(filename: str,
                         title_key: str = "title",
                         content_key: str = "content") -> list:
    """Load data from a json file.
    Args:
        filename (str): filename of the json file.
        title_key (str, optional): Key of the title. Defaults to 'title':str.
        content_key (str, optional): Key of the content. Defaults to 'content':str.

    Returns:
        list: A list of (title, content) tuples.
    """
    ret = []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
            cur = json.loads(l)
            title = cur[title_key]
            content = cur[content_key]
            ret.append((title, content))
    return ret


def sequence_padding(inputs, length=None, padding=0):
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')


class KeyDataset(Dataset):

    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def encode_dict(data,
                tokenizer: PreTrainedTokenizer,
                max_len=512,
                mode='train'):
    """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    """
    ret = []
    for title, content in data:
        text_ids = tokenizer.encode(content,
                                    max_length=max_len,
                                    truncation='only_first')
        if mode == 'train':
            summary_ids = tokenizer.encode(title,
                                           max_length=max_len,
                                           truncation='only_first')
            features = {
                'input_ids': text_ids,
                'decoder_input_ids': summary_ids,
                'attention_mask': [1] * len(text_ids),
                'decoder_attention_mask': [1] * len(summary_ids)
            }

        elif mode == 'dev':  # dev or test
            features = {
                'input_ids': text_ids,
                'attention_mask': [1] * len(text_ids),
                'title': title
            }
        elif mode == 'predict':
            features = {
                'input_ids': text_ids,
                'attention_mask': [1] * len(text_ids),
                'raw_data': content
            }
        ret.append(features)
    return ret

def to_torch_tensor(batch):
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    to_torch_tensor_err_msg_format = (
        "to_torch_tensor: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    to_torch_tensor_err_msg_format.format(elem.dtype))
            return to_torch_tensor([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: to_torch_tensor([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(to_torch_tensor(samples)
                           for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            batch = sequence_padding(batch)
        return to_torch_tensor([to_torch_tensor(elem) for elem in batch])
    raise TypeError(to_torch_tensor_err_msg_format.format(elem_type))


def prepare_data(tokenizer: PreTrainedTokenizer,
                 max_len: int = 1024,
                 batch_size: int = 2,
                 mode='train',
                 data_path=None,
                 dataloader=None):

    loaded_data = None

    if data_path is not None:
        filetype = data_path.split('.')[-1]
        if filetype == 'json':
            loaded_data = load_data_json(data_path)
        elif filetype == 'jsonl':
            loaded_data = load_data_json_lines(data_path)
        elif filetype == 'tsv':
            loaded_data = load_data_tsv(data_path)
        elif filetype == 'csv':
            loaded_data = load_data_csv(data_path)
        else:
            raise Exception('Unsupported data file type: {}'.format(filetype))
    if dataloader is None:
        loaded_data = dataloader

    encoded_data = encode_dict(loaded_data, tokenizer, max_len, mode)
    dataloader = DataLoader(KeyDataset(encoded_data),
                      batch_size=batch_size,
                      collate_fn=to_torch_tensor)
    return dataloader
