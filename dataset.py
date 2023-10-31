from __future__ import absolute_import, division

import os
import glob
import shutil
import logging
import gzip, zipfile, tarfile
import multiprocessing
from collections import defaultdict

import numpy as np
# from . import cfg

logger = logging.getLogger(__name__)

class Dataset(object):
    """
    Graph dataset.

    Parameters:
        name (str): name of dataset
        urls (dict, optional): url(s) for each split,
            can be either str or list of str
        members (dict, optional): zip member(s) for each split,
            leave empty for default

    Datasets contain several splits, such as train, valid and test.
    For each split, there are one or more URLs, specifying the file to download.
    You may also specify the zip member to extract.
    When a split is accessed, it will be automatically downloaded and decompressed
    if it is not present.

    You can assign a preprocess for each split, by defining a function with name [split]_preprocess::

        class MyDataset(Dataset):
            def __init__(self):
                super(MyDataset, self).__init__(
                    "my_dataset",
                    train="url/to/train/split",
                    test="url/to/test/split"
                )

            def train_preprocess(self, input_file, output_file):
                with open(input_file, "r") as fin, open(output_file, "w") as fout:
                    fout.write(fin.read())

        f = open(MyDataset().train)

    If the preprocess returns a non-trivial value, then it is assigned to the split,
    otherwise the file name is assigned.
    By convention, only splits ending with ``_data`` have non-trivial return value.

    See also:
        Pre-defined preprocess functions
        :func:`csv2txt`,
        :func:`top_k_label`,
        :func:`induced_graph`,
        :func:`edge_split`,
        :func:`link_prediction_split`,
        :func:`image_feature_data`
    """
    def __init__(self, name, urls=None, members=None):
        self.name = name
        self.urls = urls or {}
        self.members = members or {}
        for key in self.urls:
            if isinstance(self.urls[key], str):
                self.urls[key] = [self.urls[key]]
            if key not in self.members:
                self.members[key] = [None] * len(self.urls[key])
            elif isinstance(self.members[key], str):
                self.members[key] = [self.members[key]]
            if len(self.urls[key]) != len(self.members[key]):
                raise ValueError("Number of members is inconsistent with number of urls in `%s`" % key)
        self.path = os.path.join("./data", self.name)

    def relpath(self, path):
        return os.path.relpath(path, self.path)

    def download(self, url):
        from six.moves.urllib.request import urlretrieve

        save_file = os.path.basename(url)
        if "?" in save_file:
            save_file = save_file[:save_file.find("?")]
        save_file = os.path.join(self.path, save_file)
        if save_file in self.local_files():
            return save_file

        logger.info("downloading %s to %s" % (url, self.relpath(save_file)))
        urlretrieve(url, save_file)
        return save_file

    def extract(self, zip_file, member=None):
        zip_name, extension = os.path.splitext(zip_file)
        if zip_name.endswith(".tar"):
            extension = ".tar" + extension
            zip_name = zip_name[:-4]

        if extension == ".txt":
            return zip_file
        elif member is None:
            save_file = zip_name
        else:
            save_file = os.path.join(os.path.dirname(zip_name), os.path.basename(member))
        if save_file in self.local_files():
            return save_file

        if extension == ".gz":
            logger.info("extracting %s to %s" % (self.relpath(zip_file), self.relpath(save_file)))
            with gzip.open(zip_file, "rb") as fin, open(save_file, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        elif extension == ".tar.gz" or extension == ".tar":
            if member is None:
                logger.info("extracting %s to %s" % (self.relpath(zip_file), self.relpath(save_file)))
                with tarfile.open(zip_file, "r") as fin:
                    fin.extractall(save_file)
            else:
                logger.info("extracting %s from %s to %s" % (member, self.relpath(zip_file), self.relpath(save_file)))
                with tarfile.open(zip_file, "r").extractfile(member) as fin, open(save_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
        elif extension == ".zip":
            if member is None:
                logger.info("extracting %s to %s" % (self.relpath(zip_file), self.relpath(save_file)))
                with zipfile.ZipFile(zip_file) as fin:
                    fin.extractall(save_file)
            else:
                logger.info("extracting %s from %s to %s" % (member, self.relpath(zip_file), self.relpath(save_file)))
                with zipfile.ZipFile(zip_file).open(member, "r") as fin, open(save_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
        else:
            raise ValueError("Unknown file extension `%s`" % extension)

        return save_file

    def get_file(self, key):
        file_name = os.path.join(self.path, "%s_%s.txt" % (self.name, key))
        if file_name in self.local_files():
            return file_name

        urls = self.urls[key]
        members = self.members[key]
        preprocess_name = key + "_preprocess"
        preprocess = getattr(self, preprocess_name, None)
        if len(urls) > 1 and preprocess is None:
            raise AttributeError(
                "There are non-trivial number of files, but function `%s` is not found" % preprocess_name)

        extract_files = []
        for url, member in zip(urls, members):
            download_file = self.download(url)
            extract_file = self.extract(download_file, member)
            extract_files.append(extract_file)
        if preprocess:
            result = preprocess(*(extract_files + [file_name]))
            if result is not None:
                return result
        elif os.path.isfile(extract_files[0]):
            logger.info("renaming %s to %s" % (self.relpath(extract_files[0]), self.relpath(file_name)))
            shutil.move(extract_files[0], file_name)
        else:
            raise AttributeError(
                "There are non-trivial number of files, but function `%s` is not found" % preprocess_name)

        return file_name

    def local_files(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        return set(glob.glob(os.path.join(self.path, "*")))

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        if key in self.urls:
            return self.get_file(key)
        raise AttributeError("Can't resolve split `%s`" % key)

    def csv2txt(self, csv_file, txt_file):
        """
        Convert ``csv`` to ``txt``.

        Parameters:
            csv_file: csv file
            txt_file: txt file
        """
        logger.info("converting %s to %s" % (self.relpath(csv_file), self.relpath(txt_file)))
        with open(csv_file, "r") as fin, open(txt_file, "w") as fout:
            for line in fin:
                fout.write(line.replace(",", "\t"))

    def top_k_label(self, label_file, save_file, k, format="node-label"):
        """
        Extract top-k labels.

        Parameters:
            label_file (str): label file
            save_file (str): save file
            k (int): top-k labels will be extracted
            format (str, optional): format of label file,
            can be 'node-label' or '(label)-nodes':
                - **node-label**: each line is [node] [label]
                - **(label)-nodes**: each line is [node]..., no explicit label
        """
        logger.info("extracting top-%d labels of %s to %s" % (k, self.relpath(label_file), self.relpath(save_file)))
        if format == "node-label":
            label2nodes = defaultdict(list)
            with open(label_file, "r") as fin:
                for line in fin:
                    node, label = line.split()
                    label2nodes[label].append(node)
        elif format == "(label)-nodes":
            label2nodes = {}
            with open(label_file, "r") as fin:
                for i, line in enumerate(fin):
                    label2nodes[i] = line.split()
        else:
            raise ValueError("Unknown file format `%s`" % format)

        labels = sorted(label2nodes, key=lambda x: len(label2nodes[x]), reverse=True)[:k]
        with open(save_file, "w") as fout:
            for label in sorted(labels):
                for node in sorted(label2nodes[label]):
                    fout.write("%s\t%s\n" % (node, label))

    def induced_graph(self, graph_file, label_file, save_file):
        """
        Induce a subgraph from labeled nodes. All edges in the induced graph have at least one labeled node.

        Parameters:
            graph_file (str): graph file
            label_file (str): label file
            save_file (str): save file
        """
        logger.info("extracting subgraph of %s induced by %s to %s" %
              (self.relpath(graph_file), self.relpath(label_file), self.relpath(save_file)))
        nodes = set()
        with open(label_file, "r") as fin:
            for line in fin:
                nodes.update(line.split())
        with open(graph_file, "r") as fin, open(save_file, "w") as fout:
            for line in fin:
                if not line.startswith("#"):
                    u, v = line.split()
                    if u not in nodes or v not in nodes:
                        continue
                    fout.write("%s\t%s\n" % (u, v))

    def edge_split(self, graph_file, files, portions):
        """
        Divide a graph into several splits.

        Parameters:
            graph_file (str): graph file
            files (list of str): file names
            portions (list of float): split portions
        """
        assert len(files) == len(portions)
        logger.info("splitting graph %s into %s" %
                    (self.relpath(graph_file), ", ".join([self.relpath(file) for file in files])))
        np.random.seed(1024)

        portions = np.cumsum(portions, dtype=np.float32) / np.sum(portions)
        files = [open(file, "w") for file in files]
        with open(graph_file, "r") as fin:
            for line in fin:
                i = np.searchsorted(portions, np.random.rand())
                files[i].write(line)
        for file in files:
            file.close()

    def link_prediction_split(self, graph_file, files, portions):
        """
        Divide a normal graph into a train split and several test splits for link prediction use.
        Each test split contains half true and half false edges.

        Parameters:
            graph_file (str): graph file
            files (list of str): file names,
                the first file is treated as train file
            portions (list of float): split portions
        """
        assert len(files) == len(portions)
        logger.info("splitting graph %s into %s" %
                    (self.relpath(graph_file), ", ".join([self.relpath(file) for file in files])))
        np.random.seed(1024)

        nodes = set()
        edges = set()
        portions = np.cumsum(portions, dtype=np.float32) / np.sum(portions)
        files = [open(file, "w") for file in files]
        num_edges = [0] * len(files)
        with open(graph_file, "r") as fin:
            for line in fin:
                u, v = line.split()[:2]
                nodes.update([u, v])
                edges.add((u, v))
                i = np.searchsorted(portions, np.random.rand())
                if i == 0:
                    files[i].write(line)
                else:
                    files[i].write("%s\t%s\t1\n" % (u, v))
                num_edges[i] += 1

        nodes = list(nodes)
        for file, num_edge in zip(files[1:], num_edges[1:]):
            for _ in range(num_edge):
                valid = False
                while not valid:
                    u = nodes[int(np.random.rand() * len(nodes))]
                    v = nodes[int(np.random.rand() * len(nodes))]
                    valid = u != v and (u, v) not in edges and (v, u) not in edges
                file.write("%s\t%s\t0\n" % (u, v))
        for file in files:
            file.close()

    def image_feature_data(self, dataset, model="resnet50", batch_size=128):
        """
        Compute feature vectors for an image dataset using a neural network.

        Parameters:
            dataset (torch.utils.data.Dataset): dataset
            model (str or torch.nn.Module, optional): pretrained model.
                If it is a str, use the last hidden model of that model.
            batch_size (int, optional): batch size
        """
        import torch
        import torchvision
        from torch import nn

        logger.info("computing %s feature" % model)
        if isinstance(model, str):
            full_model = getattr(torchvision.models, model)(pretrained=True)
            model = nn.Sequential(*list(full_model.children())[:-1])
        num_worker = multiprocessing.cpu_count()
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size, num_workers=num_worker, shuffle=False)
        model = model.cuda()
        model.eval()

        features = []
        with torch.no_grad():
            for i, (batch_images, batch_labels) in enumerate(data_loader):
                if i % 100 == 0:
                    logger.info("%g%%" % (100.0 * i * batch_size / len(dataset)))
                batch_images = batch_images.cuda()
                batch_features = model(batch_images).view(batch_images.size(0), -1).cpu().numpy()
                features.append(batch_features)
        features = np.concatenate(features)

        return features

class Wikidata5m(Dataset):
    """
    Wikidata5m knowledge graph dataset.

    Splits:
        train, valid, test
    """
    def __init__(self):
        super(Wikidata5m, self).__init__(
            "wikidata5m",
            urls={
                "train": "https://www.dropbox.com/s/dty6ufe1gg6keuc/wikidata5m.txt.gz?dl=1",
                "valid": "https://www.dropbox.com/s/dty6ufe1gg6keuc/wikidata5m.txt.gz?dl=1",
                "test": "https://www.dropbox.com/s/dty6ufe1gg6keuc/wikidata5m.txt.gz?dl=1",
                "entity": "https://www.dropbox.com/s/bgmgvk8brjwpc9w/entity.txt.gz?dl=1",
                "relation": "https://www.dropbox.com/s/37jxki93gguv0pp/relation.txt.gz?dl=1",
                "alias2entity": [], # depends on `entity`
                "alias2relation": [] # depends on `relation`
            }
        )

    def train_preprocess(self, graph_file, train_file):
        valid_file = train_file[:train_file.rfind("train.txt")] + "valid.txt"
        test_file = train_file[:train_file.rfind("train.txt")] + "test.txt"
        self.edge_split(graph_file, [train_file, valid_file, test_file], portions=[4000, 1, 1])

    def valid_preprocess(self, graph_file, valid_file):
        train_file = valid_file[:valid_file.rfind("valid.txt")] + "train.txt"
        test_file = valid_file[:valid_file.rfind("valid.txt")] + "test.txt"
        self.edge_split(graph_file, [train_file, valid_file, test_file], portions=[4000, 1, 1])

    def test_preprocess(self, graph_file, test_file):
        train_file = test_file[:test_file.rfind("valid.txt")] + "train.txt"
        valid_file = test_file[:test_file.rfind("train.txt")] + "valid.txt"
        self.edge_split(graph_file, [train_file, valid_file, test_file], portions=[4000, 1, 1])

    def load_alias(self, alias_file):
        alias2object = {}
        ambiguous = set()
        with gzip.open('./entity.txt.gz','rt') as f:
            for line in f:
                tokens = line.strip().split("\t")
                object = tokens[0]
                for alias in tokens[1:]:
                    if alias in alias2object and alias2object[alias] != object:
                        ambiguous.add(alias)
                    alias2object[alias] = object
            for alias in ambiguous:
                alias2object.pop(alias)
        return alias2object

    def alias2entity_preprocess(self, save_file):
        return self.load_alias(self.entity)

    def alias2relation_preprocess(self, save_file):
        return self.load_alias(self.relation)
    
wikidata5m = Wikidata5m()
