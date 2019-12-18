import os
import numpy as np
import torch
import shutil
import json
import logging

from torch_geometric.data import InMemoryDataset, Data
from datasets.util.gxl_parser import ParsedGxlDataset


class GxlDataset(InMemoryDataset):
    def __init__(self, root_path, transform=None, pre_transform=None, categorical_features=None, rebuild_dataset=True,
                 subset='', use_position=True, features_to_use=None, **kwargs):
        """
        This class reads a IAM dataset in gxl format (tested for AIDS, Fingerprint, Letter)

        Parameters
        ----------
        use_position
        root_path: str
            Path to the dataset folder. There has to be a sub-folder 'data' where the graph gxl files and the train.cxl,
            valid.cxl and test.cxl files are.
        transform:
        pre_transform:
        categorical_features : str
            path to json file
            optional parameter: dictionary with first level 'edge' and/or 'node' and then a list of the attribute names
            that need to be one-hot encoded ( = are categorical)
            e.g. {'node': ['symbol'], 'edge': ['valence']}}
        rebuild_dataset: bool
            True if dataset should be re-processed
        subset: str
            'valid', 'train' or 'test (or empty) --> name has to match the corresponding cxl file
        features_to_use: string
            comma delimited list of the attribute names (either node or edges) that should be used
            e.g. "symbol,valence"

        """
        self.transform = None
        self.target_transform = None

        self.root = root_path
        self.subset = subset
        self.use_position = use_position
        # should we load the saved dataset from processed or should it be rebuilt
        processed_path = os.path.join(self.root, 'processed')
        if rebuild_dataset and os.path.exists(processed_path):
            # delete the root/processed folder
            shutil.rmtree(processed_path)

        # initiate the two dictionaries
        self.categorical_features = self._setup_cat_feature_dict(categorical_features)
        if features_to_use is not None:
            assert type(features_to_use) == str
            self.features_to_use = [item for item in features_to_use.split(',')]
        else:
            self.features_to_use = features_to_use


        self.name = os.path.basename(root_path)

        super(GxlDataset, self).__init__(self.root, transform, pre_transform)

        # split the dataset and the slices into three subsets
        self.data, self.slices, self.config = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """
        The name of the files to find in the :obj:`self.raw_dir` folder in order to skip the download.
        Empty list because files cannot be downloaded automatically.
        """
        return []

    @property
    def processed_file_names(self):
        processed_path = os.path.join(self.root, 'processed')
        # check if /processed folder exists, if not create it
        if not os.path.exists(processed_path):
            os.mkdir(processed_path)
        filler = '-' if len(self.subset) > 0 else ''
        return [os.path.basename(self.root) + filler + self.subset + '.dataset']

    def download(self):
        """
        Files cannot be automatically downloaded.
        """
        pass

    def process(self):
        """
        Processes the dataset to the :obj:`self.processed_dir` folder.
        """
        gxl_dataset = ParsedGxlDataset(os.path.join(self.root, 'data'), self.categorical_features, subset=self.subset,
                                       use_position=self.use_position, features_to_use=self.features_to_use)

        config = gxl_dataset.config
        data_list = []

        # create the dataset lists: transform the graphs in the GxlDataset into pytorch geometric Data objects
        for graph in gxl_dataset.graphs:
            # x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes, num_node_features]`
            # y (Tensor, optional): Graph or node targets with arbitrary shape.
            # edge_index (LongTensor, optional): Graph connectivity in COO format with shape :obj:`[2, num_edges]`
            # edge_attr (Tensor, optional): Edge feature matrix with shape :obj:`[num_edges, num_edge_features]`
            # y (Tensor, optional): Graph or node targets with arbitrary shape
            x = torch.tensor(graph.node_features, dtype=torch.float) if graph.node_features is not None else None
            edge_index = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(graph.edge_features, dtype=torch.float) if graph.edge_features is not None else None
            pos = torch.tensor(graph.node_position, dtype=torch.float) if graph.node_position is not None else None
            y = graph.class_label if graph.class_label is not None else None
            g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)

            data_list.append(g)

        # save the data
        data, slices = self.collate(data_list)

        if data.y is not None:
            index, counts = np.unique(np.array(data.y), return_counts=True)
            counts = counts / sum(counts)
            config['class_freq'] = (index, counts)

        torch.save((data, slices, config), os.path.join(self.processed_paths[0]))

    @staticmethod
    def _setup_cat_feature_dict(json_path):
        if json_path is None:
            dict = {'node': [], 'edge': []}
        else:
            # read the json file
            logging.info('Loading categorical variables from JSON ({})'.format(json_path))
            with open('strings.json') as f:
                dict = json.load(json_path)
            # add missing keys
            if 'edge' not in dict:
                dict['edge'] = []
            if 'node' not in dict:
                dict['node'] = []

        return dict

