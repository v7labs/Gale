import os
import numpy as np
import torch
import shutil
import json
import logging
import pandas as pd
import sys

from torch_geometric.data import InMemoryDataset, Data
from datasets.util.gxl_parser import ParsedGxlDataset


class GxlDataset(InMemoryDataset):
    def __init__(self, root_path, transform=None, pre_transform=None, categorical_features_json=None, rebuild_dataset=True,
                 subset='', ignore_coordinates=False, mean_std=None,
                 disable_feature_norm=False, center_coordinates=False, **kwargs):
        """
        This class reads a IAM dataset in gxl format (tested for AIDS, Fingerprint, Letter)

        Parameters
        ----------
        use_position
        root_path: str
            Path to the dataset folder. There has to be a sub-folder 'data' where the graph gxl files and the train.cxl,
            val.cxl and test.cxl files are.
        transform:
        pre_transform:
        categorical_features : str
            path to json file
            optional parameter: dictionary with first level 'edge' and/or 'node' and then a list of the attribute names
            that need to be one-hot encoded ( = are categorical)
            e.g. {'node': ['symbol'], 'edge': ['valence']}}
        rebuild_dataset: bool
            True if dataset should be re-processed (deletes and re-generates the processes folder).
        subset: str
            'val', 'train' or 'test (or empty) --> name has to match the corresponding cxl file
        mean_std: dict (optional)
            default None. Dictionary containing the mean and std of the node and edge features. If not set, the features
            are z-normalized.

        """
        self.transform = None
        self.target_transform = None
        self.mean_std = mean_std

        self.categorical_features = categorical_features_json
        self.root = root_path
        self.subset = subset
        self.use_position = ignore_coordinates
        self.disable_feature_norm = disable_feature_norm
        self.center_coordinates = center_coordinates

        # should we load the saved dataset from processed or should it be rebuilt
        processed_path = os.path.join(self.root, 'processed')
        if rebuild_dataset and os.path.exists(processed_path):
            # delete the root/processed folder
            shutil.rmtree(processed_path)

        self.name = os.path.basename(root_path)

        super(GxlDataset, self).__init__(self.root, transform, pre_transform)

        # split the dataset and the slices into three subsets
        self.data, self.slices, self.config = torch.load(self.processed_paths[0])

    @property
    def categorical_features(self):
        return self._categorical_features

    @categorical_features.setter
    def categorical_features(self, categorical_features_json):
        """
        dictionary with first level 'edge' and/or 'node' and then a list of the attribute names that should be one-hot
        encoded, e.g. {'node': ['charge'], 'edge': ['valence]}.

        Parameters
        ----------
        categorical_features_json: str
            path to the json file
        """
        if categorical_features_json is None:
            categorical_features = {'node': [], 'edge': []}
        else:
            # read the json file
            logging.info('Loading categorical variables from JSON ({})'.format(categorical_features_json))
            with open('strings.json') as f:
                categorical_features = json.load(categorical_features_json)
            # add missing keys
            if 'edge' not in categorical_features:
                categorical_features['edge'] = []
            if 'node' not in categorical_features:
                categorical_features['node'] = []
        self._categorical_features = categorical_features

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root_path):
        if not os.path.isdir(root_path):
            logging.error(f'Folder {root_path} does not exist.')
            sys.exit(-1)
        self._root = root_path

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
        gxl_dataset = ParsedGxlDataset(path_to_dataset=os.path.join(self.root, 'data'),
                                       categorical_features=self.categorical_features, subset=self.subset,
                                       ignore_coordinates=self.use_position, center_coordinates=self.center_coordinates)

        # make a csv with the number of nodes per graph
        if not os.path.isfile(os.path.join(os.path.dirname(self.processed_paths[0]), 'nb_nodes_edges_per_graph.csv')):
            df = pd.DataFrame.from_records([[g.filename, int(g.nb_of_nodes), int(g.nb_of_edges)] for g in gxl_dataset.graphs],
                                           columns=['filename', 'nb_of_nodes', 'nb_of_edges']).sort_values(by=['filename'])
            df.to_csv(os.path.join(os.path.dirname(self.processed_paths[0]), 'nb_nodes_edges_per_graph.csv'), index=False)

        config = gxl_dataset.config
        data_list = []

        # create the dataset lists: transform the graphs in the GxlDataset into pytorch geometric Data objects
        file_names = gxl_dataset.file_names
        # save the file_names list
        if not os.path.isfile(os.path.join(os.path.dirname(self.processed_paths[0]), 'file_name_list.csv')):
            pd.DataFrame({'file_names': file_names}).to_csv(os.path.join(os.path.dirname(self.processed_paths[0]), 'file_name_list.csv'), index=False)

        if not self.disable_feature_norm and self.mean_std is not None:
            logging.info('Graph features are normalized.')

        for graph in gxl_dataset.graphs:
            # x (Tensor): Node feature matrix with shape :obj:`[num_nodes, num_node_features]`
            # y (Tensor): Graph or node targets with arbitrary shape.
            # edge_index (LongTensor): Graph connectivity in COO format with shape :obj:`[2, num_edges]`
            # edge_attr (Tensor): Edge feature matrix with shape :obj:`[num_edges, num_edge_features]`
            # y (Tensor): Graph or node targets with arbitrary shape

            if not self.disable_feature_norm and self.mean_std is not None:
                graph.normalize(self.mean_std)
            # node
            x = torch.tensor(graph.node_features, dtype=torch.float)
            pos = torch.tensor(graph.node_positions, dtype=torch.float)
            # edges
            edge_index = torch.tensor(graph.edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(graph.edge_features, dtype=torch.float)
            # make graph undirected if necessary
            if gxl_dataset.edge_mode == 'undirected' and len(edge_index) == 2:
                row, col = edge_index
                new_row = torch.cat([row, col], dim=0)
                new_col = torch.cat([col, row], dim=0)
                edge_index = torch.stack([new_row, new_col], dim=0)
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
            # labels
            y = graph.class_label
            # file names
            file_name_ind = file_names.index(graph.filename)
            # make the graph
            g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y, file_name_ind=file_name_ind)
            data_list.append(g)

        # save the data
        data, slices = self.collate(data_list)

        if data.y is not None:
            index, counts = np.unique(np.array(data.y), return_counts=True)
            counts = counts / sum(counts)
            # convert from numpy for json
            config['class_freq'] = ([int(i) for i in index], [float(i) for i in counts])
            config['file_names'] = file_names

        # save the config
        if not os.path.isfile(os.path.join(os.path.dirname(self.processed_paths[0]), 'graphs_config.json')):
            with open(os.path.join(os.path.dirname(self.processed_paths[0]), 'graphs_config.json'), 'w') as fp:
                json.dump(config, fp)

        torch.save((data, slices, config), self.processed_paths[0])

