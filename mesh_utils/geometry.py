import copy
import logging
import numpy as np
import re


__num_nodes_warn_msg__ = (
    'The number of nodes in your data object can only be inferred by its {} '
    'indices, and hence may result in unexpected batch-wise behavior, e.g., '
    'in case there exists isolated nodes. Please consider explicitly setting '
    'the number of nodes for this data object by assigning it to '
    'data.num_nodes.')


def maybe_num_nodes(index, num_nodes=None):
    return int(index.max()) + 1 if num_nodes is None else num_nodes


def add_self_loops(edge_index, num_nodes=None):
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted, self-loops will be added with edge weights
    denoted by :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (float, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)

    loop_index = np.arange(0, N, dtype=np.int64)
    loop_index = np.expand_dims(loop_index, 1).repeat(2, 1)

    edge_index = np.concatenate([edge_index, loop_index], axis=1)

    return edge_index


def remove_self_loops(edge_index, edge_attr=None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


def contains_self_loops(edge_index):
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    self-loops.

    Args:
        edge_index (LongTensor): The edge indices.

    :rtype: bool
    """
    mask = edge_index[0] == edge_index[1]
    return mask.sum().item() > 0


def contains_isolated_nodes(edge_index, num_nodes=None):
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    isolated nodes.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: bool
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    (row, col), _ = remove_self_loops(edge_index)

    return np.unique(np.concatenate((row, col))).size(0) < num_nodes


def to_undirected(edge_index, num_nodes=None):
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph,
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`LongTensor`
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = np.concatenate([row, col], axis=0), np.concatenate([col, row], axis=0)
    edge_index = np.stack([row, col], dim=0)
    # edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

    return edge_index


def is_undirected(edge_index, edge_attr=None, num_nodes=None):
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` is
    undirected.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: bool
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    # edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
    #                                  num_nodes)

    if edge_attr is None:
        undirected_edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        return edge_index.size(1) == undirected_edge_index.size(1)
    else:
        # edge_index_t, edge_attr_t = np.transpose(edge_index, edge_attr, num_nodes,
        #                                       num_nodes)
        edge_index_t = np.transpose(edge_index)
        edge_attr_t = np.transpose(edge_attr)

        index_symmetric = np.all(edge_index == edge_index_t)
        attr_symmetric = np.all(edge_attr == edge_attr_t)
        return index_symmetric and attr_symmetric


class Data(object):
    r"""A plain old python object modeling a single graph with various
    (optional) attributes:

    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph or node targets with arbitrary shape.
            (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        norm (Tensor, optional): Normal vector matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        face (LongTensor, optional): Face adjacency matrix with shape
            :obj:`[3, num_faces]`. (default: :obj:`None`)

    The data object is not restricted to these attributes and can be extented
    by any other additional data.

    Example::

        data = Data(x=x, edge_index=edge_index)
        data.train_idx = torch.tensor([...], dtype=torch.long)
        data.test_mask = torch.tensor([...], dtype=torch.bool)
    """
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, norm=None, face=None, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        self.norm = norm
        self.face = face
        for key, item in kwargs.items():
            if key == 'num_nodes':
                self.__num_nodes__ = item
            else:
                self[key] = item

        if edge_index is not None and edge_index.dtype != np.int64:
            raise ValueError(
                (f'Argument `edge_index` needs to be of type `numpy.int64` but '
                 f'found type `{edge_index.dtype}`.'))

        if face is not None and face.dtype != np.int64:
            raise ValueError(
                (f'Argument `face` needs to be of type `torch.long` but found '
                 f'type `{face.dtype}`.'))

        # if torch_geometric.is_debug_enabled():
        #     self.debug()

    @classmethod
    def from_dict(cls, dictionary):
        r"""Creates a data object from a python dictionary."""
        data = cls()

        for key, item in dictionary.items():
            data[key] = item

        # if torch_geometric.is_debug_enabled():
        #     data.debug()

        return data

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __len__(self):
        r"""Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __iter__(self):
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    def __cat_dim__(self, key, value):
        r"""Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Concatenate `*index*` and `*face*` attributes in the last dimension.
        if bool(re.search('(index|face)', key)):
            return -1
        # By default, concatenate sparse matrices diagonally.
        # elif isinstance(value, SparseTensor):
        #     return (0, 1)
        return 0

    def __inc__(self, key, value):
        r"""Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Only `*index*` and `*face*` attributes should be cumulatively summed
        # up when creating batches.
        return self.num_nodes if bool(re.search('(index|face)', key)) else 0

    @property
    def num_nodes(self):
        r"""Returns or sets the number of nodes in the graph.

        .. note::
            The number of nodes in your data object is typically automatically
            inferred, *e.g.*, when node features :obj:`x` are present.
            In some cases however, a graph may only be given by its edge
            indices :obj:`edge_index`.
            PyTorch Geometric then *guesses* the number of nodes
            according to :obj:`edge_index.max().item() + 1`, but in case there
            exists isolated nodes, this number has not to be correct and can
            therefore result in unexpected batch-wise behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        if hasattr(self, '__num_nodes__'):
            return self.__num_nodes__
        for key, item in self('x', 'pos', 'norm', 'batch'):
            return item.size(self.__cat_dim__(key, item))
        if hasattr(self, 'adj'):
            return self.adj.size(0)
        if hasattr(self, 'adj_t'):
            return self.adj_t.size(1)
        if self.face is not None:
            logging.warning(__num_nodes_warn_msg__.format('face'))
            return maybe_num_nodes(self.face)
        if self.edge_index is not None:
            logging.warning(__num_nodes_warn_msg__.format('edge'))
            return maybe_num_nodes(self.edge_index)
        return None

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes

    @property
    def num_edges(self):
        r"""Returns the number of edges in the graph."""
        for key, item in self('edge_index', 'edge_attr'):
            return item.size(self.__cat_dim__(key, item))
        return None

    @property
    def num_faces(self):
        r"""Returns the number of faces in the mesh."""
        if self.face is not None:
            return self.face.size(self.__cat_dim__('face', self.face))
        return None

    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the graph."""
        if self.x is None:
            return 0
        return 1 if self.x.dim() == 1 else self.x.size(1)

    @property
    def num_features(self):
        r"""Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self):
        r"""Returns the number of features per edge in the graph."""
        if self.edge_attr is None:
            return 0
        return 1 if self.edge_attr.dim() == 1 else self.edge_attr.size(1)

    # def is_coalesced(self):
    #     r"""Returns :obj:`True`, if edge indices are ordered and do not contain
    #     duplicate entries."""
    #     edge_index, _ = coalesce(self.edge_index, None, self.num_nodes,
    #                              self.num_nodes)
    #     return self.edge_index.numel() == edge_index.numel() and (
    #         self.edge_index != edge_index).sum().item() == 0
    #
    # def coalesce(self):
    #     r""""Orders and removes duplicated entries from edge indices."""
    #     self.edge_index, self.edge_attr = coalesce(self.edge_index,
    #                                                self.edge_attr,
    #                                                self.num_nodes,
    #                                                self.num_nodes)
    #     return self

    def contains_isolated_nodes(self):
        r"""Returns :obj:`True`, if the graph contains isolated nodes."""
        return contains_isolated_nodes(self.edge_index, self.num_nodes)

    def contains_self_loops(self):
        """Returns :obj:`True`, if the graph contains self-loops."""
        return contains_self_loops(self.edge_index)

    def is_undirected(self):
        r"""Returns :obj:`True`, if graph edges are undirected."""
        return is_undirected(self.edge_index, self.edge_attr, self.num_nodes)

    def is_directed(self):
        r"""Returns :obj:`True`, if graph edges are directed."""
        return not self.is_undirected()

    def __apply__(self, item, func):
        return func(np.asarray(item))

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all tensor attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def contiguous(self, *keys):
        r"""Ensures a contiguous memory layout for all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, all present attributes are ensured to
        have a contiguous memory layout."""
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys, **kwargs):
        r"""Performs tensor dtype and/or device conversion to all attributes
        :obj:`*keys`.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.to(device, **kwargs), *keys)

    def clone(self):
        return self.__class__.from_dict({
            k: np.copy(np.asarray(v))
            for k, v in self.__dict__.items()
        })

    def debug(self):
        if self.edge_index is not None:
            if self.edge_index.dtype != np.int64:
                raise RuntimeError(
                    ('Expected edge indices of dtype {}, but found dtype '
                     ' {}').format(np.int64, self.edge_index.dtype))

        if self.face is not None:
            if self.face.dtype != np.int64:
                raise RuntimeError(
                    ('Expected face indices of dtype {}, but found dtype '
                     ' {}').format(np.int64, self.face.dtype))

        if self.edge_index is not None:
            if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
                raise RuntimeError(
                    ('Edge indices should have shape [2, num_edges] but found'
                     ' shape {}').format(self.edge_index.size()))

        if self.edge_index is not None and self.num_nodes is not None:
            if self.edge_index.numel() > 0:
                min_index = self.edge_index.min()
                max_index = self.edge_index.max()
            else:
                min_index = max_index = 0
            if min_index < 0 or max_index > self.num_nodes - 1:
                raise RuntimeError(
                    ('Edge indices must lay in the interval [0, {}]'
                     ' but found them in the interval [{}, {}]').format(
                         self.num_nodes - 1, min_index, max_index))

        if self.face is not None:
            if self.face.dim() != 2 or self.face.size(0) != 3:
                raise RuntimeError(
                    ('Face indices should have shape [3, num_faces] but found'
                     ' shape {}').format(self.face.size()))

        if self.face is not None and self.num_nodes is not None:
            if self.face.numel() > 0:
                min_index = self.face.min()
                max_index = self.face.max()
            else:
                min_index = max_index = 0
            if min_index < 0 or max_index > self.num_nodes - 1:
                raise RuntimeError(
                    ('Face indices must lay in the interval [0, {}]'
                     ' but found them in the interval [{}, {}]').format(
                         self.num_nodes - 1, min_index, max_index))

        if self.edge_index is not None and self.edge_attr is not None:
            if self.edge_index.size(1) != self.edge_attr.size(0):
                raise RuntimeError(
                    ('Edge indices and edge attributes hold a differing '
                     'number of edges, found {} and {}').format(
                         self.edge_index.size(), self.edge_attr.size()))

        if self.x is not None and self.num_nodes is not None:
            if self.x.size(0) != self.num_nodes:
                raise RuntimeError(
                    ('Node features should hold {} elements in the first '
                     'dimension but found {}').format(self.num_nodes,
                                                      self.x.size(0)))

        if self.pos is not None and self.num_nodes is not None:
            if self.pos.size(0) != self.num_nodes:
                raise RuntimeError(
                    ('Node positions should hold {} elements in the first '
                     'dimension but found {}').format(self.num_nodes,
                                                      self.pos.size(0)))

        if self.norm is not None and self.num_nodes is not None:
            if self.norm.size(0) != self.num_nodes:
                raise RuntimeError(
                    ('Node normals should hold {} elements in the first '
                     'dimension but found {}').format(self.num_nodes,
                                                      self.norm.size(0)))

    # def __repr__(self):
    #     cls = str(self.__class__.__name__)
    #     has_dict = any([isinstance(item, dict) for _, item in self])
    #
    #     if not has_dict:
    #         info = [size_repr(key, item) for key, item in self]
    #         return '{}({})'.format(cls, ', '.join(info))
    #     else:
    #         info = [size_repr(key, item, indent=2) for key, item in self]
    #         return '{}(\n{}\n)'.format(cls, ',\n'.join(info))
