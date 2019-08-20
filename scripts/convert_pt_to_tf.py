'''
pip install tensorflow onnx onnx-tf
'''
import argparse

import os
import warnings
import onnx
import torch
import numpy as np
import tensorflow as tf

from typing import List, Tuple

from onnx_tf.backend import prepare
from pytorch_pretrained_bert import BertModel

warnings.filterwarnings("default", category=UserWarning)
warnings.filterwarnings("default", category=FutureWarning)
warnings.filterwarnings("default", category=DeprecationWarning)


class SimpleNN(torch.nn.Module):

    def __init__(self, output_shapes: List[Tuple[int, int]]):
        super(SimpleNN, self).__init__()
        self.scoring_list = torch.nn.ModuleList()
        for (out, hidden) in output_shapes:
            self.scoring_list.append(torch.nn.Linear(hidden, out))

    def forward(self, X):
        outs = []
        for fc in self.scoring_list:
            outs.append(fc(X))
        return outs


def convert_pytorch_checkpoint_to_tf(model:BertModel, save_dir:str):

    """
    :param model:BertModel Pytorch model instance to be converted
    :param ckpt_dir: Tensorflow model directory
    :param model_name: model name
    :return:
    Currently supported HF models:
        Y BertModel
        N BertForMaskedLM
        N BertForPreTraining
        N BertForMultipleChoice
        N BertForNextSentencePrediction
        N BertForSequenceClassification
        N BertForQuestionAnswering
    """

    tensors_to_transpose = (
        "dense.weight",
        "attention.self.query",
        "attention.self.key",
        "attention.self.value"
    )

    var_map = (
        ('layer.', 'layer_'),
        ('word_embeddings.weight', 'word_embeddings'),
        ('position_embeddings.weight', 'position_embeddings'),
        ('token_type_embeddings.weight', 'token_type_embeddings'),
        ('.', '/'),
        ('LayerNorm/weight', 'LayerNorm/gamma'),
        ('LayerNorm/bias', 'LayerNorm/beta'),
        ('weight', 'kernel')
    )

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    state_dict = model.state_dict()

    def to_tf_var_name(name:str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return 'bert/{}'.format(name)

    def create_tf_var(tensor:np.ndarray, name:str, session:tf.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    tf.reset_default_graph()
    with tf.Session() as session:
        for var_name in state_dict:
            tf_name = to_tf_var_name(var_name)
            torch_tensor = state_dict[var_name].numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))

        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(session, os.path.join(save_dir, 'bert_model.ckpt'))


def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


def main(args):
    state_dict = torch.load(args.pytorch_model_path)

    # save fine-tuned bert weights
    bert_state_keys = list(filter(lambda x: x.startswith('bert'), state_dict['state'].keys()))
    bert_state_dict = {k: v for k, v in state_dict['state'].items() if k in bert_state_keys}

    model = BertModel.from_pretrained(
        pretrained_model_name_or_path=args.bert_dir,
        state_dict=bert_state_dict,
    )

    convert_pytorch_checkpoint_to_tf(
        model=model,
        save_dir=args.tf_model_dir
    )

    # save output layers
    output_state_dict = {k: v for k, v in state_dict['state'].items() if k not in bert_state_keys}
    shapes = []
    for k, v in output_state_dict.items():
        if k.endswith('weight'):
            shapes.append(output_state_dict[k].shape)
    model = SimpleNN(shapes)
    model.load_state_dict(output_state_dict)

    dummy_input = torch.from_numpy(np.ones(shapes[0][1]).reshape(1, -1)).float()
    dummy_output = model(dummy_input)
    print(dummy_output)


    torch.onnx.export(
        model,
        dummy_input,
        os.path.join(args.tf_model_dir, 'output_layer.onnx'),
        input_names=['input'],
        output_names=['output']
    )

    model_onnx = onnx.load(os.path.join(args.tf_model_dir, 'output_layer.onnx'))
    tf_rep = prepare(model_onnx)
    tf_rep.export_graph(os.path.join(args.tf_model_dir, 'output_layer.pb'))

    # check saved tf model
    tf_graph = load_pb(os.path.join(args.tf_model_dir, 'output_layer.pb'))
    sess = tf.Session(graph=tf_graph)
    output_tensor = tf_graph.get_tensor_by_name('add:0')
    input_tensor = tf_graph.get_tensor_by_name('input:0')
    output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input})
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_dir",
                        type=str,
                        default=None,
                        required=False,
                        help="Directory containing bert config")
    parser.add_argument("--pytorch_model_path",
                        type=str,
                        required=True,
                        help="/path/to/<pytorch-model-name>.pt")
    parser.add_argument("--tf_model_dir",
                        type=str,
                        required=True,
                        help="dir path to save tensorflow model")
    args = parser.parse_args()
    main(args)
