import argparse

import os
import torch
import numpy as np
import tensorflow as tf
from pytorch_pretrained_bert import BertModel


def convert_pytorch_checkpoint_to_tf(model:BertModel, save_path:str):

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

    save_dir = os.path.dirname(save_path)
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
        saver.save(session, save_path)


def main(args):
    state_dict = torch.load(args.pytorch_model_path)
    bert_state_keys = list(filter(lambda x: x.startswith('bert'), state_dict['state'].keys()))
    bert_state_dict = {k: v for k, v in state_dict.items() if k in bert_state_keys}

    model = BertModel.from_pretrained(
        pretrained_model_name_or_path=args.bert_dir,
        state_dict=bert_state_dict,
    )

    convert_pytorch_checkpoint_to_tf(
        model=model,
        save_path=args.tf_model_path
    )


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
    parser.add_argument("--tf_model_path",
                        type=str,
                        required=True,
                        help="file path to save tensorflow model")
    args = parser.parse_args()
    main(args)
