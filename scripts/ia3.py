from transformers.models.xglm.modeling_xglm import XGLMModel, XGLMDecoderLayer, XGLMAttention
from torch.nn.modules import Sequential, ModuleList
from elementwisemul import ElementwiseMulModule

def modify_xglm_ia3(model:XGLMModel) -> None:
    layers:ModuleList = model.layers
    decoder_layer:XGLMDecoderLayer
    for decoder_layer in layers:
        self_attention_layer:XGLMAttention = decoder_layer.self_attn

        ia3_scaling = ElementwiseMulModule((self_attention_layer.k_proj.out_features, ), 1.)
        self_attention_layer.k_proj = Sequential(self_attention_layer.k_proj, ia3_scaling)

        ia3_scaling = ElementwiseMulModule((self_attention_layer.v_proj.out_features, ), 1.)
        self_attention_layer.v_proj = Sequential(self_attention_layer.v_proj, ia3_scaling)

        ia3_scaling = ElementwiseMulModule((decoder_layer.fc1.out_features, ), 1.)
        decoder_layer.fc1 = Sequential(decoder_layer.fc1, ia3_scaling)