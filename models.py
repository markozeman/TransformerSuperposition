import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear, Dropout, LayerNorm, MultiheadAttention
from torch.nn import functional as F
from math import prod


class Transformer(nn.Module):

    def __init__(self, input_size, num_heads, num_layers, dim_feedforward, num_classes):
        super(Transformer, self).__init__()

        transformer_encoder_layer = TransformerEncoderLayer(input_size, num_heads, dim_feedforward, batch_first=True)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(input_size * 256, input_size),   # 8192 -> 32
            nn.ReLU(),
            nn.Linear(input_size, num_classes),  # 32 -> 2
        )

        # self.init_weights()

    # def init_weights(self):
    #     initrange = 1e-10
    #     self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, key_padding_mask):
        x = self.transformer_encoder(input, src_key_padding_mask=key_padding_mask)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        output = self.mlp(x)   # feed through MLP
        return output


class MyTransformer(nn.Module):

    def __init__(self, input_size, num_heads, num_layers, dim_feedforward, num_classes, use_PSP=False):
        super(MyTransformer, self).__init__()

        if use_PSP:
            # transformer_encoder_layer = MyTransformerEncoderLayerPSP(input_size, num_heads, dim_feedforward, batch_first=True)
            self.transformer_encoder = MyTransformerEncoderLayerPSP(input_size, num_heads, dim_feedforward, batch_first=True)
            # self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers)

            self.linear_1 = nn.Linear(input_size * 256, input_size)
            self.linear_2 = nn.Linear(input_size, num_classes)
            self.trainable_layers = [self.linear_1, self.linear_2]

        else:
            transformer_encoder_layer = MyTransformerEncoderLayer(input_size, num_heads, dim_feedforward, batch_first=True)
            self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers)

            self.mlp = nn.Sequential(
                nn.Linear(input_size * 256, input_size),   # 8192 -> 32
                nn.ReLU(),
                nn.Linear(input_size, num_classes),  # 32 -> 2
            )

    def forward(self, input, key_padding_mask, use_PSP=False, contexts=None, task_id=None):
        if use_PSP:
            x = self.transformer_encoder(input, None, src_key_padding_mask=key_padding_mask, contexts=contexts, task_id=task_id)
            x = torch.flatten(x, start_dim=1, end_dim=2)

            for i, lyr in enumerate(self.trainable_layers):
                context_matrix = torch.from_numpy(np.diag(contexts[task_id][i+4]).astype(np.float32)).cuda()    # i+4, because we already had 4 layers in transformer encoder
                x = torch.matmul(x, context_matrix)
                x = lyr(x)
                if i < len(self.trainable_layers) - 1:  # apply ReLU if it is not the last layer
                    x = nn.functional.relu(x)
            output = x

        else:
            x = self.transformer_encoder(input, mask=None, src_key_padding_mask=key_padding_mask)
            x = torch.flatten(x, start_dim=1, end_dim=2)
            output = self.mlp(x)   # feed through MLP
        return output


class MyTransformerEncoderLayer(nn.Module):

    def __init__(self, input_size, num_heads, dim_feedforward, batch_first, dropout=0.0, activation=F.relu):
        super(MyTransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(input_size, num_heads, dropout=dropout, batch_first=batch_first)

        # implementation of feed-forward model
        self.linear1 = Linear(input_size, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, input_size)

        # # elementwise_affine=False means no trainable parameters (if True, there are 64 trainable parameters)
        # self.norm1 = LayerNorm(input_size, elementwise_affine=False)
        # self.norm2 = LayerNorm(input_size, elementwise_affine=False)

        self.layer_norm = LayerNorm(input_size, elementwise_affine=False)

        self.activation = activation

    def self_attention_block(self, input, key_padding_mask):
        x = self.self_attn(input, input, input, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout(x)

    def ff_block(self, input):
        x = self.linear2(self.dropout(self.activation(self.linear1(input))))
        return self.dropout(x)

    def forward(self, input, src_mask, src_key_padding_mask):
        x = self.layer_norm(input + self.self_attention_block(input, src_key_padding_mask))   # self.norm1
        x = self.layer_norm(x + self.ff_block(x))   # self.norm2
        return x


class MyTransformerEncoderLayerPSP(nn.Module):

    def __init__(self, input_size, num_heads, dim_feedforward, batch_first, dropout=0.0, activation=F.relu):
        super(MyTransformerEncoderLayerPSP, self).__init__()

        self.input_size = input_size

        self.self_attn = MultiheadAttention(input_size, num_heads, dropout=dropout, batch_first=batch_first)

        # implementation of feed-forward model
        self.linear1 = Linear(input_size, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, input_size)

        self.layer_norm = LayerNorm(input_size, elementwise_affine=False)

        self.activation = activation

    def self_attention_block(self, input, key_padding_mask, contexts=None, task_id=None):
        context_matrix = torch.from_numpy(np.diag(contexts[task_id][1][:self.input_size]).astype(np.float32)).cuda()
        x = self.self_attn(torch.matmul(input, context_matrix), torch.matmul(input, context_matrix), torch.matmul(input, context_matrix),
                           key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout(x)

    def ff_block(self, input, contexts=None, task_id=None):
        context_matrix1 = torch.from_numpy(np.diag(contexts[task_id][2]).astype(np.float32)).cuda()
        context_matrix2 = torch.from_numpy(np.diag(contexts[task_id][3]).astype(np.float32)).cuda()
        x = self.linear2(torch.matmul(self.dropout(self.activation(self.linear1(torch.matmul(input, context_matrix1)))), context_matrix2))
        return self.dropout(x)

    def forward(self, input, src_mask, src_key_padding_mask, contexts=None, task_id=None):
        context_matrix = torch.from_numpy(np.diag(contexts[task_id][0][:self.input_size]).astype(np.float32)).cuda()
        x = self.layer_norm(torch.matmul(input, context_matrix) + self.self_attention_block(input, src_key_padding_mask, contexts, task_id))   # self.norm1
        x = self.layer_norm(x + self.ff_block(x, contexts, task_id))   # self.norm2
        return x


class MLP(nn.Module):

    def __init__(self, input_size, num_classes, use_PSP=False, data='NLP'):
        super(MLP, self).__init__()

        self.data = data

        if data == 'NLP':
            input_len = input_size * 256
            hidden_size = 41    # 41 - to match (or slightly increase) number of parameters in transformer model
        elif data == 'CV':
            input_len = prod(input_size)
            hidden_size = 1000

        if use_PSP:
            self.linear_1 = nn.Linear(input_len, hidden_size)
            self.linear_2 = nn.Linear(hidden_size, num_classes)
            self.trainable_layers = [self.linear_1, self.linear_2]
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_len, hidden_size),   # 8192 -> 41 for data=='NLP'
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes),  # 41 -> 2 for data=='NLP'
            )

    def forward(self, input, use_PSP=False, contexts=None, task_id=None):
        if self.data == 'NLP':
            x = torch.flatten(input, start_dim=1, end_dim=2)
        else:   # self.data = 'CV
            x = input

        if use_PSP:
            for i, lyr in enumerate(self.trainable_layers):
                context_matrix = torch.from_numpy(np.diag(contexts[task_id][i]).astype(np.float32)).cuda()
                x = torch.matmul(x, context_matrix)
                x = lyr(x)
                if i < len(self.trainable_layers) - 1:  # apply ReLU if it is not the last layer
                    x = nn.functional.relu(x)
            output = x
        else:
            output = self.mlp(x)

        return output


class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size[0], 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 1323)
        self.fc2 = nn.Linear(1323, num_classes)

    def forward(self, input, use_PSP=False, contexts=None, task_id=None):

        if use_PSP:
            context_matrix_1 = torch.from_numpy(np.reshape(contexts[task_id][0],
                                                           newshape=self.conv1.weight.cpu().size()).astype(np.float32)).cuda()

            context_matrix_2 = torch.from_numpy(np.reshape(contexts[task_id][1],
                                                           newshape=self.conv2.weight.cpu().size()).astype(np.float32)).cuda()

            x = self.pool(F.relu(F.conv2d(input, self.conv1.weight * context_matrix_1, self.conv1.bias)))
            x = self.pool(F.relu(F.conv2d(x, self.conv2.weight * context_matrix_2, self.conv2.bias)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch

            context_matrix_3 = torch.from_numpy(np.diag(contexts[task_id][2]).astype(np.float32)).cuda()
            x = torch.matmul(x, context_matrix_3)
            x = F.relu(self.fc1(x))

            context_matrix_4 = torch.from_numpy(np.diag(contexts[task_id][3]).astype(np.float32)).cuda()
            x = torch.matmul(x, context_matrix_4)
            x = self.fc2(x)
        else:
            x = self.pool(F.relu(self.conv1(input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        return x


class AdapterTransformer(nn.Module):

    def __init__(self, input_size, num_heads, num_layers, dim_feedforward, num_classes, bottleneck_size=16):
        super(AdapterTransformer, self).__init__()

        adapter_transformer_encoder_layer = AdapterTransformerEncoderLayer(input_size, num_heads, dim_feedforward,
                                                                           batch_first=True, bottleneck_size=bottleneck_size)
        self.transformer_encoder = TransformerEncoder(adapter_transformer_encoder_layer, num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(input_size * 256, input_size),   # 8192 -> 32
            nn.ReLU(),
            nn.Linear(input_size, num_classes),  # 32 -> 2
        )

    def forward(self, input, key_padding_mask):
        x = self.transformer_encoder(input, mask=None, src_key_padding_mask=key_padding_mask)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        output = self.mlp(x)   # feed through MLP
        return output


class AdapterTransformerEncoderLayer(nn.Module):

    def __init__(self, input_size, num_heads, dim_feedforward, batch_first, dropout=0.0, activation=F.relu, bottleneck_size=16):
        super(AdapterTransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(input_size, num_heads, dropout=dropout, batch_first=batch_first)

        # implementation of feed-forward model
        self.linear1 = Linear(input_size, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, input_size)

        # # elementwise_affine=False means no trainable parameters (if True, there are 64 trainable parameters)
        # self.norm1 = LayerNorm(input_size, elementwise_affine=False)
        # self.norm2 = LayerNorm(input_size, elementwise_affine=False)

        self.layer_norm = LayerNorm(input_size, elementwise_affine=False)
        self.layer_norm_trainable = LayerNorm(input_size, elementwise_affine=True)

        self.activation = activation

        self.adapter_layer = AdapterLayer(input_size, bottleneck_size)
        # self.adapter_layer = AdapterLayer1to1(input_size)

    def self_attention_block(self, input, key_padding_mask):
        x = self.self_attn(input, input, input, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout(x)

    def ff_block(self, input):
        x = self.linear2(self.dropout(self.activation(self.linear1(input))))
        return self.dropout(x)

    def forward(self, input, src_mask, src_key_padding_mask):
        x = self.layer_norm(input + self.self_attention_block(input, src_key_padding_mask))  # self.norm1
        x = self.layer_norm(x + self.ff_block(x))  # self.norm2
        x = self.layer_norm_trainable(input + self.adapter_layer(x))    # adapter
        return x


class AdapterLayer(nn.Module):

    def __init__(self, input_size, bottleneck_size=16):
        super(AdapterLayer, self).__init__()

        self.bottleneck_mlp = nn.Sequential(
            nn.Linear(input_size, bottleneck_size),  # 32 -> bottleneck_size
            nn.ReLU(),
            nn.Linear(bottleneck_size, input_size),  # bottleneck_size -> 32
        )

    def forward(self, input):
        x = self.bottleneck_mlp(input)
        output = x + input
        return output


class AdapterLayer1to1(nn.Module):

    def __init__(self, input_size):
        super(AdapterLayer1to1, self).__init__()

        self.one2one = nn.Linear(input_size, input_size)
        self.one2one.weight = torch.nn.Parameter(torch.randn(input_size, 1, requires_grad=True), requires_grad=True)
        self.one2one.bias = torch.nn.Parameter(torch.zeros(input_size, requires_grad=False), requires_grad=False)   # bias not used

    def forward(self, input):
        return self.one2one(input)



