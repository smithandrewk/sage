from torch import nn
import torch
from torch.nn.functional import relu
import copy

class ConvLayerNorm(nn.Module):
    def __init__(self,out_channels) -> None:
        super(ConvLayerNorm,self).__init__()
        self.ln = nn.LayerNorm(out_channels, elementwise_affine=False)

    def forward(self,x):
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        return x
    
class ResBlockv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm='layer',dropout=.05):
        super(ResBlockv2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=False)
        self.ln1 = ConvLayerNorm(out_channels) if norm == 'layer' else nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=False)
        self.ln2 = ConvLayerNorm(out_channels) if norm == 'layer' else nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                ConvLayerNorm(out_channels) if norm == 'layer' else nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.ln1(out)
        out = relu(out)
        out = self.dropout(out) 

        out = self.conv2(out)
        out = self.ln2(out)
        
        shortcut = self.shortcut(x)
        out += shortcut
        out = relu(out)
        return out
    
class ResNetv2(nn.Module):
    def __init__(self,block,widthi=[64],depthi=[2],n_output_neurons=3,norm='batch',stem_kernel_size=7,dropout=.05) -> None:
        super(ResNetv2, self).__init__()
        self.in_channels = widthi[0]
        self.stem = nn.Conv1d(1, widthi[0], kernel_size=stem_kernel_size, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(widthi[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.Sequential(*[self._make_layer(block=block,out_channels=width,blocks=depth,norm=norm,dropout=dropout) for width,depth in zip(widthi,depthi)])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(widthi[-1], n_output_neurons)
        print(sum([p.flatten().size()[0] for p in list(self.parameters())]),'params')
    def _make_layer(self,block,out_channels,blocks=2,norm='batch',dropout=.05):
        layers = []
        layers.append(block(self.in_channels,out_channels,3,2,norm=norm,dropout=dropout))
        self.in_channels = out_channels
        for _ in range(1,blocks):
            layers.append(block(out_channels,out_channels,3,1,norm=norm,dropout=dropout))
        return nn.Sequential(*layers)
    def forward(self,x):
        x = self.stem(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layers(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Dumbledore(nn.Module):
    def __init__(self,encoder_experiment_name,sequence_length,hidden_size=16,num_layers=1,dropout=.1,frozen_encoder=True) -> None:
        super().__init__()
        self.frozen = frozen_encoder
        self.sequence_length = sequence_length
        self.encoder = self.get_encoder(encoder_experiment_name)
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(in_features=hidden_size,out_features=3)
    def forward(self,x):
        x = x.flatten(0,1)
        x = self.encoder(x)
        x = x.reshape(-1,self.sequence_length,3)
        output, (hn, cn) = self.lstm(x)
        x = nn.functional.relu(output[:,-1])
        x = self.classifier(x)
        return x
    def get_encoder(self,encoder_experiment_path):
        state = torch.load(f'{encoder_experiment_path}/state.pt',map_location='cpu',weights_only=False)
        encoder = copy.deepcopy(state['model'])
        encoder.load_state_dict(state['best_model_wts'])
        if self.frozen:
            print("Model is freezing encoder")
            for p in encoder.parameters():
                p.requires_grad = False
        return encoder