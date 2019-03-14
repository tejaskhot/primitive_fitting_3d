import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class PointNetfeat(nn.Module):
    def __init__(self, num_points=2500, input_dim=3, feat_size=1024):
        super(PointNetfeat, self).__init__()
        self.feat_size = feat_size
        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.feat_size, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)
        x = self.mp1(x)
        x = x.view(-1, self.feat_size)
        return x

class PointNet(nn.Module):
    def __init__(self, num_points=4096, num_channels=3):
        super(PointNet, self).__init__()
        self.num_points = num_points
        # conv
        self.conv1 = torch.nn.Conv1d(num_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        # max pool
        self.mp1 = torch.nn.MaxPool1d(num_points)
        # fully connected
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 128)
        # conv
        self.conv6 = torch.nn.Conv1d(1152, 512, 1)
        self.conv7 = torch.nn.Conv1d(512, 256, 1)

    def forward(self, x):
        # conv -- bn -- leaky_relu
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        global_feat = x
        # max pool
        x = self.mp1(x).squeeze(-1)
        # FC layers -- bn -- leaky_relu
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        # concat
        x_expand = x.unsqueeze(-1).repeat(1, 1, self.num_points)
        out = torch.cat([global_feat, x_expand],1)
        # conv -- bn -- leaky_relu
        out = F.leaky_relu(self.conv6(out))
        out = F.leaky_relu(self.conv7(out))
        return out

class PrimitiveNet(nn.Module):
    def __init__(self, num_points=2500, input_dim=3, feat_size=1024, hidden_size=256, out_size=10):
        super(PrimitiveNet, self).__init__()
        self.num_points = num_points
        self.input_dim = input_dim
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.feat = PointNetfeat(self.num_points, self.input_dim, self.feat_size)
        self.fc1 = nn.Linear(self.feat_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.out_size)

    def forward(self, x):
        x = self.feat(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MetricModel(nn.Module):
    def __init__(self, num_points, num_channels, feat_size=128):
        super(MetricModel, self).__init__()
        self.pointnet = PointNet(num_points, num_channels)
        # feature convs
        self.conv1 = torch.nn.Conv1d(256, 128, 1)
        self.conv2 = torch.nn.Conv1d(256, feat_size, 1)

    def forward(self, x, seg_only=False):
        # pointnet feature extraction
        x = self.pointnet(x)

        # embedding features
        xfeat1 = self.conv2(x)
        return {'feat': xfeat1.permute(0,2,1)}

class RLNet(nn.Module):
    def __init__(self, num_points=2048, input_dim=3, feat_size=1024, hidden_size=256, out_size=7, num_primitives=3, encoder=None):
        super(RLNet, self).__init__()
        self.num_points = num_points
        self.input_dim = input_dim
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_primitives = num_primitives
        if encoder is not None:
            self.encoder = nn.Sequential(
                PointNetfeat(self.num_points),
                nn.Linear(self.feat_size, 256),
                nn.ReLU(),
                nn.Linear(256, 100),
            )
        else:
            self.encoder = nn.Sequential(
                    PointNetfeat(self.num_points),
                    nn.Linear(self.feat_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, 100),
                )
        self.lstm = nn.LSTMCell(100, 100)
        # self.gru = nn.GRUCell(100, 100)
        self.fc1 = nn.Linear(100, self.out_size)
        self.fc2 = nn.Linear(100, 1)
        #  initialize bias to be high
        # self.fc2.bias.data.fill_(.99)

    def forward(self, x):
        x = self.encoder(x)
        hx, cx = Variable(torch.zeros(x.size(0), 100).cuda()), Variable(torch.zeros(x.size(0), 100).cuda())
        # hx = Variable(torch.zeros(x.size(0), 100).cuda())
        outputs = []
        probs = []
        for i in range(self.num_primitives):
            hx, cx = self.lstm(x, (hx, cx))
            # hx = self.gru(x, hx)
            outputs.append(self.fc1(hx))
            probs.append(torch.sigmoid(self.fc2(hx)))
        return torch.cat((outputs),1), torch.cat((probs),1)

class PrimitiveProbNet(nn.Module):
    def __init__(self, num_points=2500, input_dim=3, feat_size=1024, hidden_size=256, out_size=10, num_primitives=5):
        super(PrimitiveProbNet, self).__init__()
        self.num_points = num_points
        self.input_dim = input_dim
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_primitives = num_primitives
        self.feat = PointNetfeat(self.num_points, self.input_dim, self.feat_size)
        # params
        self.fc1 = nn.Linear(self.feat_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.out_size)
        # probs
        self.fc3 = nn.Linear(self.feat_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.num_primitives)

    def forward(self, x):
        x = self.feat(x)
        probs = torch.sigmoid(self.fc4(F.relu(self.fc3(x))))
        out = self.fc2(F.relu(self.fc1(x)))
        return out, probs

class PointEncoder(nn.Module):
    def __init__(self, num_points=256, feat_size=1024, out_size=10):
        super(PointEncoder, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points=num_points, feat_size=feat_size)
        self.fc1 = nn.Linear(feat_size, 128)
        self.fc2 = nn.Linear(128, out_size)

    def forward(self, x):
        x = self.feat(x)
        x = F.relu((self.fc1(x)))
        x = self.fc2(x)
        return x

class PointDecoder(nn.Module):
    def __init__(self, num_points = 2048, k = 2):
        super(PointDecoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, self.num_points * 3)
        self.th = nn.Tanh()
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.th(self.fc5(x))
        x = x.view(batchsize, 3, self.num_points)
        return x

class PointNetAE(nn.Module):
    def __init__(self, num_points = 2048, k = 2):
        super(PointNetAE, self).__init__()
        self.num_points = num_points
        self.encoder = nn.Sequential(
        PointNetfeat(num_points),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 100),
        )
        self.decoder = PointDecoder(num_points)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    sim_data = Variable(torch.rand(2,3,4096))

    pointfeat = PointNetfeat()
    out = pointfeat(sim_data)
    print('point feat', out.size())

    # test PrimitiveNet
    model = PrimitiveNet(out_size=3*7)
    out = model(sim_data)
    loss = out.mean()
    print('out.size() : ', out.size())
    print(out)
    print(loss)
    loss.backward()
    print('backward pass done')

    print('-'*50)
    # random data
    batch_size = 4
    num_points = 4096
    num_channels = 3
    sim_data = Variable(torch.rand(batch_size,num_channels,num_points).cuda())
    print('MetricModel')
    metric = MetricModel(num_points=num_points, num_channels=num_channels)
    metric.cuda()
    output = metric(sim_data)
    print("output['feat'].shape : {}".format(output['feat'].shape))
    loss = output['feat'].mean()
    loss.backward()

    print('-'*50)
    print('RLNet')
    sim_data = Variable(torch.rand(32,3,4096).cuda(), requires_grad=True)
    model = RLNet(num_points=4096)
    model.cuda()
    outputs, probs = model(sim_data)
    loss = outputs.sum()
    loss.backward()
    print('sim_data.size() : ', sim_data.size())
    print('outputs.size() : ', outputs.size())
    print('probs.size() : ', probs.size())
    print(loss)
    print('backward pass done')

    print('-'*50)
    print('PrimitiveProbNet')
    sim_data = Variable(torch.rand(32,3,4096).cuda(), requires_grad=True)
    model = PrimitiveProbNet(num_points=4096)
    model.cuda()
    outputs, probs = model(sim_data)
    loss = outputs.sum()
    loss.backward()
    print('sim_data.size() : ', sim_data.size())
    print('outputs.size() : ', outputs.size())
    print('probs.size() : ', probs.size())
    print(loss)
    print('backward pass done')