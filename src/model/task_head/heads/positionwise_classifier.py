from torch import nn

class PositionwiseClassifier(nn.Module):
    # "Implements FFN equation."

    def __init__(self, d_model, d_ff, activation="PReLU", dropout_rate=0.1, d_out=None):
        super(PositionwiseClassifier, self).__init__()
        if d_out is None:
            d_out = d_model
        # By default, bias is on.
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_func = nn.PReLU()

        # self.act_func = nn.LeakyReLU()
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.W_1.weight)
        nn.init.xavier_normal_(self.W_2.weight)

    def forward(self, x):
        x = self.W_2(self.dropout(self.act_func(self.W_1(x))))
        
        return x