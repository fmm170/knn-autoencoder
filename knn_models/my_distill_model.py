from torch import nn

class my_ConAE_model(nn.Module):
    def __init__(self, args):
        super(my_ConAE_model, self).__init__()
        #映射hidden_state to 我们需要的维度
        self.hidden_encoder = nn.Linear(args.input_dim, args.output_dim, bias=True)
        # self.pos_example_encoder = nn.Linear(args.input_dim, args.output_dim, bias=True)
    
    def encode_queries(self, query_embeds):
        query_embeds = self.hidden_encoder(query_embeds)
        return query_embeds
    
    def forward(self):
        pass